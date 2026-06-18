"""
swing/factor_evo_agent.py - LLM-driven factor evolution agent.

Inspired by CogAlpha (arXiv:2511.18850). Each generation:
  1. GENERATE  : Claude proposes N new factor expressions given regime + IC leaderboard
  2. SANDBOX   : each factor is AST-checked then run in subprocess
  3. EVALUATE  : IC computed on train/test split
  4. SELECT    : keep factors above IC_THRESHOLD + IR_THRESHOLD
  5. MUTATE    : winning factors are passed back to Claude for next generation

Fitness function: Spearman IC on out-of-sample (test) data.
Stopping condition: max_generations reached or IC plateaus.

Usage:
  python swing/factor_evo_agent.py --regime TRENDING --generations 3
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import anthropic

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from config import ANALYST_MODEL

IC_THRESHOLD   = 0.02
IR_THRESHOLD   = 0.30
FACTORS_PER_GEN = 4   # number of new factors to propose per generation


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative researcher designing alpha factors for US equities.

You will be given:
  - The current market regime (TRENDING / VOLATILE / NEUTRAL)
  - The forward holding period (days)
  - The current IC leaderboard: existing factors and their out-of-sample ICs
  - Previous failed factors to avoid

Your job: propose NEW factor expressions that predict cross-sectional stock returns.

RULES:
1. Each factor must be a Python function named `factor_generated(close, volume) -> pd.Series`
2. You may ONLY use: pandas (pd) and numpy (np). No other imports.
3. The function receives close and volume as DataFrames (rows=dates, columns=tickers).
4. Return a pd.Series indexed by ticker (one score per stock). Higher = more bullish.
5. Factors must be computable from close and volume alone (no external data).
6. Prefer economically motivated factors over pure curve-fitting.
7. Output ONLY a JSON array — no markdown, no explanation outside the JSON.

OUTPUT FORMAT:
[
  {
    "name": "short_snake_case_name",
    "hypothesis": "one sentence economic rationale",
    "code": "def factor_generated(close, volume):\\n    ..."
  },
  ...
]

FACTOR IDEAS TO EXPLORE (not exhaustive):
  - Price/volume relationships (accumulation/distribution)
  - Volatility-adjusted momentum or reversal
  - Volume-weighted price moves
  - Cross-sectional z-score of returns
  - Ratio of recent to longer-term momentum (momentum acceleration)
  - Realized volatility spread (short-vol vs long-vol)
  - Amihud illiquidity (|return| / volume)
  - Turnover-normalized reversal"""


def _build_generation_prompt(
    regime: str,
    forward_days: int,
    leaderboard: list[dict],
    failed_names: list[str],
    generation: int,
) -> str:
    if leaderboard:
        board_text = "\n".join(
            f"  {r['name']:<30} IC_test={r['ic_mean_test']:+.4f}  IR={r['ir_test']:.3f}"
            for r in leaderboard
        )
    else:
        board_text = "  (no factors yet — this is generation 0)"

    failed_text = ", ".join(failed_names) if failed_names else "none"

    return f"""REGIME        : {regime}
FORWARD DAYS  : {forward_days}
GENERATION    : {generation}

CURRENT IC LEADERBOARD (out-of-sample):
{board_text}

FAILED / REJECTED THIS RUN: {failed_text}

Propose {FACTORS_PER_GEN} NEW factor expressions that:
  - Are different from everything on the leaderboard and failed list
  - Are motivated by the {regime} regime (e.g. momentum for TRENDING, reversal for VOLATILE)
  - Are likely to have IC > 0.02 on {forward_days}-day forward returns

Output the JSON array now:"""


def _build_mutation_prompt(
    regime: str,
    forward_days: int,
    winners: list[dict],
    generation: int,
) -> str:
    winners_text = "\n".join(
        f"  Name: {w['name']}\n"
        f"  IC_test: {w['ic_mean_test']:+.4f}  IR: {w['ir_test']:.3f}\n"
        f"  Code:\n{w['code']}\n"
        for w in winners
    )
    return f"""REGIME        : {regime}
FORWARD DAYS  : {forward_days}
GENERATION    : {generation} (MUTATION round)

WINNING FACTORS FROM PREVIOUS GENERATION:
{winners_text}

Propose {FACTORS_PER_GEN} MUTATIONS of these winning factors:
  - Change the lookback window
  - Add normalization (z-score, percentile rank)
  - Combine two winning factors
  - Invert or clip outliers

Output the JSON array now:"""


# ─────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────

def _propose_factors(prompt: str, client: anthropic.Anthropic) -> list[dict]:
    """Call Claude and parse the JSON array of proposed factors."""
    response = client.messages.create(
        model=ANALYST_MODEL,
        max_tokens=1500,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Extract JSON array
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  [evo] LLM returned no JSON array")
        return []

    try:
        proposals = json.loads(raw[start:end])
        return [p for p in proposals if "name" in p and "code" in p]
    except json.JSONDecodeError as e:
        print(f"  [evo] JSON parse failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Evaluate one proposal
# ─────────────────────────────────────────────────────────────

def _evaluate_proposal(
    proposal: dict,
    close: "pd.DataFrame",
    volume: "pd.DataFrame",
    forward_days: int,
    train_ratio: float,
) -> tuple[dict, dict]:
    """
    Run sandbox + IC eval for one proposed factor.
    Returns (proposal_with_result, ic_result).
    """
    from swing.sandbox import run_factor_in_sandbox
    from swing.eval_factors import evaluate_generated_factor

    name = proposal["name"]
    code = proposal["code"]

    # Layer 1+2: sandbox
    sandbox_result = run_factor_in_sandbox(code, close, volume)
    if sandbox_result["status"] != "ok":
        print(f"  [evo] {name}: REJECTED by sandbox — {sandbox_result['status']}: "
              f"{sandbox_result.get('violations') or sandbox_result.get('error')}")
        return proposal, {"status": sandbox_result["status"], "ic_mean_all": 0.0}

    # Layer 3: IC evaluation
    ic_result = evaluate_generated_factor(code, close, volume, forward_days, train_ratio)
    proposal["ic_result"] = ic_result

    if ic_result["status"] == "ok":
        print(
            f"  [evo] {name}: "
            f"IC_train={ic_result['ic_mean_train']:+.4f} "
            f"IC_test={ic_result['ic_mean_test']:+.4f} "
            f"IR_test={ic_result['ir_test']:.3f}"
        )
    else:
        print(f"  [evo] {name}: eval error — {ic_result.get('error')}")

    return proposal, ic_result


# ─────────────────────────────────────────────────────────────
# Main evolution loop
# ─────────────────────────────────────────────────────────────

def run_evolution(
    regime: str,
    close: "pd.DataFrame",
    volume: "pd.DataFrame",
    forward_days: int = 5,
    max_generations: int = 3,
    train_ratio: float = 0.7,
    save_to_db: bool = True,
) -> dict:
    """
    Run the full factor evolution loop for one regime.

    Args:
        regime          : "TRENDING" | "VOLATILE" | "NEUTRAL"
        close, volume   : price DataFrames (252+ days for meaningful IC)
        forward_days    : holding period for IC computation
        max_generations : number of generate→evaluate→select cycles
        train_ratio     : fraction of dates for training IC
        save_to_db      : whether to persist winners to evolved_factors

    Returns:
        {
          "regime": ...,
          "generations_run": N,
          "winners": [{name, code, ic_mean_test, ir_test}, ...],
          "all_results": [...],
        }
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Load baseline leaderboard from factor_store
    from swing.factor_store import load_top_factors, save_factor
    leaderboard = load_top_factors(regime, forward_days=forward_days, limit=5)

    all_results = []
    failed_names: list[str] = []
    session_winners: list[dict] = []

    print(f"\n[factor_evo] Regime={regime} forward={forward_days}d generations={max_generations}")
    print(f"[factor_evo] Baseline leaderboard: {len(leaderboard)} factors")

    for gen in range(max_generations):
        print(f"\n── Generation {gen} ──────────────────────────")

        # Choose: generate or mutate
        if gen == 0 or not session_winners:
            prompt = _build_generation_prompt(
                regime, forward_days, leaderboard, failed_names, gen
            )
        else:
            prompt = _build_mutation_prompt(regime, forward_days, session_winners, gen)

        proposals = _propose_factors(prompt, client)
        print(f"  [evo] Proposed {len(proposals)} factors")

        gen_winners = []
        for proposal in proposals:
            name = proposal.get("name", f"gen{gen}_unnamed")
            print(f"\n  Evaluating: {name}")
            print(f"  Hypothesis: {proposal.get('hypothesis', '')}")

            proposal, ic_result = _evaluate_proposal(
                proposal, close, volume, forward_days, train_ratio
            )
            all_results.append({
                "generation": gen,
                "name": name,
                "hypothesis": proposal.get("hypothesis", ""),
                "ic_result": ic_result,
            })

            is_winner = (
                ic_result.get("status") == "ok"
                and ic_result.get("ic_mean_test", 0) >= IC_THRESHOLD
                and ic_result.get("ir_test", 0) >= IR_THRESHOLD
            )

            if is_winner:
                gen_winners.append({
                    "name":         name,
                    "code":         proposal["code"],
                    "ic_mean_test": ic_result["ic_mean_test"],
                    "ir_test":      ic_result["ir_test"],
                    "ic_win_rate":  ic_result.get("ic_win_rate", 0),
                })
                session_winners.append({**proposal, **ic_result})

                if save_to_db:
                    save_factor(
                        name=name,
                        code=proposal["code"],
                        ic_result=ic_result,
                        regime=regime,
                        forward_days=forward_days,
                        generation=gen,
                        parent_name=None,
                    )
            else:
                failed_names.append(name)

        # Update leaderboard with this gen's winners for next round
        leaderboard = sorted(
            leaderboard + gen_winners,
            key=lambda x: -x.get("ic_mean_test", 0)
        )[:5]

        print(f"\n  Generation {gen} summary: {len(gen_winners)}/{len(proposals)} winners")

    return {
        "regime":           regime,
        "forward_days":     forward_days,
        "generations_run":  max_generations,
        "winners":          session_winners,
        "all_results":      all_results,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    sys.path.append(str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    from swing.data import fetch_price_data, UNIVERSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime",      default="TRENDING",
                        choices=["TRENDING", "VOLATILE", "NEUTRAL"])
    parser.add_argument("--forward",     type=int, default=5)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    print(f"Fetching price data ({len(UNIVERSE)} tickers, 252d)...")
    price_data = fetch_price_data(UNIVERSE, lookback_days=252)
    close  = price_data["close"]
    volume = price_data["volume"]
    print(f"Data: {close.shape[0]} days × {close.shape[1]} stocks\n")

    result = run_evolution(
        regime=args.regime,
        close=close,
        volume=volume,
        forward_days=args.forward,
        max_generations=args.generations,
        save_to_db=not args.no_save,
    )

    print("\n" + "=" * 60)
    print(f"EVOLUTION COMPLETE — {result['generations_run']} generations")
    print(f"Winners: {len(result['winners'])}")
    for w in result["winners"]:
        print(f"  {w['name']:<30} IC_test={w['ic_mean_test']:+.4f} IR={w['ir_test']:.3f}")
