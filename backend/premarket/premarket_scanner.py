"""
premarket/scanner.py - Main premarket scanner orchestrator.

Combines:
  1. Small-cap universe screening
  2. Premarket data fetch (price change, RVOL, volume)
  3. LLM catalyst quality analysis
  4. Final ranking and output

Usage:
  python premarket_scanner.py
  python premarket_scanner.py --no-llm
  python premarket_scanner.py --lang zh
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from premarket.premarket_data     import run_premarket_data_fetch
from premarket.premarket_catalyst import analyze_candidates_batch

# ─────────────────────────────────────────────────────────────
# Scoring and ranking
# ─────────────────────────────────────────────────────────────

SIGNAL_RANK = {"TRADE": 0, "WATCH": 1, "AVOID": 2}
STRENGTH_SCORE = {"STRONG": 3, "MODERATE": 2, "WEAK": 1, "NONE": 0}


def score_candidate(candidate: dict) -> float:
    """
    Compute a composite score for ranking candidates.

    Score = RVOL × catalyst_strength_score × confidence / 100
    Higher = better trade opportunity.
    """
    rvol      = candidate.get("rvol", 0)
    strength  = STRENGTH_SCORE.get(candidate.get("catalyst_strength", "NONE"), 0)
    conf      = candidate.get("confidence", 0) / 100

    return round(rvol * strength * conf, 3)


def rank_candidates(candidates: list) -> list:
    """Sort candidates by signal priority then composite score."""
    for c in candidates:
        c["composite_score"] = score_candidate(c)

    return sorted(
        candidates,
        key=lambda x: (
            SIGNAL_RANK.get(x.get("signal", "AVOID"), 99),
            -x["composite_score"],
        )
    )


# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "TRADE": "🟢",
    "WATCH": "🟡",
    "AVOID": "🔴",
}

STRENGTH_ICONS = {
    "STRONG":   "⚡⚡⚡",
    "MODERATE": "⚡⚡",
    "WEAK":     "⚡",
    "NONE":     "—",
}


def print_results(candidates: list, has_llm: bool = True) -> None:
    """Print formatted premarket scan results."""

    print("\n" + "=" * 70)
    print("PREMARKET SCANNER RESULTS")
    print("=" * 70)
    print(f"Scan time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"Candidates: {len(candidates)}")
    print(f"LLM analysis: {'yes' if has_llm else 'no (factor only)'}")
    print("=" * 70)

    if not candidates:
        print("\nNo stocks found matching criteria.")
        print("Try running closer to market open (9:00-9:30 AM ET)")
        print("or adjust the screening thresholds.")
        return

    for i, c in enumerate(candidates, 1):
        signal   = c.get("signal", "—")
        icon     = SIGNAL_COLORS.get(signal, "⚪")
        strength = c.get("catalyst_strength", "—")
        sicon    = STRENGTH_ICONS.get(strength, "—")

        print(f"\n{icon} #{i} {c['ticker']}")
        print(f"   Price      : ${c.get('premarket_price', 0):.2f} "
              f"({c.get('premarket_change_pct', 0):+.1f}%)")
        print(f"   RVOL       : {c.get('rvol', 0):.1f}x")
        print(f"   Volume     : {c.get('premarket_volume', 0):,}")
        print(f"   Market Cap : ${c.get('market_cap', 0)/1e6:.0f}M")
        print(f"   Float      : {c.get('float', 0)/1e6:.1f}M shares")

        if has_llm:
            print(f"   Signal     : {signal} ({c.get('confidence', 0)}%)")
            print(f"   Catalyst   : {c.get('catalyst_type', '—')} {sicon}")
            print(f"   Reason     : {c.get('reason', '—')}")
            print(f"   Risk       : ⚠ {c.get('risk', '—')}")
            if c.get("news"):
                print(f"   Top news   : {c['news'][0].get('headline', '')[:65]}")
        else:
            if c.get("news"):
                print(f"   Top news   : {c['news'][0].get('headline', '')[:65]}")

    print("\n" + "=" * 70)
    print("DISCLAIMER: For informational purposes only. Not financial advice.")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────

def save_results(candidates: list) -> str:
    """Save scan results to JSON."""
    output_dir = Path("premarket_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"premarket_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    return str(filepath)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_premarket_scan(
    use_llm: bool = True,
    lang: str = "en",
    save: bool = False,
    min_change: float = 5.0,
    max_change: float = 40.0,
    min_rvol: float = 2.0,
    min_volume: int = 100_000,
) -> list:
    """
    Run the full premarket scan pipeline.

    Args:
        use_llm    : whether to run LLM catalyst analysis
        lang       : "en" or "zh"
        save       : whether to save results to disk
        min_change : minimum premarket change % (absolute value)
        max_change : maximum premarket change % (to filter runaway movers)
        min_rvol   : minimum relative volume
        min_volume : minimum premarket share volume

    Returns:
        list of ranked candidate dicts
    """
    print("\n" + "=" * 70)
    print("PREMARKET SCANNER — Small Cap Momentum")
    print(f"Run time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Filters  : change {min_change}-{max_change}% | "
          f"RVOL ≥ {min_rvol} | vol ≥ {min_volume:,}")
    print("=" * 70 + "\n")

    # Step 1: Fetch premarket data
    candidates = run_premarket_data_fetch(
        min_premarket_change=min_change,
        max_premarket_change=max_change,
        min_rvol=min_rvol,
        min_volume=min_volume,
    )

    if not candidates:
        print_results([], has_llm=False)
        return []

    # Step 2: LLM catalyst analysis
    if use_llm and candidates:
        candidates = analyze_candidates_batch(candidates, lang=lang)

    # Step 3: Rank
    candidates = rank_candidates(candidates)

    # Step 4: Print
    print_results(candidates, has_llm=use_llm)

    # Step 5: Save
    if save and candidates:
        filepath = save_results(candidates)
        print(f"\n[scanner] Results saved to {filepath}")

    return candidates


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Small-cap premarket momentum scanner"
    )
    parser.add_argument("--no-llm",     action="store_true", help="Skip LLM analysis")
    parser.add_argument("--lang",       default="en",        help="Output language: en or zh")
    parser.add_argument("--save",       action="store_true", help="Save results to JSON")
    parser.add_argument("--min-change", type=float, default=5.0)
    parser.add_argument("--max-change", type=float, default=40.0)
    parser.add_argument("--min-rvol",   type=float, default=2.0)
    parser.add_argument("--min-volume", type=int,   default=100_000)

    args = parser.parse_args()

    run_premarket_scan(
        use_llm    = not args.no_llm,
        lang       = args.lang,
        save       = args.save,
        min_change = args.min_change,
        max_change = args.max_change,
        min_rvol   = args.min_rvol,
        min_volume = args.min_volume,
    )
