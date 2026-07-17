"""
swing/baseline_runner.py - Single-LLM baseline for A/B comparison.

Runs AFTER the factor scan. Takes the same pool of candidates and makes
BUY/SHORT decisions with a single Claude call per ticker (no RAG, no
Skeptic, no multi-step). Saves top-3 BUY + top-3 SHORT to
swing_results_baseline for comparison with the Orchestrator pipeline.

Usage:
  cd backend && python3 swing/baseline_runner.py --top-n 3
  cd backend && python3 swing/baseline_runner.py --top-n 3 --save-db
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYST_MODEL as DECISION_AGENT_MODEL
from logger import get_logger
from resilience import with_retry

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a swing trade analyst. Given a stock's factor score, regime,
and recent news headlines, decide whether to BUY, go SHORT, or take NO_POSITION.

Factor score is 0-1 (higher = stronger momentum/reversal signal in the signal direction).
Regime is TRENDING, VOLATILE, or NEUTRAL.

Respond ONLY with this JSON (no markdown):
{
  "signal": "BUY" | "SHORT" | "NO_POSITION",
  "confidence": 0-100,
  "reason": "one sentence max"
}"""


@with_retry(label="baseline/anthropic")
def _call_llm(client: anthropic.Anthropic, messages: list) -> anthropic.types.Message:
    return client.messages.create(
        model=DECISION_AGENT_MODEL,
        max_tokens=150,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )


def _format_headlines(articles: list) -> str:
    if not articles:
        return "No recent headlines."
    return "\n".join(
        f"  [{a.get('publisher', '')}] {a.get('title', '')}"
        for a in articles[:5]
    )


def score_one(
    client: anthropic.Anthropic,
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    yahoo_articles: list,
) -> dict:
    """
    Run single LLM call for one ticker. Returns:
      {ticker, signal, confidence, reason, prompt_tokens, cost_usd}
    """
    headlines = _format_headlines(yahoo_articles)
    user_msg = (
        f"Ticker: {ticker} | Suggested direction: {signal_direction} | "
        f"Factor score: {factor_score:.3f} | Regime: {regime}\n\n"
        f"Recent headlines:\n{headlines}\n\n"
        f"Should we take this trade?"
    )

    try:
        resp = _call_llm(client, [{"role": "user", "content": user_msg}])
        text = resp.content[0].text.strip()
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        parsed = json.loads(cleaned[start:end]) if start != -1 else {}

        # cost estimate (sonnet input ~$3/1M, output ~$15/1M)
        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost    = in_tok * 3e-6 + out_tok * 15e-6

        return {
            "ticker":        ticker,
            "signal":        parsed.get("signal", "NO_POSITION"),
            "confidence":    int(parsed.get("confidence", 0)),
            "reason":        parsed.get("reason", "")[:500],
            "prompt_tokens": in_tok,
            "cost_usd":      round(cost, 5),
        }
    except Exception as e:
        log.warning("baseline LLM call failed", extra={"ticker": ticker, "error": str(e)})
        return {
            "ticker": ticker, "signal": "NO_POSITION", "confidence": 0,
            "reason": f"error: {e}", "prompt_tokens": 0, "cost_usd": 0.0,
        }


def run_baseline(
    candidates: list[dict],
    top_n: int = 3,
    save_db: bool = False,
    scan_date: date | None = None,
) -> list[dict]:
    """
    Run baseline for a list of scan candidates.

    Args:
        candidates : list of dicts with keys:
                     ticker, signal_direction, factor_score, regime, yahoo_articles
        top_n      : how many BUY + SHORT to keep
        save_db    : write results to swing_results_baseline
        scan_date  : defaults to today

    Returns:
        List of selected decisions (top_n BUY + top_n SHORT).
    """
    scan_date = scan_date or date.today()
    client    = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    log.info("baseline start", extra={"n_candidates": len(candidates), "top_n": top_n})

    results = []
    for c in candidates:
        r = score_one(
            client,
            ticker=c["ticker"],
            signal_direction=c.get("signal_direction", "LONG"),
            factor_score=c.get("factor_score", 0.5),
            regime=c.get("regime", "NEUTRAL"),
            yahoo_articles=c.get("yahoo_articles", []),
        )
        r["regime"]     = c.get("regime", "NEUTRAL")
        r["score"]      = c.get("factor_score", 0.5)
        r["price"]      = c.get("price", None)
        results.append(r)
        log.info("scored", extra={"ticker": r["ticker"], "signal": r["signal"],
                                  "conf": r["confidence"]})

    # Select top-N BUY + top-N SHORT by confidence
    buys   = sorted([r for r in results if r["signal"] in ("BUY", "STRONG_BUY")],
                    key=lambda x: x["confidence"], reverse=True)[:top_n]
    shorts = sorted([r for r in results if r["signal"] in ("SHORT", "STRONG_SHORT")],
                    key=lambda x: x["confidence"], reverse=True)[:top_n]
    selected = buys + shorts

    total_cost = sum(r["cost_usd"] for r in results)
    log.info("baseline done", extra={
        "selected": len(selected), "buys": len(buys), "shorts": len(shorts),
        "total_cost_usd": round(total_cost, 4),
    })

    if save_db and selected:
        _save_to_db(selected, scan_date)

    return selected


def _save_to_db(selected: list[dict], scan_date: date) -> None:
    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()

        for r in selected:
            signal_id = f"BL_{scan_date.strftime('%Y%m%d')}_{r['ticker']}"
            cur.execute("""
                INSERT INTO swing_results_baseline
                  (scan_date, ticker, signal, confidence, regime, score,
                   reason, prompt_tokens, cost_usd, price_at_scan,
                   created_at, signal_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (signal_id) DO NOTHING
            """, (
                scan_date, r["ticker"], r["signal"], r["confidence"],
                r["regime"], r["score"], r["reason"],
                r["prompt_tokens"], r["cost_usd"], r.get("price"),
                datetime.now(), signal_id,
            ))

        conn.commit()
        cur.close()
        conn.close()
        log.info("saved to swing_results_baseline", extra={"n": len(selected)})
    except Exception as e:
        log.error("DB save failed", extra={"error": str(e)})


# ─────────────────────────────────────────────────────────────
# CLI entry point — runs standalone using today's scanner output
# ─────────────────────────────────────────────────────────────

def _build_candidates_from_scan(top_n: int) -> list[dict]:
    """Re-run scanner to get today's candidates (same as swing/main.py)."""
    from swing.scanner import run_scan
    from swing.data import fetch_price_data, UNIVERSE, fetch_news
    from swing.regime import detect_regime

    print("[baseline] Loading price data...")
    price_data = fetch_price_data(UNIVERSE, lookback_days=252)

    print("[baseline] Detecting regime...")
    regime_result = detect_regime()
    regime = regime_result["regime"]
    print(f"[baseline] Regime: {regime}")

    print("[baseline] Running factor scan...")
    scan_output = run_scan(price_data, regime_result, top_n=top_n * 2)

    candidates = []
    for direction in ("long", "short"):
        key = "long_candidates" if direction == "long" else "short_candidates"
        for item in scan_output.get(key, [])[:top_n * 2]:
            ticker  = item["ticker"]
            articles = fetch_news(ticker, max_articles=5)
            candidates.append({
                "ticker":           ticker,
                "signal_direction": "LONG" if direction == "long" else "SHORT",
                "factor_score":     item.get("score", 0.5),
                "regime":           regime,
                "yahoo_articles":   articles,
                "price":            item.get("price"),
            })
    return candidates


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n",   type=int,  default=3)
    parser.add_argument("--save-db", action="store_true")
    args = parser.parse_args()

    candidates = _build_candidates_from_scan(args.top_n)
    selected   = run_baseline(candidates, top_n=args.top_n, save_db=args.save_db)

    print(f"\n{'='*55}")
    print(f"BASELINE DECISIONS  (top {args.top_n} BUY + {args.top_n} SHORT)")
    print(f"{'='*55}")
    total_cost = 0.0
    for r in selected:
        print(f"  {r['ticker']:<6} {r['signal']:<12} conf={r['confidence']:>3}%  {r['reason'][:60]}")
        total_cost += r["cost_usd"]
    print(f"\n  Total cost : ${total_cost:.4f}  ({len(selected)} selected from {len(candidates)} candidates)")
    if args.save_db:
        print("  Saved to swing_results_baseline.")
