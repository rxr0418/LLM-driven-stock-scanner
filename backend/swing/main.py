"""
swing/main.py - Orchestrator for the Swing Trade scanner.

Phase 1 (unchanged):
  Regime Worker → Factor Worker → top N candidates

Phase 2 (new multi-agent):
  For each candidate (parallel across tickers):
    Search Agent -> Memory Agent -> Skeptic Agent
    → merge()
    → Decision Agent (ReAct)
  → write decision snapshot to Supabase
  → return ranked watchlist

signal_id format: YYYYMMDD_{ticker}_{hex6}
  Unique per scan, used by update_swing_outcomes.py for backfill matching.
"""

import asyncio
import json
import os
import uuid
import warnings
from datetime import date, datetime

warnings.filterwarnings("ignore")

from swing.data import fetch_price_data, fetch_news, fetch_market_overview, UNIVERSE
from swing.regime import detect_regime
from swing.scanner import run_scan
from swing.agents import (
    run_ticker_async,
    estimate_holding_period,
    get_max_candidates,
)
from database import write_decision_snapshot, write_news_evidence


# ─────────────────────────────────────────────────────────────
# signal_id generator
# ─────────────────────────────────────────────────────────────

def make_signal_id(ticker: str) -> str:
    today = date.today().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:6]
    return f"{today}_{ticker}_{suffix}"


# ─────────────────────────────────────────────────────────────
# Phase 2: analyze one candidate (async wrapper)
# ─────────────────────────────────────────────────────────────

async def analyze_candidate(
    ticker: str,
    factor_score: float,
    signal_direction: str,
    regime: str,
    factors_used: list,
) -> dict:
    """
    Run the full orchestrated pipeline for one ticker.
    Orchestrator handles Search → Memory → Skeptic → [recheck] → Decision.
    """
    print(f"\n[phase2] {ticker} ({signal_direction}, score={factor_score:.3f})")

    yahoo_articles = fetch_news(ticker, max_articles=5)

    state, decision = await run_ticker_async(
        ticker=ticker,
        signal_direction=signal_direction,
        factor_score=factor_score,
        regime=regime,
        factors_used=factors_used,
        yahoo_articles=yahoo_articles,
    )

    # Holding period fallback
    agent_hold = decision.get("holding_period_days", 0)
    if agent_hold == 0 and decision.get("signal") not in ("NO_POSITION", "NEUTRAL"):
        decision["holding_period_days"] = estimate_holding_period(
            regime, decision.get("confidence", 0)
        )

    # Attach identifiers and raw state for DB write
    from swing.agents.orchestrator_types import latest_round
    decision["signal_id"]     = make_signal_id(ticker)
    decision["search_summary"] = latest_round(state["search"]) or {}
    decision["memory_context"] = {
        **(latest_round(state["memory"]) or {}),
        "skeptic_review": latest_round(state["skeptic"]) or {},
    }
    decision["skeptic_review"]    = latest_round(state["skeptic"]) or {}
    decision["orchestrator_steps"] = state["steps"]
    decision["recheck_count"]      = state["recheck_count"]
    decision["forced_decision"]    = state["forced_decision"]

    return decision


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────

async def run_full_pipeline(
    top_n: int = 5,
    lang: str = "en",
) -> dict:
    """
    Run the complete swing trade pipeline.

    Phase 1: regime detection + factor scan → top_n candidates per side
    Phase 2: multi-agent analysis for each candidate (parallel)

    Returns watchlist dict compatible with existing API response format.
    """
    print("=" * 60)
    print(f"[swing] Starting full pipeline — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)

    # ── Phase 1 ───────────────────────────────────────────────
    print("\n[phase1] Detecting regime...")
    regime_result = detect_regime()
    regime = regime_result["regime"]
    print(f"[phase1] Regime: {regime}")

    print(f"\n[phase1] Fetching price data for {len(UNIVERSE)} tickers...")
    price_data = fetch_price_data(UNIVERSE, lookback_days=90)

    print("\n[phase1] Running factor scan...")
    # Respect regime-adjusted candidate count
    max_candidates = get_max_candidates(regime)
    scan_results = run_scan(price_data, regime_result, top_n=max_candidates)

    if "error" in scan_results:
        return {"error": scan_results["error"]}

    factors_used = scan_results.get("factors_used", [])
    long_candidates = scan_results.get("long_candidates", [])[:top_n]
    short_candidates = scan_results.get("short_candidates", [])[:top_n]

    print(f"\n[phase1] Top {len(long_candidates)} long, {len(short_candidates)} short candidates selected")

    # ── Phase 2 ───────────────────────────────────────────────
    print("\n[phase2] Starting multi-agent analysis...")

    tasks = []
    for item in long_candidates:
        tasks.append(analyze_candidate(
            ticker=item["ticker"],
            factor_score=item["score"],
            signal_direction="BUY",
            regime=regime,
            factors_used=factors_used,
        ))
    for item in short_candidates:
        tasks.append(analyze_candidate(
            ticker=item["ticker"],
            factor_score=item["score"],
            signal_direction="SHORT",
            regime=regime,
            factors_used=factors_used,
        ))

    decisions = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Process results ───────────────────────────────────────
    long_watchlist = []
    short_watchlist = []
    
    for i, decision in enumerate(decisions):
        if isinstance(decision, Exception):
            print(f"[phase2] Task {i} failed: {decision}")
            continue

        ticker = decision.get("ticker", "UNKNOWN")
        is_long = i < len(long_candidates)

        # Write to Supabase
        # Get price at scan from price_data if available
        try:
            price_at_scan = float(
                price_data["close"][ticker].iloc[-1]
            ) if ticker in price_data.get("close", {}) else 0.0
        except Exception:
            price_at_scan = 0.0

        write_decision_snapshot(
            signal_id=decision["signal_id"],
            ticker=ticker,
            signal=decision.get("signal", "NEUTRAL"),
            confidence=decision.get("confidence", 0),
            regime=regime,
            factors_used=factors_used,
            holding_period_days=decision.get("holding_period_days", 0),
            search_summary=decision.get("search_summary", {}),
            memory_context=decision.get("memory_context", {}),
            react_trace=decision.get("full_react_output", ""),
            price_at_scan=price_at_scan,
        )
        write_news_evidence(
            signal_id=decision["signal_id"],
            ticker=ticker,
            sources=decision.get("search_summary", {}).get("sources", []),
        )

        # Format for watchlist output
        formatted = {
            "ticker":               ticker,
            "factor_score":         decision.get("factor_score", 0),
            "signal":               decision.get("signal", "NEUTRAL"),
            "confidence":           decision.get("confidence", 0),
            "news_alignment":       decision.get("news_alignment", "NEUTRAL"),
            "reason":               decision.get("reason", ""),
            "risk_flag":            decision.get("risk_flag", "none"),
            "holding_period_days":  decision.get("holding_period_days", 0),
            "signal_id":            decision.get("signal_id", ""),
            "skeptic_concern":       decision.get("skeptic_review", {}).get("concern_level", ""),
            "skeptic_summary":       decision.get("skeptic_review", {}).get("summary", ""),
        }

        if is_long:
            long_watchlist.append(formatted)
        else:
            short_watchlist.append(formatted)

    # Sort: NO_POSITION last, then by confidence desc
    def sort_key(x):
        is_pass = x["signal"] in ("NO_POSITION", "NEUTRAL")
        return (is_pass, -x["confidence"])

    long_watchlist.sort(key=sort_key)
    short_watchlist.sort(key=sort_key)

    return {
        "regime":          regime,
        "description":     regime_result.get("description", ""),
        "factors_used":    factors_used,
        "timestamp":       datetime.now().isoformat(),
        "long_watchlist":  long_watchlist,
        "short_watchlist": short_watchlist,
    }


# ─────────────────────────────────────────────────────────────
# Factor scan only (Phase 1, no LLM) — keeps existing API intact
# ─────────────────────────────────────────────────────────────

def run_factor_scan(top_n: int = 10) -> dict:
    """
    Phase 1 only — factor scores without LLM analysis.
    Called by the existing /api/scan endpoint.
    """
    regime_result = detect_regime()
    price_data = fetch_price_data(UNIVERSE, lookback_days=90)
    return run_scan(price_data, regime_result, top_n=top_n)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n",   type=int, default=3)
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--lang",    default="en")
    args = parser.parse_args()

    result = asyncio.run(run_full_pipeline(top_n=args.top_n, lang=args.lang))

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SWING TRADE WATCHLIST")
    print("=" * 60)
    print(f"Regime  : {result['regime']}")
    print(f"Factors : {result['factors_used']}")

    print("\n── LONG ──")
    for item in result["long_watchlist"]:
        hold = f"hold={item['holding_period_days']}d" if item['holding_period_days'] else "skip"
        print(f"  {item['ticker']:<6} {item['signal']:<12} conf={item['confidence']:>3}% "
              f"{hold}  {item['reason']}")

    print("\n── SHORT ──")
    for item in result["short_watchlist"]:
        hold = f"hold={item['holding_period_days']}d" if item['holding_period_days'] else "skip"
        print(f"  {item['ticker']:<6} {item['signal']:<12} conf={item['confidence']:>3}% "
              f"{hold}  {item['reason']}")
