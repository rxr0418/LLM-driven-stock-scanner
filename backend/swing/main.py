"""
main.py - Entry point for the LLM-driven stock scanner.

Orchestrates the full pipeline:
  1. Fetch price data for the universe
  2. Detect market regime (with stability filter)
  3. Run factor-based scan
  4. Fetch news for top candidates
  5. Run LLM analysis
  6. Print and save the daily watchlist

Usage:
  python main.py                    # run full pipeline, top 10 each side
  python main.py --top 5            # top 5 each side
  python main.py --no-llm           # skip LLM analysis (faster, no API cost)
  python main.py --save             # save results to JSON
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Local modules ─────────────────────────────────────────────
from swing.data       import fetch_price_data, fetch_news_batch, UNIVERSE
from swing.regime     import detect_regime
from swing.scanner    import run_scan
from swing.llm_analyst import analyze_watchlist


# ─────────────────────────────────────────────────────────────
# Regime stability filter
# ─────────────────────────────────────────────────────────────

# Simple file-based persistence for regime history
REGIME_HISTORY_FILE = Path("regime_history.json")


def load_regime_history() -> list:
    """Load the last N regime detections from disk."""
    if REGIME_HISTORY_FILE.exists():
        try:
            with open(REGIME_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_regime_history(history: list) -> None:
    """Save regime history to disk (keep last 5 entries)."""
    with open(REGIME_HISTORY_FILE, "w") as f:
        json.dump(history[-5:], f, indent=2)


def stable_regime(current_result: dict, min_streak: int = 2) -> dict:
    """
    Apply a stability filter to prevent rapid regime switching.

    Logic:
      - Load the last N regime detections
      - Only switch to a new regime if it has appeared >= min_streak times
        consecutively (including today)
      - If the streak is not long enough, keep the previous confirmed regime

    Args:
        current_result : output of detect_regime()
        min_streak     : how many consecutive days needed to confirm a switch

    Returns:
        regime result dict (possibly with overridden regime label)
    """
    history = load_regime_history()
    current_regime = current_result["regime"]

    # Add today's detection to history
    history.append({
        "regime":    current_regime,
        "timestamp": current_result["timestamp"],
        "vix":       current_result["vix"],
        "realized_vol": current_result["realized_vol"],
    })
    save_regime_history(history)

    # Check if current regime has been stable for min_streak days
    recent = [h["regime"] for h in history[-min_streak:]]
    if len(recent) < min_streak:
        # Not enough history yet, trust current detection
        return current_result

    if all(r == current_regime for r in recent):
        # Streak confirmed, use current regime
        return current_result
    else:
        # Streak not met, fall back to most recent confirmed regime
        # (the one before the potential switch)
        previous_regime = history[-2]["regime"] if len(history) >= 2 else "NEUTRAL"

        if previous_regime != current_regime:
            print(
                f"[main] Regime stability filter: detected {current_regime} "
                f"but streak < {min_streak} days. "
                f"Keeping previous regime: {previous_regime}"
            )
            # Override regime in result
            overridden = current_result.copy()
            overridden["regime"] = previous_regime
            overridden["description"] = (
                f"[Stability filter active] Detected {current_regime} today "
                f"but keeping {previous_regime} (need {min_streak} consecutive days to switch). "
                + current_result["description"]
            )
            # Update recommended factors to match overridden regime
            if previous_regime == "VOLATILE":
                overridden["recommended_factors"] = [
                    "reversal_5d", "reversal_20d", "vol_adjusted_reversal"
                ]
            elif previous_regime == "TRENDING":
                overridden["recommended_factors"] = ["momentum_20d", "momentum_60d"]
            else:
                overridden["recommended_factors"] = [
                    "momentum_20d", "reversal_5d", "volume_spike"
                ]
            return overridden

        return current_result


# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────

def save_results(watchlist: dict) -> str:
    """Save watchlist to a timestamped JSON file."""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"watchlist_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(watchlist, f, indent=2, ensure_ascii=False)

    return str(filepath)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(top_n: int = 10, use_llm: bool = True, save: bool = False) -> dict:
    """
    Run the full scanner pipeline end-to-end.

    Args:
        top_n   : number of candidates per side (long/short)
        use_llm : whether to run LLM news analysis
        save    : whether to save results to disk

    Returns:
        watchlist dict (or scan_results if use_llm=False)
    """
    print("\n" + "=" * 60)
    print("LLM-DRIVEN STOCK SCANNER")
    print(f"Run time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe : {len(UNIVERSE)} stocks")
    print(f"Top N    : {top_n} per side")
    print(f"LLM      : {'enabled' if use_llm else 'disabled'}")
    print("=" * 60 + "\n")

    # ── Step 1: Price data ────────────────────────────────────
    print("Step 1/5 — Fetching price data...")
    price_data = fetch_price_data(UNIVERSE, lookback_days=90)
    if not price_data or price_data["close"].empty:
        print("[main] ERROR: Failed to fetch price data. Aborting.")
        return {}
    n_days   = price_data["close"].shape[0]
    n_stocks = price_data["close"].shape[1]
    print(f"         Got {n_days} trading days × {n_stocks} stocks\n")

    # ── Step 2: Regime detection ──────────────────────────────
    print("Step 2/5 — Detecting market regime...")
    raw_regime    = detect_regime()
    regime_result = stable_regime(raw_regime, min_streak=2)

    print(f"         Raw regime      : {raw_regime['regime']}")
    print(f"         Stable regime   : {regime_result['regime']}")
    print(f"         VIX             : {regime_result['vix']}")
    print(f"         Realized Vol    : {regime_result['realized_vol']:.2%}")
    print(f"         Trend Strength  : {regime_result['trend_strength']:.0%}")
    print(f"         Recommended     : {regime_result['recommended_factors']}\n")

    # ── Step 3: Factor scan ───────────────────────────────────
    print("Step 3/5 — Running factor scan...")
    scan_results = run_scan(price_data, regime_result, top_n=top_n)
    if "error" in scan_results:
        print(f"[main] ERROR: {scan_results['error']}")
        return {}

    long_tickers  = [x["ticker"] for x in scan_results["long_candidates"]]
    short_tickers = [x["ticker"] for x in scan_results["short_candidates"]]
    print(f"         Long  : {long_tickers}")
    print(f"         Short : {short_tickers}\n")

    # ── Step 4: News fetch ────────────────────────────────────
    print("Step 4/5 — Fetching news...")
    all_candidates = long_tickers + short_tickers
    news_data      = fetch_news_batch(all_candidates, max_articles=5)
    n_with_news    = sum(1 for t in all_candidates if news_data.get(t))
    print(f"         Got news for {n_with_news}/{len(all_candidates)} candidates\n")

    # ── Step 5: LLM analysis ──────────────────────────────────
    if use_llm:
        print("Step 5/5 — Running LLM analysis...")
        watchlist = analyze_watchlist(scan_results, news_data, top_n=top_n)
        print_watchlist(watchlist)
    else:
        print("Step 5/5 — LLM analysis skipped (--no-llm flag)\n")
        # Print factor-only results
        print("\n" + "=" * 60)
        print("FACTOR-ONLY WATCHLIST (no LLM analysis)")
        print("=" * 60)
        print(f"Regime  : {scan_results['regime']}")
        print(f"Factors : {scan_results['factors_used']}")
        print(f"\nLONG candidates:")
        for item in scan_results["long_candidates"]:
            print(f"  {item['ticker']:<8} score={item['score']:.4f}")
        print(f"\nSHORT candidates:")
        for item in scan_results["short_candidates"]:
            print(f"  {item['ticker']:<8} score={item['score']:.4f}")
        print("=" * 60)
        watchlist = scan_results

    # ── Save results ──────────────────────────────────────────
    if save:
        filepath = save_results(watchlist)
        print(f"\n[main] Results saved to {filepath}")

    return watchlist


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-driven stock scanner with regime-adaptive factor selection"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of candidates per side (default: 10)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM analysis (faster, no API cost)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to results/watchlist_TIMESTAMP.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        top_n   = args.top,
        use_llm = not args.no_llm,
        save    = args.save,
    )