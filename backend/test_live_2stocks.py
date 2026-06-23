"""
test_live_2stocks.py - Live end-to-end test with 2 stocks.

Bypasses Phase 1 factor scan, directly feeds 2 tickers into the Orchestrator.
Prints full decision output + per-agent token costs.

Usage:
  cd backend
  python3 test_live_2stocks.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

# ── Token cost tracking ────────────────────────────────────────
# Prices per million tokens (as of June 2025)
PRICES = {
    "claude-sonnet-4-6":        {"input": 3.0,  "output": 15.0, "cache_read": 0.3},
    "claude-haiku-4-5-20251001": {"input": 0.8,  "output": 4.0,  "cache_read": 0.08},
}

token_log: list[dict] = []

def _patch_anthropic():
    """Patch anthropic.resources.Messages.create to log token usage."""
    import anthropic.resources.messages as _msgs_mod

    _orig = _msgs_mod.Messages.create

    def _tracked_create(self, *args, **kwargs):
        t0 = time.time()
        response = _orig(self, *args, **kwargs)
        elapsed = time.time() - t0

        model  = kwargs.get("model", "unknown")
        usage  = response.usage
        prices = PRICES.get(model, {"input": 0, "output": 0, "cache_read": 0})

        input_tokens       = getattr(usage, "input_tokens", 0)
        output_tokens      = getattr(usage, "output_tokens", 0)
        cache_read_tokens  = getattr(usage, "cache_read_input_tokens", 0)
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0)

        billed_input = max(0, input_tokens - cache_read_tokens)
        cost = (
            billed_input      / 1_000_000 * prices["input"] +
            output_tokens     / 1_000_000 * prices["output"] +
            cache_read_tokens / 1_000_000 * prices["cache_read"]
        )

        import traceback
        stack = "".join(traceback.format_stack())
        if "skeptic" in stack:
            caller = "skeptic_agent"
        elif "decision" in stack:
            caller = "decision_agent"
        elif "search" in stack:
            caller = "search_agent"
        else:
            caller = "unknown"

        token_log.append({
            "caller":      caller,
            "model":       model,
            "input":       input_tokens,
            "output":      output_tokens,
            "cache_read":  cache_read_tokens,
            "cache_write": cache_write_tokens,
            "cost_usd":    cost,
            "latency_s":   round(elapsed, 2),
        })
        return response

    _msgs_mod.Messages.create = _tracked_create
    return (_msgs_mod, _orig)


def _restore_anthropic(orig_tuple):
    _msgs_mod, _orig = orig_tuple
    _msgs_mod.Messages.create = _orig


# ── Main test ─────────────────────────────────────────────────

CANDIDATES = [
    {"ticker": "NVDA", "signal_direction": "BUY",   "factor_score": 0.87},
    {"ticker": "INTC", "signal_direction": "SHORT",  "factor_score": 0.13},
]

async def run():
    from swing.agents.orchestrator import run_ticker_async
    from swing.data import fetch_news

    print("=" * 65)
    print("LIVE ORCHESTRATOR TEST — 2 stocks")
    print("=" * 65)

    orig = _patch_anthropic()
    results = []

    try:
        for c in CANDIDATES:
            ticker = c["ticker"]
            print(f"\n{'─'*65}")
            print(f"  {ticker}  ({c['signal_direction']}, score={c['factor_score']})")
            print(f"{'─'*65}")

            yahoo = fetch_news(ticker, max_articles=5)
            t0 = time.time()

            state, decision = await run_ticker_async(
                ticker=ticker,
                signal_direction=c["signal_direction"],
                factor_score=c["factor_score"],
                regime="TRENDING",
                factors_used=["momentum_20d", "volume_spike"],
                yahoo_articles=yahoo,
            )

            elapsed = time.time() - t0

            results.append({
                "ticker":   ticker,
                "state":    state,
                "decision": decision,
                "elapsed":  elapsed,
            })

    finally:
        _restore_anthropic(orig)

    # ── Print results ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DECISIONS")
    print("=" * 65)

    for r in results:
        d = r["decision"]
        s = r["state"]
        print(f"\n  {r['ticker']}")
        print(f"  Signal      : {d.get('signal')}  (conf={d.get('confidence')}%)")
        print(f"  Holding     : {d.get('holding_period_days')}d")
        print(f"  Reason      : {d.get('reason')}")
        print(f"  Risk flag   : {d.get('risk_flag')}")
        print(f"  Recheck cnt : {s['recheck_count']}")
        print(f"  Forced      : {s['forced_decision']}")
        print(f"  Steps       : {' → '.join(s['steps'])}")
        print(f"  Latency     : {r['elapsed']:.1f}s")

    # ── Print token costs ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("TOKEN COSTS")
    print("=" * 65)
    print(f"  {'Agent':<18} {'Model':<28} {'In':>6} {'CacheR':>7} {'Out':>6}  {'Cost':>9}  {'ms':>6}")
    print(f"  {'-'*18} {'-'*28} {'-'*6} {'-'*7} {'-'*6}  {'-'*9}  {'-'*6}")

    total_cost = 0.0
    for entry in token_log:
        total_cost += entry["cost_usd"]
        model_short = entry["model"].replace("claude-", "").replace("-20251001", "")
        print(
            f"  {entry['caller']:<18} {model_short:<28} "
            f"{entry['input']:>6} {entry['cache_read']:>7} {entry['output']:>6}  "
            f"${entry['cost_usd']:>8.5f}  {entry['latency_s']*1000:>5.0f}ms"
        )

    print(f"\n  {'TOTAL':<47} ${total_cost:.5f}")
    print(f"  Per stock: ${total_cost/len(CANDIDATES):.5f}")
    print(f"  Calls: {len(token_log)}")


if __name__ == "__main__":
    asyncio.run(run())
