"""
update_outcomes.py - Auto-fill open_return and outcome for today's scan history.

Run this at 10:00 AM ET (7:00 AM LA time) every trading day:
  python update_outcomes.py

What it does:
  1. Load premarket_history.json
  2. Find today's records with open_return = None
  3. Fetch 1-minute bars from yfinance
  4. Compute open_return = (price at 10:00 AM ET - open price) / open price
  5. Set outcome = WIN / LOSS / NEUTRAL based on signal vs actual move
  6. Save back to premarket_history.json
"""

import json
import time
from datetime import datetime, date
from pathlib import Path

import yfinance as yf

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

HISTORY_FILE   = Path(__file__).parent / "premarket_history.json"
LOOKBACK_MIN   = 30    # minutes after open to measure return
MIN_MOVE_WIN   = 2.0   # % minimum move to count as WIN (not noise)


# ─────────────────────────────────────────────────────────────
# Fetch open return for a single ticker
# ─────────────────────────────────────────────────────────────

def get_open_return(ticker: str, trade_date: str) -> dict:
    """
    Fetch 1-minute bars for ticker on trade_date.
    Returns open price, price 30min after open, and return %.

    Args:
        ticker:     stock ticker
        trade_date: "YYYY-MM-DD"

    Returns:
        dict with open_price, price_30min, open_return
        or empty dict if data unavailable
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(
            start=trade_date,
            end=trade_date,
            interval="1m",
            prepost=False,   # regular market hours only
        )

        if hist.empty:
            print(f"  [{ticker}] No intraday data found")
            return {}

        # First bar = open price (9:30 AM ET)
        open_price = float(hist.iloc[0]["Open"])

        # Bar closest to 30 minutes after open
        target_idx = min(LOOKBACK_MIN, len(hist) - 1)
        price_30   = float(hist.iloc[target_idx]["Close"])

        open_return = (price_30 - open_price) / open_price * 100

        return {
            "open_price":  round(open_price, 2),
            "price_30min": round(price_30, 2),
            "open_return": round(open_return, 2),
        }

    except Exception as e:
        print(f"  [{ticker}] Error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Determine outcome based on signal vs actual move
# ─────────────────────────────────────────────────────────────

def determine_outcome(signal: str, open_return: float) -> str:
    """
    Compare LLM signal to actual price movement.

    WIN:     signal was correct
    LOSS:    signal was wrong
    NEUTRAL: move too small to judge (<2%)
    NO_SIGNAL: no LLM signal available (Custom Scan only)

    Signal logic:
      TRADE → expected up → WIN if open_return > +2%
      AVOID → expected down/flat → WIN if open_return < -2%
      WATCH → neutral → WIN if abs(open_return) < 5% (correctly cautious)
      ""    → no signal → NO_SIGNAL
    """
    if not signal:
        # No LLM signal - just record the move, no WIN/LOSS judgment
        if open_return >= MIN_MOVE_WIN:
            return "UP"
        elif open_return <= -MIN_MOVE_WIN:
            return "DOWN"
        else:
            return "FLAT"

    if abs(open_return) < MIN_MOVE_WIN:
        return "NEUTRAL"

    if signal == "TRADE":
        return "WIN" if open_return > MIN_MOVE_WIN else "LOSS"

    if signal == "AVOID":
        return "WIN" if open_return < -MIN_MOVE_WIN else "LOSS"

    if signal == "WATCH":
        # WATCH = cautious, correct if move is moderate
        return "WIN" if abs(open_return) < 5.0 else "NEUTRAL"

    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────
# Main update loop
# ─────────────────────────────────────────────────────────────

def update_outcomes(target_date: str = None) -> None:
    """
    Update open_return and outcome for all records on target_date
    that haven't been filled in yet.

    Args:
        target_date: "YYYY-MM-DD", defaults to today
    """
    if not HISTORY_FILE.exists():
        print("No history file found. Run a scan first.")
        return

    today = target_date or date.today().strftime("%Y-%m-%d")
    print(f"\nUpdating outcomes for {today}...")

    with open(HISTORY_FILE) as f:
        history = json.load(f)

    # Find records that need updating
    to_update = [
        r for r in history
        if r.get("date") == today
        and r.get("open_return") is None
    ]

    if not to_update:
        print(f"No pending records for {today}")
        return

    print(f"Found {len(to_update)} records to update")
    updated = 0

    for record in to_update:
        ticker = record["ticker"]
        print(f"\n  Fetching {ticker}...")

        result = get_open_return(ticker, today)

        if not result:
            print(f"  [{ticker}] Skipping - no data")
            continue

        open_return = result["open_return"]
        signal      = record.get("signal", "")
        outcome     = determine_outcome(signal, open_return)

        # Update the record in-place
        record["open_price"]  = result["open_price"]
        record["price_30min"] = result["price_30min"]
        record["open_return"] = open_return
        record["outcome"]     = outcome

        print(f"  [{ticker}] open_return={open_return:+.1f}%  "
              f"signal={signal or 'none'}  outcome={outcome}")

        updated += 1
        time.sleep(0.3)  # avoid yfinance rate limit

    # Save back
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Updated {updated}/{len(to_update)} records.")
    print(f"History file: {HISTORY_FILE}")

    # Print summary
    print_summary(history, today)


# ─────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────

def print_summary(history: list, target_date: str = None) -> None:
    """Print win rate and average return by catalyst type."""
    records = [
        r for r in history
        if r.get("outcome") is not None
        and (target_date is None or r.get("date") == target_date)
    ]

    if not records:
        return

    print(f"\n{'─'*50}")
    print(f"SUMMARY {'(today)' if target_date else '(all time)'}")
    print(f"{'─'*50}")

    # Overall
    with_signal = [r for r in records if r.get("signal")]
    wins        = [r for r in with_signal if r.get("outcome") == "WIN"]
    losses      = [r for r in with_signal if r.get("outcome") == "LOSS"]

    if with_signal:
        win_rate = len(wins) / len(with_signal) * 100
        avg_ret  = sum(r["open_return"] for r in records if r.get("open_return")) / len(records)
        print(f"Total records:  {len(records)}")
        print(f"With signal:    {len(with_signal)}")
        print(f"Win rate:       {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"Avg open return:{avg_ret:+.1f}%")

    # By catalyst type
    catalysts = set(r.get("catalyst", "UNKNOWN") for r in records)
    if len(catalysts) > 1:
        print(f"\nBy catalyst:")
        for cat in sorted(catalysts):
            cat_records = [r for r in records if r.get("catalyst") == cat]
            cat_wins    = [r for r in cat_records if r.get("outcome") == "WIN"]
            cat_avg     = sum(r["open_return"] for r in cat_records
                             if r.get("open_return") is not None)
            if cat_records:
                cat_avg /= len(cat_records)
                wr = len(cat_wins) / len(cat_records) * 100 if cat_records else 0
                print(f"  {cat:<20} n={len(cat_records):>3}  "
                      f"win={wr:.0f}%  avg={cat_avg:+.1f}%")


# ─────────────────────────────────────────────────────────────
# All-time summary
# ─────────────────────────────────────────────────────────────

def print_all_time_summary() -> None:
    """Print cumulative stats across all recorded history."""
    if not HISTORY_FILE.exists():
        print("No history file found.")
        return

    with open(HISTORY_FILE) as f:
        history = json.load(f)

    print_summary(history, target_date=None)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "summary":
            # python update_outcomes.py summary
            print_all_time_summary()
        else:
            # python update_outcomes.py 2026-05-20
            update_outcomes(target_date=sys.argv[1])
    else:
        # python update_outcomes.py  → update today
        update_outcomes()
