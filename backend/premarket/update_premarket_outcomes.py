"""
premarket/update_premarket_outcomes.py - Daily outcome backfill for premarket signals.

Logic:
  - Run every day at 10:00 AM ET (after 30 min of market open)
  - Find today's scan_results with open_return = NULL
  - Fetch 1-minute bars from yfinance
  - Compute open_return = (price at 30min after open - open price) / open price
  - Determine WIN / LOSS / NEUTRAL based on signal vs actual move
  - Write back to Supabase scan_results table

Usage:
  python premarket/update_premarket_outcomes.py
  python premarket/update_premarket_outcomes.py summary
"""

import sys
import time
import warnings
from datetime import date, datetime

import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))
from database import get_connection

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

LOOKBACK_MIN = 30    # minutes after open to measure return
WIN_THRESHOLD  =  2.0  # +2% = WIN for TRADE signal
LOSS_THRESHOLD = -2.0  # -2% = LOSS for TRADE signal


# ─────────────────────────────────────────────────────────────
# Fetch open return
# ─────────────────────────────────────────────────────────────

def get_open_return(ticker: str, trade_date: str) -> dict:
    """
    Fetch 1-minute bars for ticker on trade_date.
    Returns open price, price 30min after open, and return %.
    """
    try:
        hist = yf.Ticker(ticker).history(
            start    = trade_date,
            end      = trade_date,
            interval = "1m",
            prepost  = False,
        )

        if hist.empty:
            print(f"  [{ticker}] No intraday data found")
            return {}

        open_price = float(hist.iloc[0]["Open"])
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
# Determine outcome
# ─────────────────────────────────────────────────────────────

def determine_outcome(signal: str, open_return: float) -> str:
    """
    Compare LLM signal to actual price movement.

    TRADE → expected up → WIN if open_return > +2%
    AVOID → expected down → WIN if open_return < -2%
    WATCH → neutral → WIN if abs(open_return) < 5%
    """
    if abs(open_return) < WIN_THRESHOLD:
        return "NEUTRAL"

    if signal == "TRADE":
        return "WIN" if open_return > WIN_THRESHOLD else "LOSS"
    elif signal == "AVOID":
        return "WIN" if open_return < LOSS_THRESHOLD else "LOSS"
    elif signal == "WATCH":
        return "WIN" if abs(open_return) < 5.0 else "NEUTRAL"

    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────
# Main backfill
# ─────────────────────────────────────────────────────────────

def update_outcomes(target_date: str = None) -> int:
    """
    Backfill open_return and outcome for all unfilled records on target_date.
    """
    today = target_date or date.today().strftime("%Y-%m-%d")
    print(f"\n[outcomes] Updating premarket outcomes for {today}...")

    conn = get_connection()
    cur  = conn.cursor()

    # Find records that need updating
    cur.execute("""
        SELECT id, ticker, signal
        FROM scan_results
        WHERE date = %s
          AND open_return IS NULL
          AND signal IS NOT NULL
    """, (today,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print(f"[outcomes] No pending records for {today}")
        return 0

    print(f"[outcomes] Found {len(rows)} records to update")
    updated = 0

    for row_id, ticker, signal in rows:
        print(f"  Fetching {ticker}...", end=" ")
        result = get_open_return(ticker, today)

        if not result:
            print("skipped")
            continue

        open_return = result["open_return"]
        outcome     = determine_outcome(signal, open_return)

        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute("""
                UPDATE scan_results
                SET open_price  = %s,
                    price_30min = %s,
                    open_return = %s,
                    outcome     = %s
                WHERE id = %s
            """, (
                result["open_price"],
                result["price_30min"],
                open_return,
                outcome,
                row_id,
            ))
            conn.commit()
            cur.close()
            conn.close()

            print(f"{open_return:+.1f}% → {outcome}")
            updated += 1

        except Exception as e:
            print(f"DB update failed: {e}")

        time.sleep(0.3)

    print(f"\n[outcomes] Done. Updated {updated}/{len(rows)} records.")
    return updated


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print catalyst performance summary from Supabase."""
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            SELECT catalyst, sample_size, avg_open_return, win_rate_pct
            FROM catalyst_stats
            ORDER BY sample_size DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            print("[summary] No data yet — keep accumulating!")
            return

        print("\n" + "=" * 60)
        print("PREMARKET CATALYST PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Catalyst':<20} {'N':>4}  {'Win Rate':>8}  {'Avg Return':>10}")
        print("─" * 60)
        for catalyst, n, avg_ret, win_rate in rows:
            print(f"{catalyst:<20} {n:>4}  {win_rate:>7.0f}%  {avg_ret:>+9.1f}%")
        print("=" * 60)

    except Exception as e:
        print(f"[summary] Failed: {e}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary()
    elif len(sys.argv) > 1:
        update_outcomes(target_date=sys.argv[1])
    else:
        update_outcomes()
    print_summary()
