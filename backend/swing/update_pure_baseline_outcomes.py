"""
swing/update_pure_baseline_outcomes.py - Backfill returns for pure baseline signals.
"""

import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_connection

WIN_THRESHOLD  =  0.02
LOSS_THRESHOLD = -0.02
ACTIVE_SIGNALS = ('BUY', 'STRONG_BUY', 'SHORT', 'STRONG_SHORT')


def classify_outcome(signal: str, ret: float) -> str:
    if signal in ("BUY", "STRONG_BUY"):
        if ret >= WIN_THRESHOLD:   return "WIN"
        if ret <= LOSS_THRESHOLD:  return "LOSS"
    elif signal in ("SHORT", "STRONG_SHORT"):
        if ret <= -WIN_THRESHOLD:  return "WIN"
        if ret >= -LOSS_THRESHOLD: return "LOSS"
    return "NEUTRAL"


def get_price_on_date(ticker: str, target_date: date) -> float:
    try:
        start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            return 0.0
        hist.index = hist.index.date
        available = [d for d in hist.index if d <= target_date]
        return round(float(hist.loc[max(available), "Close"]), 2) if available else 0.0
    except Exception:
        return 0.0


def fill_window(window_days: int) -> int:
    col_ret     = f"return_{window_days}d"
    col_outcome = f"outcome_{window_days}d"
    today       = date.today()
    cutoff      = today - timedelta(days=int(window_days * 1.4))

    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(f"""
        SELECT signal_id, ticker, signal, price_at_scan, scan_date
        FROM swing_results_pure_baseline
        WHERE {col_ret} IS NULL AND scan_date <= %s AND signal IN %s
        ORDER BY scan_date ASC
    """, (cutoff, ACTIVE_SIGNALS))
    rows = cur.fetchall()
    cur.close(); conn.close()

    if not rows:
        print(f"[pure-baseline-outcomes] No unfilled {window_days}d records")
        return 0

    print(f"[pure-baseline-outcomes] Filling {len(rows)} records ({window_days}d)...")
    filled = 0
    for signal_id, ticker, signal, price_at_scan, scan_date in rows:
        target_date = scan_date + timedelta(days=int(window_days * 1.4))
        if target_date > today:
            continue
        current_price = get_price_on_date(ticker, target_date)
        if not current_price:
            continue
        entry_price = price_at_scan or get_price_on_date(ticker, scan_date)
        if not entry_price:
            continue

        ret     = (current_price - entry_price) / entry_price
        outcome = classify_outcome(signal, ret)
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute(f"""
                UPDATE swing_results_pure_baseline
                SET {col_ret} = %s, {col_outcome} = %s, filled_at = %s
                WHERE signal_id = %s
            """, (round(ret * 100, 2), outcome, datetime.now(), signal_id))
            conn.commit(); cur.close(); conn.close()
            filled += 1
            print(f"[pure-baseline-outcomes] {ticker} {scan_date} → {ret:+.1%} ({outcome}) [{window_days}d]")
        except Exception as e:
            print(f"[pure-baseline-outcomes] Failed {signal_id}: {e}")

    return filled


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print(f"[pure-baseline-outcomes] Running — {date.today()}")
    total = sum(fill_window(w) for w in [5, 10])
    print(f"[pure-baseline-outcomes] Done. Filled {total} records.")
