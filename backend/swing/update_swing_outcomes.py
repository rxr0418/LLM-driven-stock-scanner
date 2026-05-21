"""
swing/update_swing_outcomes.py - Daily outcome backfill for swing trade signals.

Logic:
  - 每天收盘后运行
  - 查找 5 / 10 / 20 交易日前还没有 outcome 的信号
  - 用 yfinance 拉当时扫描价格和今天的价格
  - 计算收益率，判断 WIN / LOSS / NEUTRAL，写回 Supabase

Usage:
  python swing/update_swing_outcomes.py          # 回填所有未填的
  python swing/update_swing_outcomes.py summary  # 打印统计
"""

import sys
import warnings
from datetime import date, datetime, timedelta

import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))
from database import get_connection

# ─────────────────────────────────────────────────────────────
# 胜负判断阈值
# ─────────────────────────────────────────────────────────────

WIN_THRESHOLD  =  0.02   # +2% 算 WIN
LOSS_THRESHOLD = -0.02   # -2% 算 LOSS
# 中间算 NEUTRAL


def classify_outcome(signal: str, actual_return: float) -> str:
    """
    判断信号是否正确。

    BUY  信号：价格涨超 2% = WIN，跌超 2% = LOSS
    SHORT信号：价格跌超 2% = WIN，涨超 2% = LOSS
    """
    if signal == "BUY":
        if actual_return >= WIN_THRESHOLD:
            return "WIN"
        elif actual_return <= LOSS_THRESHOLD:
            return "LOSS"
        else:
            return "NEUTRAL"
    elif signal == "SHORT":
        if actual_return <= -WIN_THRESHOLD:
            return "WIN"
        elif actual_return >= -LOSS_THRESHOLD:
            return "LOSS"
        else:
            return "NEUTRAL"
    else:
        return "NEUTRAL"


def get_price_on_date(ticker: str, target_date: date) -> float:
    """
    获取某只股票在某天的收盘价。
    target_date 是交易日，如果当天无数据（节假日），
    yfinance 会自动用最近一个交易日。
    """
    try:
        # 拉 target_date 前后各 3 天，确保能取到数据
        start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end)

        if hist.empty:
            return 0.0

        # 找最接近 target_date 且不超过它的收盘价
        hist.index = hist.index.date
        available = [d for d in hist.index if d <= target_date]
        if not available:
            return 0.0

        closest = max(available)
        return round(float(hist.loc[closest, "Close"]), 2)

    except Exception as e:
        print(f"[outcomes] get_price_on_date failed for {ticker}: {e}")
        return 0.0


def fill_outcomes_for_window(window_days: int) -> int:
    """
    回填 window_days 天前的信号结果。

    window_days = 5  → 回填 return_5d / outcome_5d
    window_days = 10 → 回填 return_10d / outcome_10d
    window_days = 20 → 回填 return_20d / outcome_20d
    """
    col_return  = f"return_{window_days}d"
    col_outcome = f"outcome_{window_days}d"

    today      = date.today()
    # 找 window_days 个交易日前（简单用日历天 × 1.4 来估算）
    cutoff = today - timedelta(days=int(window_days * 1.4))

    conn = get_connection()
    cur  = conn.cursor()

    # 找出该窗口还没回填的记录
    cur.execute(f"""
        SELECT id, ticker, signal, score, price_at_scan, scan_date
        FROM swing_results
        WHERE {col_return} IS NULL
          AND scan_date <= %s
          AND signal IN ('BUY', 'SHORT')
    """, (cutoff,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print(f"[outcomes] No unfilled {window_days}d records found")
        return 0

    print(f"[outcomes] Filling {len(rows)} records for {window_days}d window...")
    filled = 0

    for row_id, ticker, signal, score, price_at_scan, scan_date in rows:
        # 目标日期 = 扫描日 + window_days 个交易日
        # 简单估算：window_days × 1.4 日历天
        target_date = scan_date + timedelta(days=int(window_days * 1.4))
        if target_date > today:
            continue  # 还没到时间

        current_price = get_price_on_date(ticker, target_date)
        if current_price == 0:
            continue

        # 如果扫描时没记录价格，用 yfinance 补
        entry_price = price_at_scan
        if not entry_price or entry_price == 0:
            entry_price = get_price_on_date(ticker, scan_date)
        if not entry_price or entry_price == 0:
            continue

        actual_return = (current_price - entry_price) / entry_price
        outcome       = classify_outcome(signal, actual_return)

        # 写回数据库
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute(f"""
                UPDATE swing_results
                SET {col_return}  = %s,
                    {col_outcome} = %s,
                    filled_at     = %s
                WHERE id = %s
            """, (
                round(actual_return * 100, 2),
                outcome,
                datetime.now(),
                row_id,
            ))
            conn.commit()
            cur.close()
            conn.close()
            filled += 1
            print(f"[outcomes] {ticker} {scan_date} → "
                  f"{actual_return:+.1%} ({outcome}) [{window_days}d]")
        except Exception as e:
            print(f"[outcomes] Failed to update {ticker}: {e}")

    return filled


def print_summary() -> None:
    """打印 swing trade 信号的历史统计。"""
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            SELECT regime, signal, sample_size,
                   avg_return_5d, win_rate_5d,
                   avg_return_10d, win_rate_10d
            FROM swing_stats
            ORDER BY sample_size DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            print("[summary] No data yet — keep accumulating!")
            return

        print("\n" + "=" * 65)
        print("SWING TRADE SIGNAL PERFORMANCE SUMMARY")
        print("=" * 65)
        print(f"{'Regime':<12} {'Signal':<8} {'N':>4}  "
              f"{'5d ret':>7} {'5d win':>7}  "
              f"{'10d ret':>7} {'10d win':>7}")
        print("─" * 65)
        for regime, signal, n, r5, w5, r10, w10 in rows:
            print(f"{regime:<12} {signal:<8} {n:>4}  "
                  f"{r5:>+6.1f}% {w5:>6.0f}%  "
                  f"{r10:>+6.1f}% {w10:>6.0f}%")
        print("=" * 65)

    except Exception as e:
        print(f"[summary] Failed: {e}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary()
        sys.exit(0)

    print(f"[outcomes] Running swing outcome backfill — {date.today()}")
    total = 0
    for window in [5, 10, 20]:
        total += fill_outcomes_for_window(window)

    print(f"\n[outcomes] Done. Filled {total} records total.")
    print_summary()
