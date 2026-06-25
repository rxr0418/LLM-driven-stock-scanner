"""
backtest_analysis.py - Swing signal performance analysis.

Reads from swing_results and prints:
  1. Overall direction accuracy (5d / 10d)
  2. Signal-level breakdown
  3. Confidence calibration
  4. Pipeline era comparison (old vs orchestrator)
  5. NO_POSITION filter quality

Run:
  cd backend && python3 backtest_analysis.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from database import get_connection

ORCHESTRATOR_START = "2026-06-16"  # first date with Orchestrator pipeline


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"


def run():
    conn = get_connection()
    cur  = conn.cursor()

    # ── 1. Data availability ──────────────────────────────────
    cur.execute("SELECT COUNT(*), MIN(scan_date), MAX(scan_date) FROM swing_results")
    total, d_min, d_max = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM swing_results WHERE return_5d IS NOT NULL")
    filled_5d = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM swing_results WHERE return_10d IS NOT NULL")
    filled_10d = cur.fetchone()[0]

    print(f"\n{'='*60}")
    print("SWING BACKTEST ANALYSIS")
    print(f"{'='*60}")
    print(f"Date range   : {d_min} → {d_max}")
    print(f"Total rows   : {total}")
    print(f"5d filled    : {filled_5d} ({pct(filled_5d, total)})")
    print(f"10d filled   : {filled_10d} ({pct(filled_10d, total)})")

    if filled_5d == 0:
        print("\nNo outcome data yet.")
        return

    # ── 2. Overall direction accuracy ─────────────────────────
    cur.execute("""
        SELECT
          COUNT(*) as n,
          ROUND(AVG(return_5d)::numeric, 2) as avg_5d,
          ROUND(100.0 * SUM(CASE
            WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
              OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
            THEN 1 ELSE 0 END) / COUNT(*), 1) as dir_acc,
          ROUND(100.0 * SUM(CASE WHEN outcome_5d = 'WIN'  THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct,
          ROUND(100.0 * SUM(CASE WHEN outcome_5d = 'LOSS' THEN 1 ELSE 0 END) / COUNT(*), 1) as loss_pct
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('BUY','STRONG_BUY','SHORT','STRONG_SHORT')
          AND return_5d::text != 'NaN'
    """)
    r = cur.fetchone()
    print(f"\n── Overall (5d, n={r[0]}) ────────────────────────────")
    print(f"  Direction accuracy : {r[2]}%")
    print(f"  Avg 5d return      : {r[1]:+.2f}%")
    print(f"  WIN (>2%)          : {r[3]}%")
    print(f"  LOSS (<-2%)        : {r[4]}%")

    # ── 3. Signal breakdown ───────────────────────────────────
    cur.execute("""
        SELECT signal, COUNT(*) as n,
          ROUND(AVG(return_5d)::numeric, 2) as avg_5d,
          ROUND(100.0 * SUM(CASE
            WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
              OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
            THEN 1 ELSE 0 END) / COUNT(*), 1) as dir_acc
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('BUY','STRONG_BUY','SHORT','STRONG_SHORT')
          AND return_5d::text != 'NaN'
        GROUP BY signal ORDER BY n DESC
    """)
    rows = cur.fetchall()
    print(f"\n── By signal ─────────────────────────────────────────")
    print(f"  {'Signal':<15} {'N':>5}  {'Avg5d':>8}  {'DirAcc':>8}")
    for row in rows:
        print(f"  {row[0]:<15} {row[1]:>5}  {float(row[2] or 0):>+7.2f}%  {float(row[3] or 0):>7.1f}%")

    # ── 4. Confidence calibration ─────────────────────────────
    cur.execute("""
        SELECT
          CASE
            WHEN confidence >= 60 THEN '60-100 (high)'
            WHEN confidence >= 30 THEN '30-59 (mid)'
            ELSE '0-29  (low)'
          END as bucket,
          COUNT(*) as n,
          ROUND(AVG(return_5d)::numeric, 2) as avg_5d,
          ROUND(100.0 * SUM(CASE
            WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
              OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
            THEN 1 ELSE 0 END) / COUNT(*), 1) as dir_acc
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('BUY','STRONG_BUY','SHORT','STRONG_SHORT')
          AND return_5d::text != 'NaN'
        GROUP BY bucket ORDER BY bucket
    """)
    rows = cur.fetchall()
    print(f"\n── Confidence calibration (5d) ───────────────────────")
    print(f"  {'Bucket':<16} {'N':>5}  {'Avg5d':>8}  {'DirAcc':>8}")
    for row in rows:
        print(f"  {row[0]:<16} {row[1]:>5}  {float(row[2] or 0):>+7.2f}%  {float(row[3] or 0):>7.1f}%")

    # ── 5. Pipeline era comparison ────────────────────────────
    cur.execute("""
        SELECT
          CASE WHEN scan_date >= %s THEN 'orchestrator' ELSE 'old_pipeline' END as era,
          COUNT(*) as n,
          ROUND(AVG(confidence)::numeric, 1) as avg_conf,
          ROUND(AVG(return_5d)::numeric, 2) as avg_5d,
          ROUND(100.0 * SUM(CASE
            WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
              OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
            THEN 1 ELSE 0 END) / COUNT(*), 1) as dir_acc
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('BUY','STRONG_BUY','SHORT','STRONG_SHORT')
          AND return_5d::text != 'NaN'
        GROUP BY era
    """, (ORCHESTRATOR_START,))
    rows = cur.fetchall()
    print(f"\n── Pipeline era comparison ───────────────────────────")
    print(f"  {'Era':<16} {'N':>5}  {'AvgConf':>8}  {'Avg5d':>8}  {'DirAcc':>8}")
    for row in rows:
        print(f"  {row[0]:<16} {row[1]:>5}  {float(row[2] or 0):>7.1f}%  "
              f"{float(row[3] or 0):>+7.2f}%  {float(row[4] or 0):>7.1f}%")
    print(f"  (orchestrator started {ORCHESTRATOR_START}; need more data for meaningful comparison)")

    # ── 6. NO_POSITION filter quality ─────────────────────────
    cur.execute("""
        SELECT COUNT(*),
          ROUND(AVG(ABS(return_5d))::numeric, 2) as avg_abs_move
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('NO_POSITION', 'NEUTRAL')
    """)
    r = cur.fetchone()
    if r[0]:
        print(f"\n── NO_POSITION / NEUTRAL filter ──────────────────────")
        print(f"  Rows passed    : {r[0]}")
        print(f"  Avg |5d move|  : {r[1]}%  (avoided these moves)")

    # ── 7. Regime breakdown ───────────────────────────────────
    cur.execute("""
        SELECT regime, COUNT(*) as n,
          ROUND(AVG(return_5d)::numeric, 2) as avg_5d,
          ROUND(100.0 * SUM(CASE
            WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
              OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
            THEN 1 ELSE 0 END) / COUNT(*), 1) as dir_acc
        FROM swing_results
        WHERE return_5d IS NOT NULL
          AND signal IN ('BUY','STRONG_BUY','SHORT','STRONG_SHORT')
          AND return_5d::text != 'NaN'
        GROUP BY regime ORDER BY n DESC
    """)
    rows = cur.fetchall()
    print(f"\n── By regime (5d) ────────────────────────────────────")
    print(f"  {'Regime':<12} {'N':>5}  {'Avg5d':>8}  {'DirAcc':>8}")
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:>5}  {float(row[2] or 0):>+7.2f}%  {float(row[3] or 0):>7.1f}%")

    print(f"\n{'='*60}\n")
    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
