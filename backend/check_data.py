"""
check_data.py - Step 1: verify what data exists in swing_results.

Run:
  cd backend && python3 check_data.py
"""

import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from database import get_connection


def run():
    conn = get_connection()
    cur  = conn.cursor()

    # ── 1. Total rows ─────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM swing_results")
    total = cur.fetchone()[0]
    print(f"\nTotal rows in swing_results: {total}")

    if total == 0:
        print("No data yet — system hasn't run in production.")
        return

    # ── 2. return_5d fill rate ────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM swing_results WHERE return_5d IS NOT NULL")
    filled = cur.fetchone()[0]
    print(f"Rows with return_5d filled     : {filled} / {total} ({filled/total*100:.0f}%)")

    # ── 3. Date range ─────────────────────────────────────────
    cur.execute("SELECT MIN(scan_date), MAX(scan_date) FROM swing_results")
    min_d, max_d = cur.fetchone()
    print(f"Date range: {min_d} → {max_d}")

    # ── 4. Signal distribution ────────────────────────────────
    cur.execute("""
        SELECT signal, COUNT(*) as n
        FROM swing_results
        GROUP BY signal ORDER BY n DESC
    """)
    print("\nSignal distribution:")
    for row in cur.fetchall():
        print(f"  {row[0]:<15} {row[1]}")

    # ── 5. Regime distribution ────────────────────────────────
    cur.execute("""
        SELECT regime, COUNT(*) as n
        FROM swing_results
        GROUP BY regime ORDER BY n DESC
    """)
    print("\nRegime distribution:")
    for row in cur.fetchall():
        print(f"  {row[0]:<12} {row[1]}")

    # ── 6. Confidence distribution (bucketed) ─────────────────
    cur.execute("""
        SELECT
          CASE
            WHEN confidence >= 80 THEN '80-100'
            WHEN confidence >= 60 THEN '60-79'
            WHEN confidence >= 40 THEN '40-59'
            WHEN confidence >= 20 THEN '20-39'
            ELSE '0-19'
          END as bucket,
          COUNT(*) as n
        FROM swing_results
        GROUP BY bucket ORDER BY bucket DESC
    """)
    print("\nConfidence distribution:")
    for row in cur.fetchall():
        print(f"  {row[0]:<10} {row[1]}")

    # ── 7. Win rate preview (rows with return_5d) ────────────
    if filled > 0:
        cur.execute("""
            SELECT
              signal,
              COUNT(*) as n,
              ROUND(AVG(return_5d)::numeric, 2) as avg_ret,
              ROUND(
                100.0 * SUM(CASE
                  WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
                    OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
                  THEN 1 ELSE 0 END) / COUNT(*), 1
              ) as dir_acc
            FROM swing_results
            WHERE return_5d IS NOT NULL
              AND signal NOT IN ('NO_POSITION','NEUTRAL')
            GROUP BY signal ORDER BY n DESC
        """)
        rows = cur.fetchall()
        if rows:
            print("\nDirection accuracy by signal (5d return):")
            print(f"  {'Signal':<15} {'N':>4}  {'Avg5dRet':>10}  {'DirAcc':>8}")
            for row in rows:
                print(f"  {row[0]:<15} {row[1]:>4}  {float(row[2] or 0):>+9.2f}%  {float(row[3] or 0):>7.1f}%")

        # Confidence calibration
        cur.execute("""
            SELECT
              CASE
                WHEN confidence >= 60 THEN 'high (60+)'
                WHEN confidence >= 30 THEN 'mid (30-59)'
                ELSE 'low (0-29)'
              END as bucket,
              COUNT(*) as n,
              ROUND(AVG(return_5d)::numeric, 2) as avg_ret,
              ROUND(
                100.0 * SUM(CASE
                  WHEN (signal IN ('BUY','STRONG_BUY') AND return_5d > 0)
                    OR (signal IN ('SHORT','STRONG_SHORT') AND return_5d < 0)
                  THEN 1 ELSE 0 END) / COUNT(*), 1
              ) as dir_acc
            FROM swing_results
            WHERE return_5d IS NOT NULL
              AND signal NOT IN ('NO_POSITION','NEUTRAL')
            GROUP BY bucket ORDER BY bucket
        """)
        rows = cur.fetchall()
        if rows:
            print("\nConfidence calibration (5d):")
            print(f"  {'Bucket':<15} {'N':>4}  {'Avg5dRet':>10}  {'DirAcc':>8}")
            for row in rows:
                print(f"  {row[0]:<15} {row[1]:>4}  {float(row[2] or 0):>+9.2f}%  {float(row[3] or 0):>7.1f}%")

    # ── 8. update_swing_outcomes health check ─────────────────
    cur.execute("""
        SELECT
          scan_date,
          COUNT(*) as total,
          SUM(CASE WHEN return_5d IS NOT NULL THEN 1 ELSE 0 END) as filled_5d
        FROM swing_results
        GROUP BY scan_date
        ORDER BY scan_date DESC
        LIMIT 10
    """)
    print("\nLast 10 scan dates (return_5d backfill status):")
    print(f"  {'Date':<12} {'Total':>6} {'5d_filled':>10}")
    for row in cur.fetchall():
        print(f"  {str(row[0]):<12} {row[1]:>6} {row[2]:>10}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
