"""
swing/factor_evo/factor_store.py - Persist winning factors and IC history to Supabase.

Tables:
  evolved_factors      : approved factor code + metadata + IC results
  factor_ic_history    : per-evaluation IC record (factor_id × date × regime)

Usage:
  from swing.factor_store import save_factor, load_top_factors, record_ic
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))

# IC threshold to consider a factor worth keeping
IC_THRESHOLD     = 0.02
IR_THRESHOLD     = 0.30


# ─────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────

def save_factor(
    name: str,
    code: str,
    ic_result: dict,
    regime: str,
    forward_days: int,
    generation: int = 0,
    parent_name: Optional[str] = None,
) -> Optional[int]:
    """
    Save a winning factor to Supabase evolved_factors table.

    Returns the new factor's id, or None on failure.
    Only saves if IC_test > IC_THRESHOLD and IR_test > IR_THRESHOLD.
    """
    if ic_result.get("status") != "ok":
        return None
    if ic_result.get("ic_mean_test", 0) < IC_THRESHOLD:
        return None
    if ic_result.get("ir_test", 0) < IR_THRESHOLD:
        return None

    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO evolved_factors
              (name, code, regime, forward_days, generation, parent_name,
               ic_mean_train, ic_std_train, ir_train,
               ic_mean_test,  ic_std_test,  ir_test,
               ic_win_rate, n_train_dates, n_test_dates,
               created_date, active)
            VALUES
              (%s, %s, %s, %s, %s, %s,
               %s, %s, %s,
               %s, %s, %s,
               %s, %s, %s,
               %s, TRUE)
            ON CONFLICT (name, regime) DO UPDATE SET
              code           = EXCLUDED.code,
              ic_mean_test   = EXCLUDED.ic_mean_test,
              ir_test        = EXCLUDED.ir_test,
              ic_win_rate    = EXCLUDED.ic_win_rate,
              generation     = EXCLUDED.generation,
              updated_date   = NOW()::date
            RETURNING id
        """, (
            name, code, regime, forward_days, generation, parent_name,
            ic_result.get("ic_mean_train"), ic_result.get("ic_std_train"), ic_result.get("ir_train"),
            ic_result.get("ic_mean_test"),  ic_result.get("ic_std_test"),  ic_result.get("ir_test"),
            ic_result.get("ic_win_rate"),
            ic_result.get("n_train_dates"), ic_result.get("n_test_dates"),
            date.today().isoformat(),
        ))
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        factor_id = row[0] if row else None
        print(f"[factor_store] saved factor '{name}' id={factor_id} "
              f"IC_test={ic_result['ic_mean_test']:+.4f}")
        return factor_id
    except Exception as e:
        print(f"[factor_store] save_factor failed: {e}")
        return None


def record_ic(
    factor_id: int,
    eval_date: str,
    regime: str,
    ic_value: float,
    forward_days: int,
) -> None:
    """Append a single IC observation to factor_ic_history."""
    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO factor_ic_history (factor_id, eval_date, regime, ic_value, forward_days)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (factor_id, eval_date, regime, ic_value, forward_days))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[factor_store] record_ic failed: {e}")


# ─────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────

def load_top_factors(
    regime: str,
    forward_days: int = 5,
    limit: int = 5,
) -> list[dict]:
    """
    Load top active factors for a given regime, sorted by IC_test desc.
    Returns list of dicts with 'name', 'code', 'ic_mean_test', 'ir_test'.
    """
    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT id, name, code, ic_mean_test, ic_std_test, ir_test, ic_win_rate, generation
            FROM evolved_factors
            WHERE regime = %s AND forward_days = %s AND active = TRUE
            ORDER BY ic_mean_test DESC
            LIMIT %s
        """, (regime, forward_days, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "id":           r[0],
                "name":         r[1],
                "code":         r[2],
                "ic_mean_test": r[3],
                "ic_std_test":  r[4],
                "ir_test":      r[5],
                "ic_win_rate":  r[6],
                "generation":   r[7],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[factor_store] load_top_factors failed: {e}")
        return []


def load_ic_history(factor_id: int) -> list[dict]:
    """Return IC time series for a factor (for trend analysis)."""
    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT eval_date, regime, ic_value, forward_days
            FROM factor_ic_history
            WHERE factor_id = %s
            ORDER BY eval_date
        """, (factor_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"eval_date": str(r[0]), "regime": r[1], "ic_value": r[2], "forward_days": r[3]}
            for r in rows
        ]
    except Exception as e:
        print(f"[factor_store] load_ic_history failed: {e}")
        return []


def deactivate_factor(factor_id: int) -> None:
    """Mark a factor inactive (IC degraded in production)."""
    try:
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("UPDATE evolved_factors SET active = FALSE WHERE id = %s", (factor_id,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[factor_store] deactivate_factor failed: {e}")
