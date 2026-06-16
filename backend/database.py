"""
swing/database.py - Supabase write operations for Swing Trade Phase 2.

All DB writes happen here, called from main.py after Decision Agent returns.
Memory Agent and other agents only READ from the DB — they never write.

Write flow:
  Decision Agent finishes
    → main.py calls write_decision_snapshot()
    → swing_results row inserted with full decision context

update_swing_outcomes.py later backfills:
  return_5d, return_10d, return_20d, outcome_5d, outcome_10d, outcome_20d
  matched by signal_id (unique per scan)
"""

import json
import os

import psycopg2


def get_connection():
    """Get Supabase PostgreSQL connection via connection string."""
    url = os.environ.get("SUPABASE_URL", "")
    if not url:
        raise ValueError("SUPABASE_URL not set")
    return psycopg2.connect(url)


def write_decision_snapshot(
    signal_id: str,
    ticker: str,
    signal: str,
    confidence: int,
    regime: str,
    factors_used: list,
    holding_period_days: int,
    search_summary: dict,
    memory_context: dict,
    react_trace: str,
    price_at_scan: float,
) -> bool:
    """
    Insert one row into swing_results after Decision Agent completes.

    Called from main.py — never from any agent.

    Args:
        signal_id           : unique ID e.g. "20250615_GS_a3f9c1"
        ticker              : stock symbol
        signal              : STRONG_BUY | BUY | NEUTRAL | SHORT | STRONG_SHORT | NO_POSITION
        confidence          : 0–100
        regime              : TRENDING | VOLATILE | NEUTRAL
        factors_used        : list of factor names from scanner
        holding_period_days : recommended hold in trading days (0 = no position)
        search_summary      : full Search Agent output dict
        memory_context      : full Memory Agent output dict
        react_trace         : Decision Agent's raw ReAct text output
        price_at_scan       : closing price on scan date (for outcome backfill)

    Returns:
        True on success, False on failure (non-fatal — pipeline continues).
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 序列化前先处理好
        clean_memory = {k: v for k, v in memory_context.items() if k != 'react_trace'}
        trace_json = json.dumps({"trace": react_trace}) if isinstance(react_trace, str) else json.dumps(react_trace)

        cur.execute("""
            INSERT INTO swing_results (
                signal_id,
                ticker,
                signal,
                confidence,
                regime,
                factors_used,
                holding_period_days,
                search_summary,
                memory_context,
                react_trace,
                price_at_scan,
                scan_date
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, NOW()::date
            )
            ON CONFLICT (signal_id) DO NOTHING
        """, (
            signal_id,
            ticker,
            signal,
            confidence,
            regime,
            factors_used,
            holding_period_days,
            json.dumps(search_summary),
            json.dumps(clean_memory),
            trace_json,
            price_at_scan,
        ))

        conn.commit()
        cur.close()
        conn.close()
        print(f"  [db] wrote snapshot: {signal_id} ({ticker} {signal})")
        return True

    except Exception as e:
        print(f"  [db] write_decision_snapshot failed for {ticker}: {e}")
        return False


def write_news_evidence(
    signal_id: str,
    ticker: str,
    sources: list,
) -> bool:
    """
    Insert news sources used in this decision into swing_news table.
    Enables future evidence tracing: which news led to which outcome.

    Args:
        signal_id : foreign key back to swing_results
        ticker    : stock symbol
        sources   : list of headline strings from Search Agent
    """
    if not sources:
        return True

    try:
        conn = get_connection()
        cur = conn.cursor()

        for source in sources:
            cur.execute("""
                INSERT INTO swing_news (signal_id, ticker, title, source_type)
                VALUES (%s, %s, %s, 'web')
                ON CONFLICT DO NOTHING
            """, (signal_id, ticker, str(source)[:500]))

        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"  [db] write_news_evidence failed for {ticker}: {e}")
        return False