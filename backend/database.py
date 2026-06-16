"""
database.py - Supabase access layer for LLM-Driven Stock Scanner.

Handles:
  - Scan result logging (premarket history)
  - News storage
  - Knowledge base (your own rules/observations)
  - Historical context retrieval for RAG
  - Swing Trade Phase 2 decision snapshots

Usage:
  from database import log_scan_results, get_historical_context, add_knowledge
  from database import write_decision_snapshot, write_news_evidence

Setup:
  Set environment variable:
    SUPABASE_URL = postgresql://postgres:password@db.xxx.supabase.co:5432/postgres
"""

import json
import os
import warnings
from datetime import datetime, date
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────

def get_connection():
    """Get a PostgreSQL connection to Supabase."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
    url = os.environ.get("SUPABASE_URL", "")
    if not url:
        raise ValueError("SUPABASE_URL environment variable not set")
    return psycopg2.connect(url)


# ─────────────────────────────────────────────────────────────
# Premarket — scan result logging
# ─────────────────────────────────────────────────────────────

def log_scan_results(candidates: list) -> None:
    """
    Log premarket scan results to Supabase scan_results table.
    Called from api.py after each premarket scan.
    """
    if not candidates:
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        for c in candidates:
            cur.execute("""
                INSERT INTO scan_results (
                    ticker, scan_date, premarket_change_pct,
                    premarket_volume, rvol, market_cap, float,
                    signal, confidence, reason, catalyst_type,
                    entry_timing, risk
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                c.get("ticker"),
                date.today().isoformat(),
                c.get("premarket_change_pct"),
                c.get("premarket_volume"),
                c.get("rvol"),
                c.get("market_cap"),
                c.get("float"),
                c.get("signal"),
                c.get("confidence"),
                c.get("reason"),
                c.get("catalyst_type"),
                c.get("entry_timing"),
                c.get("risk"),
            ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] log_scan_results failed: {e}")


# ─────────────────────────────────────────────────────────────
# Premarket — RAG: historical context retrieval
# ─────────────────────────────────────────────────────────────

def get_historical_context(catalyst_type: str, limit: int = 5) -> list:
    """Retrieve historical scan results for a given catalyst type."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ticker, scan_date, signal, confidence,
                   reason, open_return, outcome
            FROM scan_results
            WHERE catalyst_type = %s
              AND outcome IS NOT NULL
            ORDER BY scan_date DESC
            LIMIT %s
        """, (catalyst_type, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "ticker":      r[0],
                "scan_date":   str(r[1]),
                "signal":      r[2],
                "confidence":  r[3],
                "reason":      r[4],
                "open_return": r[5],
                "outcome":     r[6],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[db] get_historical_context failed: {e}")
        return []


def get_catalyst_stats(catalyst_type: str) -> dict:
    """Get win rate and average return for a catalyst type."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT win_rate, avg_return, sample_size
            FROM catalyst_stats
            WHERE catalyst_type = %s
        """, (catalyst_type,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {"win_rate": row[0], "avg_return": row[1], "sample_size": row[2]}
        return {}
    except Exception as e:
        print(f"[db] get_catalyst_stats failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Premarket — knowledge base
# ─────────────────────────────────────────────────────────────

def add_knowledge(content: str, category: str = "general",
                  confidence: str = "MEDIUM", source: str = "manual") -> bool:
    """Add a trading rule or observation to the knowledge base."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO knowledge (content, category, confidence, source)
            VALUES (%s, %s, %s, %s)
        """, (content, category, confidence, source))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[db] add_knowledge failed: {e}")
        return False


def get_knowledge(category: Optional[str] = None, limit: int = 10) -> list:
    """Retrieve knowledge base entries."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        if category:
            cur.execute("""
                SELECT content, category, confidence, source
                FROM knowledge
                WHERE category = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (category, limit))
        else:
            cur.execute("""
                SELECT content, category, confidence, source
                FROM knowledge
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"content": r[0], "category": r[1],
             "confidence": r[2], "source": r[3]}
            for r in rows
        ]
    except Exception as e:
        print(f"[db] get_knowledge failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Swing Trade Phase 2 — decision snapshot writes
# ─────────────────────────────────────────────────────────────

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
    Insert one decision snapshot into swing_results after Decision Agent completes.
    Called from swing/main.py — never from any agent directly.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

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
        print(f"[db] wrote snapshot: {signal_id} ({ticker} {signal})")
        return True

    except Exception as e:
        print(f"[db] write_decision_snapshot failed for {ticker}: {e}")
        return False


def write_news_evidence(
    signal_id: str,
    ticker: str,
    sources: list,
) -> bool:
    """
    Insert news sources used in this decision into swing_news table.
    """
    if not sources:
        return True
    try:
        conn = get_connection()
        cur  = conn.cursor()
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
        print(f"[db] write_news_evidence failed for {ticker}: {e}")
        return False