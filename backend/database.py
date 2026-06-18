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

# Lazy import — only needed for RAG write paths
def _get_embedding(text: str) -> Optional[list]:
    try:
        from embeddings import get_embedding
        return get_embedding(text)
    except Exception as e:
        print(f"[db] embedding generation failed: {e}")
        return None


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
    """
    Add a trading rule or observation to the knowledge base.
    Generates and stores an embedding for semantic retrieval.
    """
    embedding = _get_embedding(content)
    try:
        conn = get_connection()
        cur  = conn.cursor()
        if embedding is not None:
            cur.execute("""
                INSERT INTO knowledge (content, category, confidence, source, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
            """, (content, category, confidence, source, embedding))
        else:
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
    """Retrieve knowledge base entries ordered by recency (fallback / admin use)."""
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


def search_knowledge_semantic(query: str, limit: int = 8) -> list:
    """
    Retrieve knowledge entries most semantically similar to the query.
    Uses pgvector cosine distance (<->) for ranking.
    Falls back to recency-based get_knowledge if embedding fails.
    """
    query_embedding = _get_embedding(query)
    if query_embedding is None:
        print("[db] semantic search unavailable, falling back to recency")
        return get_knowledge(limit=limit)

    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT content, category, confidence, source,
                   1 - (embedding <-> %s::vector) AS similarity
            FROM knowledge
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return get_knowledge(limit=limit)

        return [
            {
                "content":    r[0],
                "category":   r[1],
                "confidence": r[2],
                "source":     r[3],
                "similarity": round(float(r[4]), 4),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[db] search_knowledge_semantic failed: {e}, falling back to recency")
        return get_knowledge(limit=limit)


# ─────────────────────────────────────────────────────────────
# Swing Trade Phase 2 — decision snapshot writes
# ─────────────────────────────────────────────────────────────

def _build_snapshot_embedding_text(
    ticker: str,
    signal: str,
    confidence: int,
    regime: str,
    search_summary: dict,
) -> str:
    """
    Build a natural-language description of a decision snapshot for embedding.
    This is what gets semantically searched later by Memory Agent.
    """
    catalyst     = search_summary.get("catalyst_type", "UNKNOWN")
    strength     = search_summary.get("catalyst_strength", "UNKNOWN")
    alignment    = search_summary.get("news_alignment", "NEUTRAL")
    summary      = search_summary.get("summary", "")
    return (
        f"{ticker} | {regime} regime | {signal} (conf={confidence}%) | "
        f"catalyst={catalyst} strength={strength} alignment={alignment} | "
        f"{summary}"
    )


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
    Generates and stores an embedding for future semantic retrieval.
    Called from swing/main.py — never from any agent directly.
    """
    embedding_text = _build_snapshot_embedding_text(
        ticker, signal, confidence, regime, search_summary
    )
    embedding = _get_embedding(embedding_text)

    try:
        conn = get_connection()
        cur  = conn.cursor()

        clean_memory = {k: v for k, v in memory_context.items() if k != 'react_trace'}
        trace_json = json.dumps({"trace": react_trace}) if isinstance(react_trace, str) else json.dumps(react_trace)

        if embedding is not None:
            cur.execute("""
                INSERT INTO swing_results (
                    signal_id, ticker, signal, confidence, regime,
                    factors_used, holding_period_days,
                    search_summary, memory_context, react_trace,
                    price_at_scan, scan_date, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, NOW()::date, %s::vector
                )
                ON CONFLICT (signal_id) DO NOTHING
            """, (
                signal_id, ticker, signal, confidence, regime,
                factors_used, holding_period_days,
                json.dumps(search_summary), json.dumps(clean_memory), trace_json,
                price_at_scan, embedding,
            ))
        else:
            cur.execute("""
                INSERT INTO swing_results (
                    signal_id, ticker, signal, confidence, regime,
                    factors_used, holding_period_days,
                    search_summary, memory_context, react_trace,
                    price_at_scan, scan_date
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, NOW()::date
                )
                ON CONFLICT (signal_id) DO NOTHING
            """, (
                signal_id, ticker, signal, confidence, regime,
                factors_used, holding_period_days,
                json.dumps(search_summary), json.dumps(clean_memory), trace_json,
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


# ─────────────────────────────────────────────────────────────
# Swing Trade — semantic case retrieval
# ─────────────────────────────────────────────────────────────

def search_swing_cases_semantic(query: str, limit: int = 5) -> list:
    """
    Retrieve historically similar swing trade decisions via pgvector.
    Used by Memory Agent as a third retrieval layer (concrete past cases).
    Falls back to empty list if embeddings unavailable or table is empty.
    """
    query_embedding = _get_embedding(query)
    if query_embedding is None:
        return []

    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ticker, signal, confidence, regime,
                   search_summary, scan_date, actual_return,
                   1 - (embedding <-> %s::vector) AS similarity
            FROM swing_results
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for r in rows:
            search_summary = r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}")
            results.append({
                "ticker":          r[0],
                "signal":          r[1],
                "confidence":      r[2],
                "regime":          r[3],
                "catalyst_type":   search_summary.get("catalyst_type", "UNKNOWN"),
                "catalyst_summary": search_summary.get("summary", ""),
                "scan_date":       str(r[5]),
                "actual_return":   r[6],
                "similarity":      round(float(r[7]), 4),
            })
        return results

    except Exception as e:
        print(f"[db] search_swing_cases_semantic failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Events — earnings / FDA / macro calendar
# ─────────────────────────────────────────────────────────────

def upsert_events(events: list) -> None:
    """Insert or update upcoming events. Keyed on (ticker, event_type, event_date)."""
    if not events:
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        for ev in events:
            cur.execute("""
                INSERT INTO events (ticker, event_type, event_date, days_away, description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker, event_type, event_date) DO UPDATE
                  SET days_away   = EXCLUDED.days_away,
                      description = EXCLUDED.description,
                      updated_at  = NOW()
            """, (
                ev["ticker"], ev["event_type"], ev["event_date"],
                ev.get("days_away"), ev.get("description", ""),
            ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] upsert_events failed: {e}")


def delete_stale_events() -> None:
    """Remove events whose date has passed."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM events WHERE event_date < NOW()::date")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] delete_stale_events failed: {e}")


def get_upcoming_events(ticker: str, within_days: int = 14) -> list:
    """Return upcoming events for a ticker within the next N days."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT event_type, event_date, days_away, description
            FROM events
            WHERE ticker = %s
              AND event_date >= NOW()::date
              AND event_date <= NOW()::date + INTERVAL '%s days'
            ORDER BY event_date
        """, (ticker, within_days))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "event_type":  r[0],
                "event_date":  str(r[1]),
                "days_away":   r[2],
                "description": r[3],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[db] get_upcoming_events failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Analyst ratings
# ─────────────────────────────────────────────────────────────

def upsert_analyst_ratings(ratings: list) -> None:
    """Insert analyst rating changes with embeddings."""
    if not ratings:
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        for r in ratings:
            embedding = _get_embedding(r["summary"])
            if embedding is not None:
                cur.execute("""
                    INSERT INTO analyst_ratings
                      (ticker, firm, old_rating, new_rating, action, rating_date, summary, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (ticker, firm, rating_date) DO NOTHING
                """, (
                    r["ticker"], r["firm"], r.get("old_rating", ""),
                    r["new_rating"], r.get("action", ""),
                    r["rating_date"], r["summary"], embedding,
                ))
            else:
                cur.execute("""
                    INSERT INTO analyst_ratings
                      (ticker, firm, old_rating, new_rating, action, rating_date, summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, firm, rating_date) DO NOTHING
                """, (
                    r["ticker"], r["firm"], r.get("old_rating", ""),
                    r["new_rating"], r.get("action", ""),
                    r["rating_date"], r["summary"],
                ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] upsert_analyst_ratings failed: {e}")


def search_analyst_ratings(ticker: str, limit: int = 5) -> list:
    """Return recent analyst ratings for a ticker, newest first."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT firm, old_rating, new_rating, action, rating_date, summary
            FROM analyst_ratings
            WHERE ticker = %s
            ORDER BY rating_date DESC
            LIMIT %s
        """, (ticker, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "firm":        r[0],
                "old_rating":  r[1],
                "new_rating":  r[2],
                "action":      r[3],
                "rating_date": str(r[4]),
                "summary":     r[5],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[db] search_analyst_ratings failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# SEC filings
# ─────────────────────────────────────────────────────────────

def upsert_sec_filing(filing: dict) -> None:
    """Insert SEC filing summary with embedding."""
    embedding = _get_embedding(filing.get("embedding_text", filing.get("summary", "")))
    try:
        conn = get_connection()
        cur  = conn.cursor()
        if embedding is not None:
            cur.execute("""
                INSERT INTO sec_filings
                  (ticker, filing_type, filed_date, summary, key_metrics, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (ticker, filing_type, filed_date) DO NOTHING
            """, (
                filing["ticker"], filing["filing_type"], filing["filed_date"],
                filing["summary"], json.dumps(filing.get("key_metrics", {})), embedding,
            ))
        else:
            cur.execute("""
                INSERT INTO sec_filings
                  (ticker, filing_type, filed_date, summary, key_metrics)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker, filing_type, filed_date) DO NOTHING
            """, (
                filing["ticker"], filing["filing_type"], filing["filed_date"],
                filing["summary"], json.dumps(filing.get("key_metrics", {})),
            ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[db] upsert_sec_filing failed: {e}")


def search_sec_filings(ticker: str, limit: int = 3) -> list:
    """Return most recent SEC filings for a ticker."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT filing_type, filed_date, summary, key_metrics
            FROM sec_filings
            WHERE ticker = %s
            ORDER BY filed_date DESC
            LIMIT %s
        """, (ticker, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "filing_type": r[0],
                "filed_date":  str(r[1]),
                "summary":     r[2],
                "key_metrics": r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}"),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[db] search_sec_filings failed: {e}")
        return []