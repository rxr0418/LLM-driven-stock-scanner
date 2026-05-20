"""
db.py - Database access layer for Supabase PostgreSQL.

Handles:
  - Scan result logging (premarket history)
  - News storage
  - Knowledge base (your own rules/observations)
  - Historical context retrieval for RAG

Usage:
  from db import log_scan_results, get_historical_context, add_knowledge

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
# Scan result logging
# ─────────────────────────────────────────────────────────────

def log_scan_results(candidates: list) -> int:
    """
    Log premarket scan results to scan_results table.

    Args:
        candidates: list of candidate dicts from premarket scanner

    Returns:
        number of records inserted
    """
    if not candidates:
        return 0

    conn = get_connection()
    cur  = conn.cursor()

    today    = date.today().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M ET")
    inserted = 0

    for c in candidates:
        try:
            cur.execute("""
                INSERT INTO scan_results (
                    date, time, ticker, change_pct, rvol, volume,
                    market_cap, float_shares, signal, confidence,
                    catalyst, catalyst_strength, proportionality,
                    manipulation_risk, reason, risk, entry_timing,
                    news_headline
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s
                )
            """, (
                today,
                now_time,
                c.get("ticker"),
                c.get("premarket_change_pct"),
                c.get("rvol"),
                c.get("premarket_volume"),
                c.get("market_cap"),
                c.get("float"),
                c.get("signal", ""),
                c.get("confidence", 0),
                c.get("catalyst_type", "UNKNOWN"),
                c.get("catalyst_strength", ""),
                c.get("proportionality", ""),
                c.get("manipulation_risk", ""),
                c.get("reason", ""),
                c.get("risk", ""),
                c.get("entry_timing", ""),
                c["news"][0]["headline"] if c.get("news") else "",
            ))
            inserted += 1
        except Exception as e:
            print(f"[db] Failed to insert {c.get('ticker')}: {e}")
            continue

    conn.commit()
    cur.close()
    conn.close()

    print(f"[db] Logged {inserted} scan results to Supabase")
    return inserted


# ─────────────────────────────────────────────────────────────
# Update outcomes (called by update_outcomes.py)
# ─────────────────────────────────────────────────────────────

def update_outcome(
    ticker: str,
    trade_date: str,
    open_price: float,
    price_30min: float,
    open_return: float,
    outcome: str,
) -> bool:
    """
    Fill in open_return and outcome for a scan result.

    Args:
        ticker:      stock ticker
        trade_date:  "YYYY-MM-DD"
        open_price:  price at 9:30 AM open
        price_30min: price 30 minutes after open
        open_return: percentage return
        outcome:     WIN / LOSS / NEUTRAL / UP / DOWN / FLAT

    Returns:
        True if updated successfully
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            UPDATE scan_results
            SET open_price  = %s,
                price_30min = %s,
                open_return = %s,
                outcome     = %s
            WHERE ticker = %s
              AND date   = %s
              AND open_return IS NULL
        """, (open_price, price_30min, open_return, outcome, ticker, trade_date))

        rows = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()

        return rows > 0

    except Exception as e:
        print(f"[db] Failed to update outcome for {ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Historical context for RAG
# ─────────────────────────────────────────────────────────────

def get_historical_context(
    catalyst_type: str,
    market_cap: Optional[float] = None,
    min_samples: int = 5,
) -> str:
    """
    Retrieve historical stats for a catalyst type.
    Used to inject into LLM prompt for RAG.

    Args:
        catalyst_type: e.g. "FDA_APPROVAL"
        market_cap:    filter to similar market cap (±2x range)
        min_samples:   minimum records needed to return stats

    Returns:
        formatted string to inject into LLM prompt
        empty string if not enough data
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        # Base query
        query  = """
            SELECT
                count(*) as n,
                round(avg(open_return)::numeric, 1) as avg_return,
                round(
                    sum(case when outcome = 'WIN' then 1 else 0 end)::numeric
                    / nullif(count(case when outcome in ('WIN','LOSS') then 1 end), 0) * 100
                , 0) as win_rate
            FROM scan_results
            WHERE catalyst = %s
              AND outcome IS NOT NULL
        """
        params = [catalyst_type]

        # Optional market cap filter
        if market_cap and market_cap > 0:
            query  += " AND market_cap BETWEEN %s AND %s"
            params += [market_cap * 0.3, market_cap * 3.0]

        cur.execute(query, params)
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row or row[0] < min_samples:
            return ""

        n, avg_ret, win_rate = row
        return (
            f"OUR HISTORICAL DATA ({catalyst_type}, {n} cases):\n"
            f"  - Win rate: {win_rate:.0f}%\n"
            f"  - Avg open return: {avg_ret:+.1f}%\n"
            f"  - Sample size: {n} trades\n"
        )

    except Exception as e:
        print(f"[db] get_historical_context failed: {e}")
        return ""


def get_all_catalyst_stats() -> str:
    """
    Return a summary of all catalyst types with enough data.
    Used for general context in LLM prompts.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            SELECT catalyst, sample_size, avg_open_return, win_rate_pct
            FROM catalyst_stats
            WHERE sample_size >= 3
            ORDER BY sample_size DESC
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return ""

        lines = ["OUR HISTORICAL CATALYST STATS:"]
        for catalyst, n, avg_ret, win_rate in rows:
            lines.append(
                f"  {catalyst:<20} n={n:>3}  "
                f"win={win_rate:.0f}%  avg={avg_ret:+.1f}%"
            )

        return "\n".join(lines)

    except Exception as e:
        print(f"[db] get_all_catalyst_stats failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Knowledge base
# ─────────────────────────────────────────────────────────────

def add_knowledge(
    category: str,
    content: str,
    confidence: str = "MEDIUM",
    source: str = "manual observation",
) -> bool:
    """
    Add a rule or observation to the knowledge base.

    Args:
        category:   "catalyst" / "pattern" / "timing" / "risk"
        content:    the rule text
        confidence: "HIGH" / "MEDIUM" / "LOW"
        source:     where this came from

    Example:
        add_knowledge(
            category="catalyst",
            content="FDA Fast Track is NOT approval. Stocks often reverse 50% at open.",
            confidence="HIGH",
            source="observed 8 times"
        )
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            INSERT INTO knowledge (category, content, confidence, source)
            VALUES (%s, %s, %s, %s)
        """, (category, content, confidence, source))

        conn.commit()
        cur.close()
        conn.close()

        print(f"[db] Added knowledge: [{category}] {content[:60]}...")
        return True

    except Exception as e:
        print(f"[db] add_knowledge failed: {e}")
        return False


def get_relevant_knowledge(
    catalyst_type: str,
    float_shares: float = 0,
    keywords: list = None,
) -> str:
    """
    Retrieve relevant rules from knowledge base.
    Used to inject into LLM prompt.

    Args:
        catalyst_type: e.g. "FDA_FAST_TRACK"
        float_shares:  to trigger float-related rules
        keywords:      additional search terms

    Returns:
        formatted string for LLM prompt
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("SELECT category, content, confidence FROM knowledge")
        all_rules = cur.fetchall()
        cur.close()
        conn.close()

        relevant = []

        for category, content, confidence in all_rules:
            content_lower = content.lower()

            # Match by catalyst type
            if catalyst_type.lower().replace("_", " ") in content_lower:
                relevant.append(content)
                continue

            # Match by float size
            if float_shares < 5e6 and "float" in content_lower:
                relevant.append(content)
                continue

            # Match by keywords
            if keywords:
                for kw in keywords:
                    if kw.lower() in content_lower:
                        relevant.append(content)
                        break

        if not relevant:
            return ""

        lines = ["OUR KNOWLEDGE BASE:"]
        for rule in relevant[:5]:  # max 5 rules to keep prompt concise
            lines.append(f"  - {rule}")

        return "\n".join(lines)

    except Exception as e:
        print(f"[db] get_relevant_knowledge failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# News storage
# ─────────────────────────────────────────────────────────────

def store_news(ticker: str, news_items: list) -> int:
    """
    Store news articles for a ticker.

    Args:
        ticker:     stock ticker
        news_items: list of {headline, summary, source, datetime}

    Returns:
        number of articles inserted
    """
    if not news_items:
        return 0

    try:
        conn = get_connection()
        cur  = conn.cursor()
        inserted = 0

        for n in news_items:
            try:
                # Convert unix timestamp to datetime if needed
                pub_ts = n.get("datetime", 0)
                pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else None

                cur.execute("""
                    INSERT INTO news (ticker, headline, summary, source, published_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    ticker,
                    n.get("headline", ""),
                    n.get("summary", ""),
                    n.get("source", ""),
                    pub_dt,
                ))
                inserted += 1
            except Exception:
                continue

        conn.commit()
        cur.close()
        conn.close()

        return inserted

    except Exception as e:
        print(f"[db] store_news failed for {ticker}: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# Quick stats (for monitoring)
# ─────────────────────────────────────────────────────────────

def print_db_stats() -> None:
    """Print a quick summary of what's in the database."""
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("SELECT count(*) FROM scan_results")
        total = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM scan_results WHERE outcome IS NOT NULL")
        with_outcome = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM news")
        news_count = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM knowledge")
        knowledge_count = cur.fetchone()[0]

        cur.close()
        conn.close()

        print(f"\n{'─'*40}")
        print(f"DATABASE STATS")
        print(f"{'─'*40}")
        print(f"Scan results:    {total} total, {with_outcome} with outcome")
        print(f"News articles:   {news_count}")
        print(f"Knowledge rules: {knowledge_count}")
        print(f"{'─'*40}\n")

    except Exception as e:
        print(f"[db] print_db_stats failed: {e}")


# ─────────────────────────────────────────────────────────────
# Test connection
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Supabase connection...")

    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"Connected: {version[:50]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")
        exit(1)

    print_db_stats()

    # Test adding a knowledge rule
    print("Adding test knowledge rule...")
    add_knowledge(
        category   = "catalyst",
        content    = "FDA Fast Track is NOT approval. Stocks often reverse 50%+ of premarket gains at open.",
        confidence = "HIGH",
        source     = "initial setup",
    )

    add_knowledge(
        category   = "pattern",
        content    = "No news + float < 5M shares + RVOL > 10x = high probability pump and dump. Avoid.",
        confidence = "HIGH",
        source     = "initial setup",
    )

    add_knowledge(
        category   = "timing",
        content    = "Bitcoin miners (MARA, RIOT, CLSK, HUT) move together. Use BTC price as leading indicator.",
        confidence = "MEDIUM",
        source     = "initial setup",
    )

    print_db_stats()
    print("All tests passed!")
