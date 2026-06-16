"""
agents/memory_agent.py - Memory Worker for Swing Trade Phase 2.

Responsibilities:
  - Query Supabase swing_stats for historical win rates (tiered fallback)
  - Query knowledge table for relevant trading rules
  - Return structured memory context for the Decision Agent

No LLM calls — pure Python with deterministic query logic.

Query key priority:
  1. (regime, signal)   — most specific, use if sample_size >= MIN_SAMPLE
  2. (signal only)      — drop regime if step 1 insufficient
  3. knowledge rules only — if no reliable stats anywhere

Design note:
  Memory Agent only READS from Supabase during Phase 2.
  All writes happen in database.py, called from main.py
  after Decision Agent completes.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))

MIN_SAMPLE = 10


def get_connection():
    """Get Supabase PostgreSQL connection."""
    import psycopg2
    url = os.environ.get("SUPABASE_URL", "")
    if not url:
        raise ValueError("SUPABASE_URL not set")
    return psycopg2.connect(url)


def _query_stats(regime: str = None, signal: str = None) -> dict:
    """
    Query swing_stats with optional regime and signal filters.
    Returns first matching row as dict, or empty dict if no results.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        conditions = []
        params = []
        if regime:
            conditions.append("regime = %s")
            params.append(regime)
        if signal:
            conditions.append("signal = %s")
            params.append(signal)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        cur.execute(f"""
            SELECT regime, signal, sample_size,
                   win_rate_5d, avg_return_5d,
                   win_rate_10d, avg_return_10d
            FROM swing_stats
            {where}
            ORDER BY sample_size DESC
            LIMIT 1
        """, params)

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {}

        return {
            "regime":       row[0],
            "signal":       row[1],
            "sample_size":  row[2],
            "win_rate_5d":  row[3],
            "avg_return_5d": row[4],
            "win_rate_10d": row[5],
            "avg_return_10d": row[6],
        }

    except Exception as e:
        print(f"  [memory_agent] stats query failed: {e}")
        return {}


def _query_knowledge() -> list:
    """Fetch trading rules from knowledge table."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT content, category, confidence
            FROM knowledge
            ORDER BY created_at DESC
            LIMIT 8
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            f"{r[0]} (category={r[1]}, confidence={r[2]})" if r[1] else r[0]
            for r in rows
        ]
    except Exception:
        return []


def run(
    ticker: str,
    signal_direction: str,
    regime: str,
    factors_used: list,
) -> dict:
    """
    Retrieve historical context for a single ticker.
    Pure Python — no LLM calls.

    Tiered query strategy:
      Level 1: regime + signal  (most specific)
      Level 2: signal only      (broader)
      Level 3: no stats         (knowledge rules only)

    Args:
        ticker           : e.g. "AAPL"
        signal_direction : "BUY" or "SHORT"
        regime           : "TRENDING" | "VOLATILE" | "NEUTRAL"
        factors_used     : list of factor names from scanner

    Returns:
        Structured memory context dict for merge() and Decision Agent.
    """
    print(f"  [memory_agent] {ticker}: querying DB")

    # ── Level 1: regime + signal ──────────────────────────────
    stats = _query_stats(regime=regime, signal=signal_direction)
    query_level = "specific"

    # ── Level 2: signal only ──────────────────────────────────
    if not stats or stats.get("sample_size", 0) < MIN_SAMPLE:
        stats = _query_stats(signal=signal_direction)
        query_level = "signal"

    # ── Level 3: no reliable stats ───────────────────────────
    if not stats or stats.get("sample_size", 0) < MIN_SAMPLE:
        stats = {}
        query_level = "none"

    # ── Knowledge rules (always fetch) ───────────────────────
    knowledge_rules = _query_knowledge()

    # ── Assemble result ───────────────────────────────────────
    has_stats = bool(stats)
    sample_size = stats.get("sample_size") if has_stats else None
    win_rate = stats.get("win_rate_10d") if has_stats else None
    avg_return = stats.get("avg_return_10d") if has_stats else None
    confidence = "HIGH" if has_stats else "NONE"

    if has_stats:
        context_summary = (
            f"{regime}/{signal_direction} regime shows {win_rate:.0f}% win rate "
            f"and {avg_return:+.1f}% avg 10d return across {sample_size} samples "
            f"(query level: {query_level})."
        )
    else:
        context_summary = (
            f"No reliable historical stats for {regime}/{signal_direction}. "
            f"Rely on knowledge rules only."
        )

    print(f"  [memory_agent] {ticker}: done (level={query_level}, n={sample_size}, win={win_rate})")

    return {
        "ticker":              ticker,
        "has_stats":           has_stats,
        "win_rate":            win_rate,
        "avg_return":          avg_return,
        "sample_size":         sample_size,
        "query_level":         query_level,
        "knowledge_rules":     knowledge_rules,
        "confidence_in_prior": confidence,
        "context_summary":     context_summary,
        "react_trace":         [],  # no LLM calls — empty trace
    }


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = "GS"
    print(f"Testing Memory Agent on {ticker}...\n")

    result = run(
        ticker=ticker,
        signal_direction="BUY",
        regime="NEUTRAL",
        factors_used=["momentum_20d", "reversal_5d", "volume_spike"],
    )

    print("\n── Memory Agent Result ──")
    for k, v in result.items():
        if k == "knowledge_rules":
            print(f"  knowledge_rules ({len(v)}):")
            for r in v:
                print(f"    - {r}")
        else:
            print(f"  {k}: {v}")