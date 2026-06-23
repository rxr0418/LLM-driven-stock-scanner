"""
agents/memory_agent.py - Memory Agent for Swing Trade Phase 2.

Retrieval layers (all semantic or ticker-keyed, no LLM calls):
  1. Knowledge rules     : pgvector semantic search (trading rules + sector rules)
  2. Similar past cases  : pgvector semantic search on swing_results
  3. Upcoming events     : exact ticker lookup (earnings, FDA, macro)
  4. Analyst ratings     : exact ticker lookup, last 30 days
  5. SEC filings         : exact ticker lookup, last 2 filings

Runs AFTER Search Agent so catalyst_type is available for retrieval.
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))
from embeddings import build_knowledge_query
from config import MAX_KNOWLEDGE_RULES, MAX_SIMILAR_CASES, MAX_ANALYST_RATINGS, MAX_SEC_FILINGS


def _query_knowledge_semantic(query: str) -> list:
    try:
        from database import search_knowledge_semantic
        rows = search_knowledge_semantic(query, limit=6)
        return [
            f"{r['content']} (category={r['category']}, sim={r.get('similarity', '?'):.2f})"
            if r.get("category") else r["content"]
            for r in rows
        ]
    except Exception as e:
        print(f"  [memory_agent] knowledge query failed: {e}")
        return []


def _query_similar_cases(query: str) -> list:
    try:
        from database import search_swing_cases_semantic
        return search_swing_cases_semantic(query, limit=3)
    except Exception as e:
        print(f"  [memory_agent] case retrieval failed: {e}")
        return []


def _query_events(ticker: str) -> list:
    try:
        from database import get_upcoming_events
        return get_upcoming_events(ticker, within_days=14)
    except Exception as e:
        print(f"  [memory_agent] events query failed: {e}")
        return []


def _query_analyst_ratings(ticker: str) -> list:
    try:
        from database import search_analyst_ratings
        return search_analyst_ratings(ticker, limit=5)
    except Exception as e:
        print(f"  [memory_agent] analyst ratings query failed: {e}")
        return []


def _query_sec_filings(ticker: str) -> list:
    try:
        from database import search_sec_filings
        return search_sec_filings(ticker, limit=2)
    except Exception as e:
        print(f"  [memory_agent] SEC filings query failed: {e}")
        return []


def _build_event_risk_flag(events: list) -> str | None:
    """Return a risk flag string if a binary event is imminent (<= 5 days)."""
    for ev in events:
        if ev.get("days_away", 99) <= 5:
            return f"{ev['event_type']} in {ev['days_away']}d ({ev['event_date']})"
    return None


def run(
    ticker: str,
    signal_direction: str,
    regime: str,
    factors_used: list,
    catalyst_type: Optional[str] = None,
) -> dict:
    """
    Retrieve all context layers for a single ticker.

    Args:
        ticker           : e.g. "AAPL"
        signal_direction : "BUY" or "SHORT"
        regime           : "TRENDING" | "VOLATILE" | "NEUTRAL"
        factors_used     : list of factor names from scanner
        catalyst_type    : from Search Agent output (e.g. "CONTRACT_WIN")

    Returns:
        Memory context dict for merge() and Decision Agent.
    """
    print(f"  [memory_agent] {ticker}: querying (catalyst={catalyst_type})")

    semantic_query  = build_knowledge_query(ticker, signal_direction, regime, catalyst_type)

    knowledge_rules = _query_knowledge_semantic(semantic_query)[:MAX_KNOWLEDGE_RULES]
    similar_cases   = _query_similar_cases(semantic_query)[:MAX_SIMILAR_CASES]
    upcoming_events = _query_events(ticker)
    analyst_ratings = _query_analyst_ratings(ticker)[:MAX_ANALYST_RATINGS]
    sec_filings     = _query_sec_filings(ticker)[:MAX_SEC_FILINGS]

    cases_with_outcome = [c for c in similar_cases if c.get("actual_return") is not None]

    if cases_with_outcome:
        context_summary = (
            f"{len(cases_with_outcome)} similar past cases found with outcomes. "
            f"Avg return: {sum(c['actual_return'] for c in cases_with_outcome) / len(cases_with_outcome):+.1f}%."
        )
    else:
        context_summary = "No similar past cases with outcomes yet. Rely on knowledge rules."

    event_risk_flag = _build_event_risk_flag(upcoming_events)

    print(f"  [memory_agent] {ticker}: done "
          f"(rules={len(knowledge_rules)}, cases={len(similar_cases)}, "
          f"events={len(upcoming_events)}, ratings={len(analyst_ratings)}, "
          f"filings={len(sec_filings)})")

    return {
        "ticker":          ticker,
        "knowledge_rules": knowledge_rules,
        "similar_cases":   similar_cases,
        "upcoming_events": upcoming_events,
        "analyst_ratings": analyst_ratings,
        "sec_filings":     sec_filings,
        "event_risk_flag": event_risk_flag,
        "context_summary": context_summary,
        "react_trace":     [],
    }


# ─────────────────────────────────────────────────────────────
# Recheck entry point — called by Orchestrator on second retrieval
# ─────────────────────────────────────────────────────────────

_RECHECK_STRATEGIES = {
    "relax_similarity",    # HISTORICAL_CONTEXT: more similar_cases, lower threshold
    "extend_date_range",   # EVENT_VERIFICATION: look further out for events
    "extend_sec_window",   # SEC_FILING_DETAIL:  more filings, older window
}


def run_recheck(
    ticker: str,
    signal_direction: str,
    regime: str,
    factors_used: list,
    catalyst_type: Optional[str],
    strategy: str,
    recheck_questions: list[str],
) -> dict:
    """
    Re-run Memory Agent with a different retrieval strategy.
    Returns same schema as run() so Orchestrator can append as a new round.

    Strategies (from CAPABILITY_REGISTRY):
      relax_similarity  : double similar_cases limit, recall > precision
      extend_date_range : look 60d ahead for events (vs 14d default)
      extend_sec_window : retrieve 5 filings (vs 2 default)
    """
    if strategy not in _RECHECK_STRATEGIES:
        print(f"  [memory_agent] {ticker}: unknown recheck strategy '{strategy}', using run()")
        return run(ticker, signal_direction, regime, factors_used, catalyst_type)

    print(f"  [memory_agent] {ticker}: recheck strategy={strategy} "
          f"questions={recheck_questions}")

    semantic_query = build_knowledge_query(ticker, signal_direction, regime, catalyst_type)

    # Always re-fetch all layers; only the flagged layer uses relaxed params
    knowledge_rules = _query_knowledge_semantic(semantic_query)[:MAX_KNOWLEDGE_RULES]

    if strategy == "relax_similarity":
        # Double the case limit — lower effective similarity threshold via more results
        try:
            from database import search_swing_cases_semantic
            similar_cases = search_swing_cases_semantic(semantic_query, limit=MAX_SIMILAR_CASES * 2)
        except Exception as e:
            print(f"  [memory_agent] {ticker}: relax_similarity failed: {e}")
            similar_cases = _query_similar_cases(semantic_query)
    else:
        similar_cases = _query_similar_cases(semantic_query)[:MAX_SIMILAR_CASES]

    if strategy == "extend_date_range":
        try:
            from database import get_upcoming_events
            upcoming_events = get_upcoming_events(ticker, within_days=60)
        except Exception as e:
            print(f"  [memory_agent] {ticker}: extend_date_range failed: {e}")
            upcoming_events = _query_events(ticker)
    else:
        upcoming_events = _query_events(ticker)

    analyst_ratings = _query_analyst_ratings(ticker)[:MAX_ANALYST_RATINGS]

    if strategy == "extend_sec_window":
        try:
            from database import search_sec_filings
            sec_filings = search_sec_filings(ticker, limit=5)
        except Exception as e:
            print(f"  [memory_agent] {ticker}: extend_sec_window failed: {e}")
            sec_filings = _query_sec_filings(ticker)[:MAX_SEC_FILINGS]
    else:
        sec_filings = _query_sec_filings(ticker)[:MAX_SEC_FILINGS]

    cases_with_outcome = [c for c in similar_cases if c.get("actual_return") is not None]
    if cases_with_outcome:
        context_summary = (
            f"[recheck:{strategy}] {len(cases_with_outcome)} cases with outcomes. "
            f"Avg return: {sum(c['actual_return'] for c in cases_with_outcome) / len(cases_with_outcome):+.1f}%."
        )
    else:
        context_summary = f"[recheck:{strategy}] No additional cases found with relaxed retrieval."

    event_risk_flag = _build_event_risk_flag(upcoming_events)

    print(f"  [memory_agent] {ticker}: recheck done "
          f"(cases={len(similar_cases)}, events={len(upcoming_events)}, "
          f"filings={len(sec_filings)})")

    return {
        "ticker":           ticker,
        "knowledge_rules":  knowledge_rules,
        "similar_cases":    similar_cases,
        "upcoming_events":  upcoming_events,
        "analyst_ratings":  analyst_ratings,
        "sec_filings":      sec_filings,
        "event_risk_flag":  event_risk_flag,
        "context_summary":  context_summary,
        "react_trace":      [],
        "recheck_strategy": strategy,
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    result = run(
        ticker="GS",
        signal_direction="BUY",
        regime="NEUTRAL",
        factors_used=["momentum_20d", "reversal_5d", "volume_spike"],
        catalyst_type="MACRO",
    )

    print("\n── Memory Agent Result ──")
    print(f"  context_summary: {result['context_summary']}")
    print(f"  event_risk_flag: {result['event_risk_flag']}")
    print(f"  knowledge_rules ({len(result['knowledge_rules'])}):")
    for r in result["knowledge_rules"]:
        print(f"    - {r}")
    print(f"  upcoming_events ({len(result['upcoming_events'])}):")
    for e in result["upcoming_events"]:
        print(f"    - {e}")
    print(f"  analyst_ratings ({len(result['analyst_ratings'])}):")
    for r in result["analyst_ratings"]:
        print(f"    - {r['summary']} ({r['rating_date']})")
    print(f"  sec_filings ({len(result['sec_filings'])}):")
    for f in result["sec_filings"]:
        print(f"    - {f['filing_type']} {f['filed_date']}: {f['summary']}")
    print(f"  similar_cases ({len(result['similar_cases'])}):")
    for c in result["similar_cases"]:
        print(f"    - {c}")
