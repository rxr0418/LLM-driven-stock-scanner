"""
agents/merge.py - Context assembly for Phase 2.

Responsibilities:
  - Build Decision Agent context from SharedState (latest round per agent + deltas)
  - Apply confidence_in_prior gating (don't inject LOW/NONE stats)
  - Add regime-level metadata (candidate count hint, holding period hint)

This is pure Python — no LLM calls, no DB calls.
"""

from typing import Optional
from swing.agents.orchestrator_types import SharedState, latest_round


# Regime-level hints passed to Decision Agent
REGIME_HINTS = {
    "TRENDING": {
        "candidate_hint": "Normal confidence — trending market supports momentum signals.",
        "holding_period_hint": "5–10 trading days typical for trending regime.",
        "max_candidates": 10,
    },
    "VOLATILE": {
        "candidate_hint": "Reduce confidence — choppy market increases false signals.",
        "holding_period_hint": "1–3 trading days max — volatile regime, exit quickly.",
        "max_candidates": 6,
    },
    "NEUTRAL": {
        "candidate_hint": "Mixed signals — evaluate each stock individually.",
        "holding_period_hint": "3–5 trading days typical for neutral regime.",
        "max_candidates": 8,
    },
}


def build_decision_context(state: SharedState) -> dict:
    """
    Build Decision Agent context from SharedState.

    Uses the latest round per agent + delta summary so the Decision Agent
    always sees current information at a fixed context size, regardless
    of how many recheck rounds ran.
    """
    ticker           = state["ticker"]
    signal_direction = state["signal_direction"]
    factor_score     = state["factor_score"]
    regime           = state["regime"]
    regime_meta      = REGIME_HINTS.get(regime, REGIME_HINTS["NEUTRAL"])

    search_result  = latest_round(state["search"]) or {}
    memory_result  = latest_round(state["memory"]) or {}
    skeptic_result = latest_round(state["skeptic"]) or {}

    # ── Search context (latest round only) ───────────────────
    search_context = {
        "catalyst_type":     search_result.get("catalyst_type", "OTHER"),
        "catalyst_strength": search_result.get("catalyst_strength", "NONE"),
        "news_alignment":    search_result.get("news_alignment", "NEUTRAL"),
        "summary":           search_result.get("summary", "No news summary."),
        "risk_flag":         search_result.get("risk_flag", "none"),
        "sources":           search_result.get("sources", [])[:3],
        "search_count":      search_result.get("search_count", 0),
        "recheck_delta":     state["search"].get("delta"),  # what changed vs round 1
    }

    # ── Memory context (latest round only) ───────────────────
    memory_context: dict = {
        "knowledge_rules": memory_result.get("knowledge_rules", [])[:5],
        "similar_cases":   memory_result.get("similar_cases", []),
        "upcoming_events": memory_result.get("upcoming_events", []),
        "analyst_ratings": memory_result.get("analyst_ratings", []),
        "sec_filings":     memory_result.get("sec_filings", []),
        "event_risk_flag": memory_result.get("event_risk_flag"),
        "context_summary": memory_result.get("context_summary", "No historical context."),
        "recheck_delta":   state["memory"].get("delta"),
    }

    # ── Skeptic context (latest round only) ──────────────────
    skeptic_context = {
        "thesis_quality":    skeptic_result.get("thesis_quality", "MIXED"),
        "concern_level":     skeptic_result.get("concern_level", "MEDIUM"),
        "needs_recheck":     skeptic_result.get("needs_recheck", False),
        "confidence_cap":    skeptic_result.get("confidence_cap", 70),
        "concerns":          skeptic_result.get("concerns", [])[:4],
        "recheck_questions": skeptic_result.get("requested_recheck_questions", [])[:3],
        "summary":           skeptic_result.get("summary", "No skeptic review available."),
        "recheck_delta":     state["skeptic"].get("delta"),
    }

    # ── Orchestrator audit info for Decision Agent ────────────
    audit = {
        "recheck_count":          state["recheck_count"],
        "decision_recheck_used":  state["decision_recheck_used"],
        "forced_decision":        state["forced_decision"],
        "steps":                  state["steps"][-5:],  # last 5 steps only
    }

    return {
        "ticker":              ticker,
        "signal_direction":    signal_direction,
        "factor_score":        round(factor_score, 4),
        "regime":              regime,
        "regime_hint":         regime_meta["candidate_hint"],
        "holding_period_hint": regime_meta["holding_period_hint"],
        "search":              search_context,
        "memory":              memory_context,
        "skeptic":             skeptic_context,
        "orchestrator":        audit,
    }


def merge(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    search_result: dict,
    memory_result: dict,
    skeptic_result: Optional[dict] = None,
) -> dict:
    """
    Legacy flat merge — kept for backward compatibility with existing tests.
    New code should use build_decision_context(state) instead.
    """
    regime_meta = REGIME_HINTS.get(regime, REGIME_HINTS["NEUTRAL"])

    search_context = {
        "catalyst_type":     search_result.get("catalyst_type", "OTHER"),
        "catalyst_strength": search_result.get("catalyst_strength", "NONE"),
        "news_alignment":    search_result.get("news_alignment", "NEUTRAL"),
        "summary":           search_result.get("summary", "No news summary."),
        "risk_flag":         search_result.get("risk_flag", "none"),
        "sources":           search_result.get("sources", [])[:3],
        "search_count":      search_result.get("search_count", 0),
        "recheck_delta":     None,
    }
    memory_context: dict = {
        "knowledge_rules": memory_result.get("knowledge_rules", [])[:5],
        "similar_cases":   memory_result.get("similar_cases", []),
        "upcoming_events": memory_result.get("upcoming_events", []),
        "analyst_ratings": memory_result.get("analyst_ratings", []),
        "sec_filings":     memory_result.get("sec_filings", []),
        "event_risk_flag": memory_result.get("event_risk_flag"),
        "context_summary": memory_result.get("context_summary", "No historical context."),
        "recheck_delta":   None,
    }
    skeptic_result = skeptic_result or {}
    skeptic_context = {
        "thesis_quality":    skeptic_result.get("thesis_quality", "MIXED"),
        "concern_level":     skeptic_result.get("concern_level", "MEDIUM"),
        "needs_recheck":     skeptic_result.get("needs_recheck", False),
        "confidence_cap":    skeptic_result.get("confidence_cap", 70),
        "concerns":          skeptic_result.get("concerns", [])[:4],
        "recheck_questions": skeptic_result.get("requested_recheck_questions", [])[:3],
        "summary":           skeptic_result.get("summary", "No skeptic review available."),
        "recheck_delta":     None,
    }
    return {
        "ticker":              ticker,
        "signal_direction":    signal_direction,
        "factor_score":        round(factor_score, 4),
        "regime":              regime,
        "regime_hint":         regime_meta["candidate_hint"],
        "holding_period_hint": regime_meta["holding_period_hint"],
        "search":              search_context,
        "memory":              memory_context,
        "skeptic":             skeptic_context,
        "orchestrator":        {"recheck_count": 0, "decision_recheck_used": False,
                                "forced_decision": False, "steps": []},
    }


def estimate_holding_period(regime: str, confidence: int) -> int:
    """
    Estimate recommended holding period in trading days.
    Used by Decision Agent and stored in swing_results.

    Rules:
      VOLATILE  : 1–3 days regardless of confidence
      TRENDING  : 5–10 days, scaled by confidence
      NEUTRAL   : 3–5 days, scaled by confidence
    """
    if regime == "VOLATILE":
        if confidence >= 75:
            return 3
        return 1
    elif regime == "TRENDING":
        if confidence >= 80:
            return 10
        elif confidence >= 60:
            return 7
        return 5
    else:  # NEUTRAL
        if confidence >= 75:
            return 5
        return 3


def get_max_candidates(regime: str) -> int:
    """Return the max number of candidates to output for this regime."""
    return REGIME_HINTS.get(regime, REGIME_HINTS["NEUTRAL"])["max_candidates"]
