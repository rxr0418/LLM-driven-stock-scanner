"""
agents/merge.py - Context merge and trim for Phase 2.

Responsibilities:
  - Combine Search Agent and Memory Agent outputs into a clean
    context dict for the Decision Agent
  - Apply confidence_in_prior gating (don't inject LOW/NONE stats)
  - Trim to stay within a safe token budget
  - Add regime-level metadata (candidate count hint, holding period hint)

This is pure Python — no LLM calls, no DB calls.
"""

from typing import Optional


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


def merge(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    search_result: dict,
    memory_result: dict,
) -> dict:
    """
    Merge Search Agent and Memory Agent outputs into Decision Agent context.

    Args:
        ticker           : stock ticker
        signal_direction : "LONG" or "SHORT"
        factor_score     : composite score from scanner (0–1)
        regime           : "TRENDING" | "VOLATILE" | "NEUTRAL"
        search_result    : output of search_agent.run()
        memory_result    : output of memory_agent.run()

    Returns:
        Clean context dict for Decision Agent.
    """
    regime_meta = REGIME_HINTS.get(regime, REGIME_HINTS["NEUTRAL"])

    # ── Search context ────────────────────────────────────────
    search_context = {
        "catalyst_type":     search_result.get("catalyst_type", "OTHER"),
        "catalyst_strength": search_result.get("catalyst_strength", "NONE"),
        "news_alignment":    search_result.get("news_alignment", "NEUTRAL"),
        "summary":           search_result.get("summary", "No news summary."),
        "risk_flag":         search_result.get("risk_flag", "none"),
        "sources":           search_result.get("sources", [])[:3],
        "search_count":      search_result.get("search_count", 0),
    }

    # ── Memory context (gated by confidence) ─────────────────
    confidence_in_prior = memory_result.get("confidence_in_prior", "NONE")
    include_stats = confidence_in_prior == "HIGH"

    memory_context: dict = {
        "confidence_in_prior": confidence_in_prior,
        "knowledge_rules":     memory_result.get("knowledge_rules", [])[:5],
        "context_summary":     memory_result.get("context_summary", "No historical context."),
    }

    if include_stats:
        memory_context.update({
            "win_rate":    memory_result.get("win_rate"),
            "avg_return":  memory_result.get("avg_return"),
            "sample_size": memory_result.get("sample_size"),
            "query_level": memory_result.get("query_level"),
        })
    else:
        memory_context["stats_note"] = (
            f"Historical stats not injected "
            f"(confidence_in_prior={confidence_in_prior}, "
            f"sample_size={memory_result.get('sample_size', 'unknown')}). "
            f"Rely on knowledge rules only."
        )

    # ── Assemble final context ────────────────────────────────
    return {
        "ticker":            ticker,
        "signal_direction":  signal_direction,
        "factor_score":      round(factor_score, 4),
        "regime":            regime,
        "regime_hint":       regime_meta["candidate_hint"],
        "holding_period_hint": regime_meta["holding_period_hint"],
        "search":            search_context,
        "memory":            memory_context,
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
