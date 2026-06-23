"""
agents/orchestrator_types.py - Shared state schema and capability registry.

All agents read from / write into SharedState.
The Orchestrator is the only writer of top-level control fields.
"""

from typing import TypedDict, Optional


# ─────────────────────────────────────────────────────────────
# Per-agent round (one execution of an agent)
# ─────────────────────────────────────────────────────────────

class AgentRound(TypedDict):
    result: dict        # raw output from the agent
    recheck_strategy: Optional[str]  # e.g. "extended_window", "targeted_query"


class AgentHistory(TypedDict):
    rounds: list[AgentRound]
    delta: Optional[str]   # human-readable summary of what changed between rounds


# ─────────────────────────────────────────────────────────────
# Recheck request (from Decision Agent or Skeptic Agent)
# ─────────────────────────────────────────────────────────────

# All valid missing_info_type values — Decision Agent must use one of these.
# Maps to which agent the Orchestrator will dispatch.
CAPABILITY_REGISTRY: dict[str, dict] = {
    "CATALYST_VERIFICATION": {
        "agent": "SEARCH",
        "description": "Re-search with targeted query to verify catalyst classification",
        "memory_strategy": None,
    },
    "NEWS_DEEP_DIVE": {
        "agent": "SEARCH",
        "description": "Broader news search with extended time window",
        "memory_strategy": None,
    },
    "HISTORICAL_CONTEXT": {
        "agent": "MEMORY",
        "description": "Re-query similar cases with relaxed similarity threshold",
        "memory_strategy": "relax_similarity",
    },
    "EVENT_VERIFICATION": {
        "agent": "MEMORY",
        "description": "Re-query events table with extended date range",
        "memory_strategy": "extend_date_range",
    },
    "SEC_FILING_DETAIL": {
        "agent": "MEMORY",
        "description": "Re-query SEC filings with broader ticker/date scope",
        "memory_strategy": "extend_sec_window",
    },
}

VALID_MISSING_INFO_TYPES = set(CAPABILITY_REGISTRY.keys())


class RecheckRequest(TypedDict):
    source: str                  # "DECISION" | "SKEPTIC"
    missing_info_type: str       # must be a key in CAPABILITY_REGISTRY
    requested_agent: str         # "SEARCH" | "MEMORY" (derived from registry)
    recheck_questions: list[str] # specific questions to guide the recheck


# ─────────────────────────────────────────────────────────────
# Shared state — the single object passed through the pipeline
# ─────────────────────────────────────────────────────────────

class SharedState(TypedDict):
    # ── Input (set once at pipeline start) ───────────────────
    ticker: str
    signal_direction: str        # "BUY" | "SHORT"
    factor_score: float
    regime: str
    factors_used: list[str]

    # ── Agent histories (appended as rounds complete) ─────────
    search: AgentHistory
    memory: AgentHistory
    skeptic: AgentHistory

    # ── Orchestrator control fields ───────────────────────────
    recheck_count: int           # total recheck dispatches so far (across all agents)
    search_recheck_used: bool    # Search Agent has been re-run once
    decision_recheck_used: bool  # Decision Agent has requested recheck once
    skeptic_block_used: bool     # Skeptic has blocked Decision once

    # ── Audit trail ──────────────────────────────────────────
    steps: list[str]             # ordered log of what happened, for debugging
    forced_decision: bool        # True if we hit a limit and forced conclusion


def make_shared_state(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    factors_used: list[str],
) -> SharedState:
    """Create a fresh SharedState for one ticker."""
    return SharedState(
        ticker=ticker,
        signal_direction=signal_direction,
        factor_score=factor_score,
        regime=regime,
        factors_used=factors_used,
        search=AgentHistory(rounds=[], delta=None),
        memory=AgentHistory(rounds=[], delta=None),
        skeptic=AgentHistory(rounds=[], delta=None),
        recheck_count=0,
        search_recheck_used=False,
        decision_recheck_used=False,
        skeptic_block_used=False,
        steps=[],
        forced_decision=False,
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def latest_round(history: AgentHistory) -> Optional[dict]:
    """Return the most recent agent result, or None if no rounds yet."""
    if not history["rounds"]:
        return None
    return history["rounds"][-1]["result"]


def append_round(
    history: AgentHistory,
    result: dict,
    recheck_strategy: Optional[str] = None,
) -> None:
    """Append a new round to an agent's history and compute delta."""
    prev = latest_round(history)
    history["rounds"].append(AgentRound(result=result, recheck_strategy=recheck_strategy))

    # Compute delta only when there is a previous round to compare against
    if prev is not None:
        history["delta"] = _compute_delta(prev, result)


def _compute_delta(prev: dict, curr: dict) -> str:
    """
    Produce a short human-readable summary of what changed between two rounds.
    Keeps the delta field small so it doesn't bloat Decision Agent context.
    """
    changes = []

    # Search-specific fields
    for field in ("catalyst_type", "catalyst_strength", "news_alignment", "risk_flag"):
        if field in prev and field in curr and prev[field] != curr[field]:
            changes.append(f"{field}: {prev[field]} → {curr[field]}")

    # Memory-specific fields
    if prev.get("event_risk_flag") != curr.get("event_risk_flag"):
        changes.append(
            f"event_risk_flag: {prev.get('event_risk_flag')} → {curr.get('event_risk_flag')}"
        )
    prev_cases = len(prev.get("similar_cases", []))
    curr_cases = len(curr.get("similar_cases", []))
    if prev_cases != curr_cases:
        changes.append(f"similar_cases: {prev_cases} → {curr_cases} found")

    # Skeptic-specific fields
    for field in ("thesis_quality", "concern_level", "confidence_cap"):
        if field in prev and field in curr and prev[field] != curr[field]:
            changes.append(f"{field}: {prev[field]} → {curr[field]}")

    if not changes:
        return "No material changes from recheck."
    return "; ".join(changes)
