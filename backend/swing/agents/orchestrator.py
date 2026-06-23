"""
agents/orchestrator.py - Deterministic Orchestrator for Swing Trade Phase 2.

State machine:
  INIT → SEARCH → MEMORY → SKEPTIC → [recheck?] → DECISION → DONE

Recheck rules (enforced here, not in agents):
  - Skeptic can block Decision once (skeptic_block_used)
  - Decision can request recheck once (decision_recheck_used)
  - If both limits are hit, force Decision with current state (forced_decision)
  - Total recheck_count cap = 2 (one per source max)

The Orchestrator is the only place that writes SharedState control fields.
Agents only return results; they never mutate state directly.
"""

import asyncio
from typing import Optional

from swing.agents.orchestrator_types import (
    SharedState,
    make_shared_state,
    append_round,
    latest_round,
    CAPABILITY_REGISTRY,
)
from swing.agents.merge import build_decision_context

MAX_RECHECK = 2  # hard cap across all recheck sources


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _log(state: SharedState, msg: str) -> None:
    state["steps"].append(msg)
    print(f"  [orchestrator] {state['ticker']}: {msg}")


def _run_search(state: SharedState, recheck_questions: Optional[list[str]] = None) -> None:
    """Run Search Agent and append result to state."""
    from swing.agents.search_agent import run as search_run

    strategy = "targeted_query" if recheck_questions else None
    _log(state, f"search {'recheck' if strategy else 'run'} start")

    result = search_run(
        state["ticker"],
        state["signal_direction"],
        state["factor_score"],
        state["regime"],
        _get_yahoo_articles(state),
        recheck_questions=recheck_questions,
    )
    append_round(state["search"], result, recheck_strategy=strategy)
    _log(state, f"search done catalyst={result.get('catalyst_type')} "
                f"strength={result.get('catalyst_strength')}")


def _run_memory(
    state: SharedState,
    strategy: Optional[str] = None,
    recheck_questions: Optional[list[str]] = None,
) -> None:
    """Run Memory Agent (or recheck variant) and append result to state."""
    from swing.agents.memory_agent import run as memory_run, run_recheck as memory_recheck

    catalyst_type = latest_round(state["search"], ).get("catalyst_type") if state["search"]["rounds"] else None

    if strategy:
        _log(state, f"memory recheck start strategy={strategy}")
        result = memory_recheck(
            ticker=state["ticker"],
            signal_direction=state["signal_direction"],
            regime=state["regime"],
            factors_used=state["factors_used"],
            catalyst_type=catalyst_type,
            strategy=strategy,
            recheck_questions=recheck_questions or [],
        )
    else:
        _log(state, "memory run start")
        result = memory_run(
            ticker=state["ticker"],
            signal_direction=state["signal_direction"],
            regime=state["regime"],
            factors_used=state["factors_used"],
            catalyst_type=catalyst_type,
        )

    append_round(state["memory"], result, recheck_strategy=strategy)
    _log(state, f"memory done cases={len(result.get('similar_cases', []))} "
                f"event_risk={result.get('event_risk_flag')}")


def _run_skeptic(state: SharedState) -> dict:
    """Run Skeptic Agent and append result to state."""
    from swing.agents.skeptic_agent import run as skeptic_run

    search_result = latest_round(state["search"]) or {}
    memory_result = latest_round(state["memory"]) or {}

    _log(state, "skeptic run start")
    result = skeptic_run(
        ticker=state["ticker"],
        signal_direction=state["signal_direction"],
        factor_score=state["factor_score"],
        regime=state["regime"],
        search_result=search_result,
        memory_result=memory_result,
    )
    append_round(state["skeptic"], result)
    _log(state, f"skeptic done concern={result.get('concern_level')} "
                f"cap={result.get('confidence_cap')} "
                f"needs_recheck={result.get('needs_recheck')}")
    return result


def _run_decision(state: SharedState, forced: bool = False) -> dict:
    """Run Decision Agent with current state and return its output."""
    from swing.agents.decision_agent import run as decision_run

    context = build_decision_context(state)
    _log(state, f"decision run start forced={forced}")
    return decision_run(context, forced=forced)


def _get_yahoo_articles(state: SharedState) -> list:
    """Retrieve cached Yahoo articles from state, or empty list."""
    return state.get("_yahoo_articles", [])  # type: ignore[typeddict-item]


def _resolve_memory_strategy(missing_info_type: str) -> str:
    """Map capability type to memory recheck strategy."""
    cap = CAPABILITY_REGISTRY.get(missing_info_type, {})
    return cap.get("memory_strategy") or "relax_similarity"


# ─────────────────────────────────────────────────────────────
# Main orchestration loop
# ─────────────────────────────────────────────────────────────

def run_ticker(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    factors_used: list[str],
    yahoo_articles: list,
) -> tuple[SharedState, dict]:
    """
    Run the full orchestration loop for one ticker (synchronous).

    Returns:
        (final_state, decision_result)
        decision_result always has status="DECIDE" on return.
    """
    state = make_shared_state(ticker, signal_direction, factor_score, regime, factors_used)
    state["_yahoo_articles"] = yahoo_articles  # type: ignore[typeddict-unknown-key]

    # ── Phase 1: initial pipeline ─────────────────────────────
    _run_search(state)
    _run_memory(state)
    skeptic_result = _run_skeptic(state)

    # ── Phase 2: Skeptic recheck routing ─────────────────────
    if (
        skeptic_result.get("needs_recheck")
        and not state["skeptic_block_used"]
        and state["recheck_count"] < MAX_RECHECK
    ):
        questions = skeptic_result.get("requested_recheck_questions", [])
        _log(state, f"skeptic requested recheck questions={questions}")
        state["skeptic_block_used"] = True
        state["recheck_count"] += 1

        # Skeptic recheck always targets Search (catalyst verification)
        _run_search(state, recheck_questions=questions)
        # Re-run Skeptic with updated search result
        skeptic_result = _run_skeptic(state)
    else:
        if skeptic_result.get("needs_recheck"):
            state["forced_decision"] = True
            _log(state, "skeptic recheck limit hit — proceeding with current state")

    # ── Phase 3: Decision Agent ───────────────────────────────
    decision = _run_decision(state, forced=state["forced_decision"])

    # ── Phase 4: Decision recheck routing ────────────────────
    if (
        decision.get("status") == "NEED_RECHECK"
        and not state["decision_recheck_used"]
        and state["recheck_count"] < MAX_RECHECK
    ):
        missing_info_type = decision["missing_info_type"]
        requested_agent   = decision["requested_agent"]
        questions         = decision.get("recheck_questions", [])

        _log(state, f"decision requested recheck "
                    f"type={missing_info_type} agent={requested_agent}")
        state["decision_recheck_used"] = True
        state["recheck_count"] += 1

        if requested_agent == "SEARCH":
            _run_search(state, recheck_questions=questions)
            # Re-run Skeptic so it audits the new search result
            _run_skeptic(state)
        elif requested_agent == "MEMORY":
            strategy = _resolve_memory_strategy(missing_info_type)
            _run_memory(state, strategy=strategy, recheck_questions=questions)

        # Final Decision — forced, no more recheks allowed
        state["forced_decision"] = True
        decision = _run_decision(state, forced=True)
    elif decision.get("status") == "NEED_RECHECK":
        # Limit already hit — force with current state
        state["forced_decision"] = True
        _log(state, "decision recheck limit hit — forcing decision")
        decision = _run_decision(state, forced=True)

    _log(state, f"done signal={decision.get('signal')} conf={decision.get('confidence')}")
    return state, decision


async def run_ticker_async(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    factors_used: list[str],
    yahoo_articles: list,
) -> tuple[SharedState, dict]:
    """Async wrapper — runs the synchronous loop in a thread."""
    return await asyncio.to_thread(
        run_ticker,
        ticker, signal_direction, factor_score, regime, factors_used, yahoo_articles,
    )
