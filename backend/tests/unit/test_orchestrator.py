"""
tests/unit/test_orchestrator.py - Unit tests for the Orchestrator state machine.

All LLM and DB calls are mocked.
Tests cover:
  1. Happy path (no recheks needed)
  2. Skeptic requests recheck → Search re-runs → Skeptic re-audits → Decision
  3. Decision requests NEED_RECHECK → correct agent dispatched → forced Decision
  4. Both limits hit → forced_decision=True
  5. Recheck limit enforcement (max 2 total)
  6. SharedState delta computed correctly

Run:
  cd backend
  pytest tests/unit/test_orchestrator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swing.agents.orchestrator_types import (
    make_shared_state,
    append_round,
    latest_round,
    _compute_delta,
)


# ─────────────────────────────────────────────────────────────
# Mock factories
# ─────────────────────────────────────────────────────────────

def _search_result(catalyst_type="EARNINGS_BEAT", strength="STRONG", alignment="SUPPORTS"):
    return {
        "catalyst_type": catalyst_type,
        "catalyst_strength": strength,
        "news_alignment": alignment,
        "summary": "Mock search summary.",
        "risk_flag": "none",
        "sources": ["Reuters"],
        "search_count": 1,
    }


def _memory_result(n_cases=2, event_risk=None):
    cases = [
        {"ticker": "AAPL", "signal": "BUY", "confidence": 75,
         "catalyst_type": "EARNINGS_BEAT", "actual_return": 3.2,
         "similarity": 0.87, "scan_date": "2025-01-10"}
    ] * n_cases
    return {
        "knowledge_rules": ["Earnings beats in TRENDING sustain 5–10d momentum."],
        "similar_cases": cases,
        "upcoming_events": [],
        "analyst_ratings": [],
        "sec_filings": [],
        "event_risk_flag": event_risk,
        "context_summary": f"{n_cases} similar cases found.",
    }


def _skeptic_result(concern="LOW", cap=80, needs_recheck=False, questions=None):
    return {
        "thesis_quality": "STRONG" if concern == "LOW" else "MIXED",
        "concern_level": concern,
        "needs_recheck": needs_recheck,
        "confidence_cap": cap,
        "concerns": ["Mock concern."],
        "requested_recheck_questions": questions or [],
        "summary": "Mock skeptic summary.",
    }


def _decision_decide(signal="BUY", confidence=75):
    return {
        "status": "DECIDE",
        "signal": signal,
        "confidence": confidence,
        "news_alignment": "SUPPORTS",
        "reason": "Mock reason for the trade.",
        "risk_flag": "none",
        "holding_period_days": 5,
        "react_summary": "Mock react summary.",
        "ticker": "AAPL",
        "factor_score": 0.75,
        "full_react_output": "Thought: ... Action: decide",
    }


def _decision_need_recheck(missing="CATALYST_VERIFICATION", agent="SEARCH"):
    return {
        "status": "NEED_RECHECK",
        "missing_info_type": missing,
        "requested_agent": agent,
        "recheck_questions": ["Is this FDA approval or just designation?"],
        "ticker": "AAPL",
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _run_orchestrator(
    search_side_effects,
    memory_side_effects,
    skeptic_side_effects,
    decision_side_effects,
):
    """
    Patch all agent calls and run orchestrator.run_ticker().
    side_effects are lists consumed in order per call.
    """
    from swing.agents.orchestrator import run_ticker

    with patch("swing.agents.orchestrator._run_search") as mock_search, \
         patch("swing.agents.orchestrator._run_memory") as mock_memory, \
         patch("swing.agents.orchestrator._run_skeptic") as mock_skeptic, \
         patch("swing.agents.orchestrator._run_decision") as mock_decision, \
         patch("swing.agents.orchestrator._get_yahoo_articles", return_value=[]):

        # Patch _run_search/_run_memory to append a round into state
        def search_effect(state, recheck_questions=None):
            result = search_side_effects.pop(0)
            append_round(state["search"], result,
                         recheck_strategy="targeted_query" if recheck_questions else None)
        mock_search.side_effect = search_effect

        def memory_effect(state, strategy=None, recheck_questions=None):
            result = memory_side_effects.pop(0)
            append_round(state["memory"], result, recheck_strategy=strategy)
        mock_memory.side_effect = memory_effect

        def skeptic_effect(state):
            result = skeptic_side_effects.pop(0)
            append_round(state["skeptic"], result)
            return result
        mock_skeptic.side_effect = skeptic_effect

        mock_decision.side_effect = decision_side_effects

        state, decision = run_ticker(
            ticker="AAPL",
            signal_direction="BUY",
            factor_score=0.75,
            regime="TRENDING",
            factors_used=["momentum_20d"],
            yahoo_articles=[],
        )

    return state, decision, mock_search, mock_memory, mock_skeptic, mock_decision


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestHappyPath:
    """No recheks — straight through the pipeline."""

    def test_call_counts(self):
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[_search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[_skeptic_result(concern="LOW", needs_recheck=False)],
            decision_side_effects=[_decision_decide()],
        )
        assert ms.call_count == 1
        assert mm.call_count == 1
        assert msk.call_count == 1
        assert md.call_count == 1

    def test_decision_is_returned(self):
        _, decision, *_ = _run_orchestrator(
            search_side_effects=[_search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[_skeptic_result()],
            decision_side_effects=[_decision_decide(signal="STRONG_BUY", confidence=88)],
        )
        assert decision["status"] == "DECIDE"
        assert decision["signal"] == "STRONG_BUY"
        assert decision["confidence"] == 88

    def test_no_forced_decision(self):
        state, _, *_ = _run_orchestrator(
            search_side_effects=[_search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[_skeptic_result()],
            decision_side_effects=[_decision_decide()],
        )
        assert state["forced_decision"] is False
        assert state["recheck_count"] == 0


class TestSkepticRecheck:
    """Skeptic needs_recheck=True → Search re-runs → Skeptic re-audits."""

    def test_search_called_twice(self):
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result(catalyst_type="FDA_APPROVAL")],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[
                _skeptic_result(concern="HIGH", needs_recheck=True,
                                questions=["Is this FDA approval or designation?"]),
                _skeptic_result(concern="MEDIUM", needs_recheck=False),
            ],
            decision_side_effects=[_decision_decide()],
        )
        assert ms.call_count == 2
        assert msk.call_count == 2
        assert state["skeptic_block_used"] is True
        assert state["recheck_count"] == 1

    def test_skeptic_recheck_delta_computed(self):
        state, *_ = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result(catalyst_type="FDA_APPROVAL")],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[
                _skeptic_result(concern="HIGH", needs_recheck=True, questions=["q"]),
                _skeptic_result(concern="LOW"),
            ],
            decision_side_effects=[_decision_decide()],
        )
        # Skeptic had 2 rounds, delta should capture concern change
        assert state["skeptic"]["delta"] is not None
        assert "concern_level" in state["skeptic"]["delta"]

    def test_skeptic_cannot_block_twice(self):
        """Second skeptic recheck request must be ignored."""
        state, _, ms, *_ = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[
                _skeptic_result(concern="HIGH", needs_recheck=True, questions=["q"]),
                _skeptic_result(concern="HIGH", needs_recheck=True, questions=["q2"]),
            ],
            decision_side_effects=[_decision_decide()],
        )
        # Search called twice (one initial + one from first skeptic recheck)
        # Second needs_recheck must be ignored
        assert ms.call_count == 2
        assert state["recheck_count"] == 1


class TestDecisionRecheck:
    """Decision Agent requests NEED_RECHECK → correct agent dispatched."""

    def test_search_recheck_dispatched(self):
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result(catalyst_type="CONTRACT_WIN")],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[
                _skeptic_result(),
                _skeptic_result(),   # re-run after search recheck
            ],
            decision_side_effects=[
                _decision_need_recheck(missing="CATALYST_VERIFICATION", agent="SEARCH"),
                _decision_decide(signal="BUY", confidence=72),
            ],
        )
        assert ms.call_count == 2
        assert md.call_count == 2
        assert state["decision_recheck_used"] is True
        assert decision["signal"] == "BUY"

    def test_memory_recheck_dispatched(self):
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[_search_result()],
            memory_side_effects=[_memory_result(n_cases=0), _memory_result(n_cases=3)],
            skeptic_side_effects=[_skeptic_result()],
            decision_side_effects=[
                _decision_need_recheck(missing="HISTORICAL_CONTEXT", agent="MEMORY"),
                _decision_decide(signal="BUY", confidence=65),
            ],
        )
        assert mm.call_count == 2
        assert md.call_count == 2
        # Memory recheck should not re-run Search or Skeptic
        assert ms.call_count == 1
        assert msk.call_count == 1

    def test_second_decision_is_forced(self):
        state, _, *_, md = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[_skeptic_result(), _skeptic_result()],
            decision_side_effects=[
                _decision_need_recheck(),
                _decision_decide(),
            ],
        )
        # Second call to decision must have forced=True
        _, forced_kwarg = md.call_args_list[1]
        assert forced_kwarg.get("forced") is True or md.call_args_list[1][0][1] is True

    def test_decision_cannot_recheck_twice(self):
        """If decision_recheck_used, a second NEED_RECHECK triggers forced decision."""
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[_search_result(), _search_result()],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[_skeptic_result(), _skeptic_result()],
            decision_side_effects=[
                _decision_need_recheck(),
                # This second NEED_RECHECK should be impossible (forced=True in prompt)
                # but if it somehow comes back, orchestrator falls back to _run_decision forced
                _decision_decide(signal="NO_POSITION", confidence=0),
            ],
        )
        assert state["decision_recheck_used"] is True
        assert md.call_count == 2


class TestBothLimitsHit:
    """Skeptic recheck + Decision recheck both used → forced_decision."""

    def test_forced_decision_set(self):
        state, decision, ms, mm, msk, md = _run_orchestrator(
            search_side_effects=[
                _search_result(),
                _search_result(),   # skeptic recheck
                _search_result(),   # decision recheck
            ],
            memory_side_effects=[_memory_result()],
            skeptic_side_effects=[
                _skeptic_result(needs_recheck=True, questions=["q"]),
                _skeptic_result(needs_recheck=False),
                _skeptic_result(),   # after decision recheck
            ],
            decision_side_effects=[
                _decision_need_recheck(),
                _decision_decide(signal="BUY"),
            ],
        )
        assert state["recheck_count"] == 2
        assert state["skeptic_block_used"] is True
        assert state["decision_recheck_used"] is True
        assert state["forced_decision"] is True


class TestSharedStateInternals:
    """Unit tests for orchestrator_types helpers."""

    def test_append_round_no_delta_on_first(self):
        from swing.agents.orchestrator_types import AgentHistory
        h = AgentHistory(rounds=[], delta=None)
        append_round(h, {"catalyst_type": "EARNINGS_BEAT"})
        assert len(h["rounds"]) == 1
        assert h["delta"] is None   # no previous round to diff against

    def test_append_round_delta_on_second(self):
        from swing.agents.orchestrator_types import AgentHistory
        h = AgentHistory(rounds=[], delta=None)
        append_round(h, {"catalyst_type": "EARNINGS_BEAT", "catalyst_strength": "STRONG"})
        append_round(h, {"catalyst_type": "FDA_APPROVAL", "catalyst_strength": "STRONG"})
        assert h["delta"] is not None
        assert "catalyst_type" in h["delta"]
        assert "EARNINGS_BEAT" in h["delta"]
        assert "FDA_APPROVAL" in h["delta"]

    def test_no_change_delta(self):
        result = _compute_delta(
            {"catalyst_type": "EARNINGS_BEAT", "concern_level": "LOW"},
            {"catalyst_type": "EARNINGS_BEAT", "concern_level": "LOW"},
        )
        assert "No material changes" in result

    def test_latest_round_empty(self):
        from swing.agents.orchestrator_types import AgentHistory
        h = AgentHistory(rounds=[], delta=None)
        assert latest_round(h) is None

    def test_latest_round_returns_last(self):
        from swing.agents.orchestrator_types import AgentHistory
        h = AgentHistory(rounds=[], delta=None)
        append_round(h, {"val": 1})
        append_round(h, {"val": 2})
        assert latest_round(h)["val"] == 2

    def test_make_shared_state_defaults(self):
        state = make_shared_state("AAPL", "BUY", 0.75, "TRENDING", ["momentum_20d"])
        assert state["recheck_count"] == 0
        assert state["skeptic_block_used"] is False
        assert state["decision_recheck_used"] is False
        assert state["forced_decision"] is False
        assert state["steps"] == []
        assert state["search"]["rounds"] == []
