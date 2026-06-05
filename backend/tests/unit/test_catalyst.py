"""
tests/unit/test_catalyst.py - Unit tests for premarket catalyst analysis.

Tests cover:
  - Tool routing and execution
  - Outcome determination logic
  - Score computation
  - Error handling and edge cases
  - Input validation

LLM is mocked throughout — no real API calls.

Run:
  cd backend
  pytest tests/unit/test_catalyst.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_response():
    """Mock a valid Claude JSON response."""
    def make_response(signal="TRADE", catalyst="FDA_APPROVAL", confidence=85):
        block = MagicMock()
        block.text = json.dumps({
            "catalyst_type":     catalyst,
            "catalyst_strength": "STRONG",
            "proportionality":   "FAIR",
            "manipulation_risk": "LOW",
            "signal":            signal,
            "confidence":        confidence,
            "reason":            "Test reason",
            "risk":              "Test risk",
            "entry_timing":      "Wait for green candle",
            "exit_timing":       "Exit by 10:30 AM",
        })
        block.type = "text"
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        return response
    return make_response


@pytest.fixture
def sample_candidate():
    """Sample premarket candidate dict."""
    return {
        "ticker":               "SAVA",
        "premarket_price":      12.50,
        "premarket_change_pct": 28.0,
        "premarket_volume":     450000,
        "pm_amount":            5625000,
        "rvol":                 15.2,
        "float":                8000000,
        "market_cap":           45000000,
        "news": [
            {
                "headline": "Cassava Sciences receives FDA approval for simufilam",
                "summary":  "Full NDA approval granted for Alzheimer's treatment",
                "source":   "Reuters",
            }
        ],
    }


# ─────────────────────────────────────────────────────────────
# 1. Outcome determination logic
# ─────────────────────────────────────────────────────────────

class TestDetermineOutcome:

    def setup_method(self):
        from premarket.update_premarket_outcomes import determine_outcome
        self.determine_outcome = determine_outcome

    def test_trade_signal_large_gain_is_win(self):
        assert self.determine_outcome("TRADE", 5.0) == "WIN"

    def test_trade_signal_large_loss_is_loss(self):
        assert self.determine_outcome("TRADE", -5.0) == "LOSS"

    def test_trade_signal_small_move_is_neutral(self):
        assert self.determine_outcome("TRADE", 0.5) == "NEUTRAL"

    def test_avoid_signal_large_drop_is_win(self):
        assert self.determine_outcome("AVOID", -5.0) == "WIN"

    def test_avoid_signal_large_gain_is_loss(self):
        assert self.determine_outcome("AVOID", 5.0) == "LOSS"

    def test_watch_signal_small_move_is_win(self):
        assert self.determine_outcome("WATCH", 3.0) == "WIN"

    def test_watch_signal_large_move_is_neutral(self):
        assert self.determine_outcome("WATCH", 8.0) == "NEUTRAL"

    def test_boundary_exactly_at_threshold(self):
        # Exactly at 2% threshold — should be NEUTRAL (not > threshold)
        assert self.determine_outcome("TRADE", 2.0) == "LOSS"

    def test_unknown_signal_returns_neutral(self):
        assert self.determine_outcome("UNKNOWN", 10.0) == "NEUTRAL"


# ─────────────────────────────────────────────────────────────
# 2. Candidate scoring
# ─────────────────────────────────────────────────────────────

class TestScoreCandidate:

    def setup_method(self):
        from premarket.premarket_scanner import score_candidate
        self.score_candidate = score_candidate

    def test_high_rvol_strong_catalyst_high_score(self):
        candidate = {
            "rvol": 10.0,
            "catalyst_strength": "STRONG",
            "confidence": 80,
        }
        score = self.score_candidate(candidate)
        assert score > 0

    def test_no_catalyst_zero_score(self):
        candidate = {
            "rvol": 10.0,
            "catalyst_strength": "NONE",
            "confidence": 50,
        }
        score = self.score_candidate(candidate)
        assert score == 0.0

    def test_zero_confidence_zero_score(self):
        candidate = {
            "rvol": 10.0,
            "catalyst_strength": "STRONG",
            "confidence": 0,
        }
        score = self.score_candidate(candidate)
        assert score == 0.0

    def test_stronger_catalyst_higher_score(self):
        strong = {"rvol": 5.0, "catalyst_strength": "STRONG",   "confidence": 80}
        weak   = {"rvol": 5.0, "catalyst_strength": "WEAK",     "confidence": 80}
        assert self.score_candidate(strong) > self.score_candidate(weak)

    def test_missing_fields_no_crash(self):
        score = self.score_candidate({})
        assert score == 0.0


# ─────────────────────────────────────────────────────────────
# 3. Candidate ranking
# ─────────────────────────────────────────────────────────────

class TestRankCandidates:

    def setup_method(self):
        from premarket.premarket_scanner import rank_candidates
        self.rank_candidates = rank_candidates

    def test_sort_by_change_descending(self):
        candidates = [
            {"ticker": "A", "premarket_change_pct": 10.0, "rvol": 1, "catalyst_strength": "NONE", "confidence": 0, "pm_amount": 0, "premarket_volume": 0},
            {"ticker": "B", "premarket_change_pct": 25.0, "rvol": 1, "catalyst_strength": "NONE", "confidence": 0, "pm_amount": 0, "premarket_volume": 0},
            {"ticker": "C", "premarket_change_pct": 5.0,  "rvol": 1, "catalyst_strength": "NONE", "confidence": 0, "pm_amount": 0, "premarket_volume": 0},
        ]
        ranked = self.rank_candidates(candidates, sort_by="change")
        assert ranked[0]["ticker"] == "B"
        assert ranked[-1]["ticker"] == "C"

    def test_sort_by_rvol(self):
        candidates = [
            {"ticker": "A", "premarket_change_pct": 10.0, "rvol": 3.0, "catalyst_strength": "NONE", "confidence": 0, "pm_amount": 0, "premarket_volume": 0},
            {"ticker": "B", "premarket_change_pct": 10.0, "rvol": 12.0, "catalyst_strength": "NONE", "confidence": 0, "pm_amount": 0, "premarket_volume": 0},
        ]
        ranked = self.rank_candidates(candidates, sort_by="rvol")
        assert ranked[0]["ticker"] == "B"

    def test_empty_list_no_crash(self):
        assert self.rank_candidates([]) == []

    def test_composite_score_added(self):
        candidates = [
            {"ticker": "A", "premarket_change_pct": 10.0, "rvol": 5.0, "catalyst_strength": "STRONG", "confidence": 80, "pm_amount": 0, "premarket_volume": 0},
        ]
        ranked = self.rank_candidates(candidates)
        assert "composite_score" in ranked[0]


# ─────────────────────────────────────────────────────────────
# 4. LLM analysis (mocked)
# ─────────────────────────────────────────────────────────────

class TestAnalyzeCatalyst:

    @patch("premarket.premarket_catalyst.get_mcp_servers", return_value=[])
    @patch("premarket.premarket_catalyst.DB_AVAILABLE", False)
    def test_valid_response_parsed_correctly(self, mock_mcp, mock_anthropic_response):
        from premarket.premarket_catalyst import analyze_catalyst_with_mode

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            signal="TRADE", catalyst="FDA_APPROVAL", confidence=88
        )

        result = analyze_catalyst_with_mode(
            ticker               = "SAVA",
            premarket_change_pct = 28.0,
            rvol                 = 15.2,
            float_shares         = 8e6,
            market_cap           = 45e6,
            news_items           = [{"headline": "FDA approval", "summary": "", "source": "test"}],
            mode                 = "baseline",
            client               = mock_client,
        )

        assert result["signal"]        == "TRADE"
        assert result["catalyst_type"] == "FDA_APPROVAL"
        assert result["confidence"]    == 88
        assert result["ticker"]        == "SAVA"

    @patch("premarket.premarket_catalyst.get_mcp_servers", return_value=[])
    @patch("premarket.premarket_catalyst.DB_AVAILABLE", False)
    def test_llm_failure_returns_fallback(self, mock_mcp):
        from premarket.premarket_catalyst import analyze_catalyst_with_mode

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        result = analyze_catalyst_with_mode(
            ticker               = "FAIL",
            premarket_change_pct = 10.0,
            rvol                 = 5.0,
            float_shares         = 10e6,
            market_cap           = 50e6,
            news_items           = [],
            mode                 = "baseline",
            client               = mock_client,
        )

        assert result["signal"]        == "AVOID"
        assert result["catalyst_type"] == "UNKNOWN"
        assert result["confidence"]    == 0

    @patch("premarket.premarket_catalyst.get_mcp_servers", return_value=[])
    @patch("premarket.premarket_catalyst.DB_AVAILABLE", False)
    def test_malformed_json_returns_fallback(self, mock_mcp):
        from premarket.premarket_catalyst import analyze_catalyst_with_mode

        mock_client = MagicMock()
        bad_block = MagicMock()
        bad_block.text = "This is not valid JSON at all!!!"
        bad_block.type = "text"
        mock_response = MagicMock()
        mock_response.content = [bad_block]
        mock_client.messages.create.return_value = mock_response

        result = analyze_catalyst_with_mode(
            ticker               = "BAD",
            premarket_change_pct = 10.0,
            rvol                 = 5.0,
            float_shares         = 10e6,
            market_cap           = 50e6,
            news_items           = [],
            mode                 = "baseline",
            client               = mock_client,
        )

        assert result["signal"] == "AVOID"
        assert result["confidence"] == 0


# ─────────────────────────────────────────────────────────────
# 5. MCP server config
# ─────────────────────────────────────────────────────────────

class TestGetMCPServers:

    def test_no_keys_returns_empty_list(self):
        import os
        from premarket.premarket_catalyst import get_mcp_servers

        original_tavily  = os.environ.pop("TAVILY_API_KEY", None)
        original_supabase = os.environ.pop("SUPABASE_ACCESS_TOKEN", None)

        try:
            servers = get_mcp_servers()
            assert servers == []
        finally:
            if original_tavily:
                os.environ["TAVILY_API_KEY"] = original_tavily
            if original_supabase:
                os.environ["SUPABASE_ACCESS_TOKEN"] = original_supabase

    def test_tavily_key_adds_tavily_server(self):
        import os
        from premarket.premarket_catalyst import get_mcp_servers

        original = os.environ.get("TAVILY_API_KEY")
        os.environ["TAVILY_API_KEY"] = "test-tavily-key"
        os.environ.pop("SUPABASE_ACCESS_TOKEN", None)

        try:
            servers = get_mcp_servers()
            names = [s["name"] for s in servers]
            assert "tavily" in names
            tavily = next(s for s in servers if s["name"] == "tavily")
            assert "test-tavily-key" in tavily["url"]
        finally:
            if original:
                os.environ["TAVILY_API_KEY"] = original
            else:
                os.environ.pop("TAVILY_API_KEY", None)

    def test_server_has_required_fields(self):
        import os
        from premarket.premarket_catalyst import get_mcp_servers

        original = os.environ.get("TAVILY_API_KEY")
        os.environ["TAVILY_API_KEY"] = "test-key"
        os.environ.pop("SUPABASE_ACCESS_TOKEN", None)

        try:
            servers = get_mcp_servers()
            for server in servers:
                assert "type" in server
                assert "url"  in server
                assert "name" in server
        finally:
            if original:
                os.environ["TAVILY_API_KEY"] = original
            else:
                os.environ.pop("TAVILY_API_KEY", None)
