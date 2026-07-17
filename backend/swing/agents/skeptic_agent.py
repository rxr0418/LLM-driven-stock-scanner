"""
agents/skeptic_agent.py - Thesis audit agent for Swing Trade Phase 2.

Responsibilities:
  - Challenge the combined quant/search/memory thesis before final decision
  - Identify overstated catalysts, weak evidence, sample-size problems,
    binary event risk, and confidence overreach
  - Return a structured critique for the Decision Agent

The Skeptic Agent does not call other agents directly. It can request a
recheck, but the orchestrator decides whether to route follow-up work.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import anthropic

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SEARCH_AGENT_MODEL, SKEPTIC_AGENT_MAX_TOKENS
from logger import get_logger
from resilience import with_retry

log = get_logger(__name__)
warnings.filterwarnings("ignore")


SYSTEM_PROMPT = """You are a skeptical investment reviewer for a stock scanner.

Your job is NOT to make the final trading decision.
Your job is to audit the proposed thesis and find flaws:
- overstated or ambiguous catalysts
- headline misreadings
- weak or missing evidence
- historical sample-size problems
- event risk that should cap confidence
- reasons the move may already be priced in

You can request rechecks, but you cannot call tools or other agents directly.
Return ONLY valid JSON:
{
  "thesis_quality": "STRONG|MIXED|WEAK",
  "concern_level": "LOW|MEDIUM|HIGH",
  "needs_recheck": boolean,
  "confidence_cap": integer 0-100,
  "concerns": ["short concern 1", "short concern 2"],
  "requested_recheck_questions": ["question for Search/Memory Agent if needed"],
  "summary": "one sentence audit summary"
}

Guidelines:
- Use HIGH concern when catalyst classification is likely wrong, a binary event is imminent (within 5 days), or news clearly contradicts the factor direction.
- Use MEDIUM concern when evidence is thin, sample size is small, catalyst may be priced in, or historical cases are limited.
- Use LOW concern when factor and news are aligned, even if historical cases are limited or memory context is sparse.
- Do NOT penalize for lack of historical cases alone — a new system with limited history is expected.
- confidence_cap is a cap for the Decision Agent, not a final confidence score.
- Typical caps: HIGH concern → 30-50, MEDIUM concern → 55-70, LOW concern → 75-90.
"""


def _extract_json(text: str) -> dict | None:
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except Exception:
        pass
    return None


def _validate(result: dict) -> list[str]:
    errors = []
    if result.get("thesis_quality") not in {"STRONG", "MIXED", "WEAK"}:
        errors.append("thesis_quality must be STRONG, MIXED, or WEAK")
    if result.get("concern_level") not in {"LOW", "MEDIUM", "HIGH"}:
        errors.append("concern_level must be LOW, MEDIUM, or HIGH")
    if not isinstance(result.get("needs_recheck"), bool):
        errors.append("needs_recheck must be boolean")
    cap = result.get("confidence_cap")
    if not isinstance(cap, int) or not (0 <= cap <= 100):
        errors.append("confidence_cap must be int 0-100")
    if not isinstance(result.get("concerns"), list):
        errors.append("concerns must be a list")
    if not isinstance(result.get("requested_recheck_questions"), list):
        errors.append("requested_recheck_questions must be a list")
    if not result.get("summary"):
        errors.append("summary is required")
    return errors


def _format_cases(cases: list) -> str:
    if not cases:
        return "No similar cases."
    lines = []
    for c in cases[:3]:
        outcome = (
            f"return={c.get('actual_return'):+.1f}%"
            if c.get("actual_return") is not None
            else "outcome=pending"
        )
        lines.append(
            f"- {c.get('ticker')} {c.get('signal')} conf={c.get('confidence')} "
            f"catalyst={c.get('catalyst_type')} {outcome}"
        )
    return "\n".join(lines)


def _fallback_result(ticker: str, search_result: dict, memory_result: dict) -> dict:
    """Deterministic audit when LLM critique is unavailable."""
    concerns = []
    cap = 80
    concern_level = "LOW"
    thesis_quality = "STRONG"
    needs_recheck = False

    catalyst_strength = search_result.get("catalyst_strength", "NONE")
    alignment = search_result.get("news_alignment", "NEUTRAL")
    catalyst_type = search_result.get("catalyst_type", "OTHER")

    if catalyst_strength in {"NONE", "WEAK"}:
        concerns.append("Catalyst evidence is weak or absent.")
        cap = min(cap, 50)
        concern_level = "MEDIUM"
        thesis_quality = "MIXED"

    if alignment == "CONTRADICTS":
        concerns.append("News contradicts the factor direction.")
        cap = min(cap, 35)
        concern_level = "HIGH"
        thesis_quality = "WEAK"

    if catalyst_type == "FDA_FAST_TRACK":
        concerns.append("FDA Fast Track may be misread as approval.")
        cap = min(cap, 45)
        concern_level = "HIGH"
        thesis_quality = "WEAK"
        needs_recheck = True

    if memory_result.get("event_risk_flag"):
        concerns.append(f"Upcoming binary event: {memory_result['event_risk_flag']}.")
        cap = min(cap, 30)
        concern_level = "HIGH"
        thesis_quality = "WEAK"

    if not memory_result.get("similar_cases"):
        concerns.append("No similar historical cases with outcomes.")
        cap = min(cap, 75)
        if concern_level == "LOW":
            concern_level = "MEDIUM"

    if not concerns:
        concerns.append("No major thesis flaws detected.")

    return {
        "ticker": ticker,
        "thesis_quality": thesis_quality,
        "concern_level": concern_level,
        "needs_recheck": needs_recheck,
        "confidence_cap": cap,
        "concerns": concerns[:4],
        "requested_recheck_questions": (
            ["Verify whether the catalyst classification is precise."]
            if needs_recheck else []
        ),
        "summary": concerns[0],
        "fallback": True,
    }


def run(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    search_result: dict,
    memory_result: dict,
) -> dict:
    """
    Audit a candidate thesis and return critique context for Decision Agent.
    """
    print(f"  [skeptic_agent] {ticker}: auditing thesis")

    prompt = f"""Audit this stock scanner thesis.

Ticker: {ticker}
Signal direction: {signal_direction}
Factor score: {factor_score:.3f}
Market regime: {regime}

SEARCH AGENT:
  catalyst_type: {search_result.get('catalyst_type', 'UNKNOWN')}
  catalyst_strength: {search_result.get('catalyst_strength', 'NONE')}
  news_alignment: {search_result.get('news_alignment', 'NEUTRAL')}
  summary: {search_result.get('summary', 'No summary.')}
  risk_flag: {search_result.get('risk_flag', 'none')}
  sources: {search_result.get('sources', [])[:3]}

MEMORY AGENT:
  context_summary: {memory_result.get('context_summary', 'No context.')}
  event_risk_flag: {memory_result.get('event_risk_flag')}
  upcoming_events: {memory_result.get('upcoming_events', [])[:3]}
  analyst_ratings: {memory_result.get('analyst_ratings', [])[:3]}
  sec_filings: {memory_result.get('sec_filings', [])[:2]}
  similar_cases:
{_format_cases(memory_result.get('similar_cases', []))}

Return the audit JSON only."""

    log.info("auditing thesis", extra={"ticker": ticker})
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

        @with_retry(label="skeptic_agent/anthropic")
        def _create():
            return client.messages.create(
                model=SEARCH_AGENT_MODEL,
                max_tokens=SKEPTIC_AGENT_MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )

        raw = _create().content[0].text
        result = _extract_json(raw)
        errors = _validate(result) if result else ["JSON parse failed"]
        if errors:
            log.warning("invalid output, using fallback", extra={"ticker": ticker, "errors": errors})
            return _fallback_result(ticker, search_result, memory_result)

        result["ticker"] = ticker
        result["react_trace"] = raw
        log.info("done", extra={"ticker": ticker, "concern_level": result["concern_level"],
                                "confidence_cap": result["confidence_cap"]})
        return result
    except Exception as e:
        log.error("failed, using fallback", extra={"ticker": ticker}, exc_info=True)
        return _fallback_result(ticker, search_result, memory_result)
