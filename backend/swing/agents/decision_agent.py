"""
agents/decision_agent.py - Decision Agent for Swing Trade Phase 2.

Responsibilities:
  - Take merged context (search + memory) for one ticker
  - Run a ReAct reasoning loop to produce a final signal
  - Output: signal, confidence, reason, holding_period_days, risk_flag
  - Support no_position output when information is insufficient or
    regime volatility is too high

Signal taxonomy:
  STRONG_BUY  : high-confidence long, news strongly supports factor
  BUY         : moderate-confidence long
  NEUTRAL     : no clear edge, skip
  SHORT       : moderate-confidence short
  STRONG_SHORT: high-confidence short
  NO_POSITION : explicit pass — insufficient info, high vol, or correlation

ReAct in Decision Agent:
  Unlike Search/Memory agents, Decision Agent rarely needs tool calls.
  The ReAct here is primarily for explicit Thought chain logging —
  visible in Langfuse, useful for eval and debug.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import anthropic

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    ANALYST_MODEL, DECISION_AGENT_MAX_TOKENS, MAX_DECISION_RETRIES,
    MAX_KNOWLEDGE_RULES, MAX_ANALYST_RATINGS, MAX_SEC_FILINGS,
)

warnings.filterwarnings("ignore")

SYSTEM_PROMPT = """You are a quantitative swing trade decision agent.

You receive a merged context containing:
- Factor score and signal direction from the quant scanner
- News catalyst summary from the Search Agent
- Historical win rates and trading rules from the Memory Agent

Your job is to produce a final trading decision using explicit reasoning.

You MUST follow this ReAct format:

Thought: [reason through each piece of evidence]
Action: decide | no_position(reason)

Use Action: no_position(reason) when:
  - catalyst_strength is NONE and news_alignment is NEUTRAL/CONTRADICTS
  - risk_flag contains earnings/FDA binary event within 48 hours
  - confidence_in_prior is NONE and catalyst is ambiguous
  - regime is VOLATILE and factor signal is weak (score < 0.3 for LONG, > 0.7 for SHORT)

Use Action: decide when you have enough conviction to output a signal.

After Action: decide, output JSON in this exact format:
{
  "signal": "STRONG_BUY|BUY|NEUTRAL|SHORT|STRONG_SHORT|NO_POSITION",
  "confidence": integer 0-100,
  "news_alignment": "SUPPORTS|CONTRADICTS|NEUTRAL",
  "reason": "one sentence max 25 words, specific to this stock",
  "risk_flag": "none or specific risk",
  "holding_period_days": integer,
  "react_summary": "2-3 sentence summary of your reasoning chain"
}

Confidence calibration:
  85-100: Factor + news both strongly confirm, reliable historical stats
  70-84 : Factor confirmed by news, moderate historical support
  50-69 : Factor signal only, neutral news, limited history
  30-49 : Mixed signals or contradicting news
  10-29 : News contradicts signal or major risk present
  0-9   : Active risk event (earnings, regulatory, fraud)

After Action: no_position, output JSON:
{
  "signal": "NO_POSITION",
  "confidence": 0,
  "news_alignment": "NEUTRAL",
  "reason": "brief reason for pass",
  "risk_flag": "stated reason",
  "holding_period_days": 0,
  "react_summary": "why you chose to pass"
}

Return ONLY the JSON after your action. No markdown, no extra text.
Keep your Thought to 3 sentences maximum. Do not use ** or any markdown formatting."""


FEW_SHOT = """
EXAMPLES (abbreviated):
1. LONG 0.89 TRENDING + CONTRACT_WIN STRONG + win_rate=68% n=23 HIGH
   → STRONG_BUY conf=87 hold=8d | "DoD contract confirms momentum; 68% win rate supports."

2. SHORT 0.08 NEUTRAL + FDA_FAST_TRACK CONTRADICTS + n=8 LOW
   → NO_POSITION | "FDA Fast Track contradicts short; binary event risk too high."

3. LONG 0.61 VOLATILE + NO_CATALYST NONE + rule:"reversal fails without catalyst"
   → NO_POSITION | "No catalyst in volatile regime; signal insufficient."
"""


VALID_SIGNALS = {"STRONG_BUY", "BUY", "NEUTRAL", "SHORT", "STRONG_SHORT", "NO_POSITION"}
VALID_ALIGNMENTS = {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}


def _extract_json(text: str) -> dict | None:
    """Extract JSON from decision output. Returns None on parse failure."""
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
    """
    Validate required fields and value ranges.
    Returns a list of error strings (empty = valid).
    """
    errors = []

    signal = result.get("signal")
    if signal not in VALID_SIGNALS:
        errors.append(f"signal must be one of {VALID_SIGNALS}, got {signal!r}")

    confidence = result.get("confidence")
    if not isinstance(confidence, int) or not (0 <= confidence <= 100):
        errors.append(f"confidence must be int 0-100, got {confidence!r}")

    alignment = result.get("news_alignment")
    if alignment not in VALID_ALIGNMENTS:
        errors.append(f"news_alignment must be one of {VALID_ALIGNMENTS}, got {alignment!r}")

    holding = result.get("holding_period_days")
    if not isinstance(holding, int) or holding < 0:
        errors.append(f"holding_period_days must be non-negative int, got {holding!r}")

    if not result.get("reason"):
        errors.append("reason is missing or empty")

    return errors


def _format_cases(cases: list) -> str:
    if not cases:
        return "  - No similar past cases found."
    lines = []
    for c in cases:
        ret = f"return={c['actual_return']:+.1f}%" if c.get("actual_return") is not None else "outcome=pending"
        lines.append(
            f"  - {c['scan_date']} | {c['ticker']} | {c['signal']} conf={c['confidence']}% | "
            f"{c['catalyst_type']} | {ret} (sim={c['similarity']:.2f})"
        )
    return "\n".join(lines)


def _format_events(events: list) -> str:
    if not events:
        return "  - None in next 14 days."
    return "\n".join(
        f"  - {e['event_type']} on {e['event_date']} ({e['days_away']}d away)"
        for e in events
    )


def _format_ratings(ratings: list) -> str:
    if not ratings:
        return "  - No recent analyst rating changes."
    return "\n".join(
        f"  - {r['summary']} ({r['rating_date']})"
        for r in ratings[:MAX_ANALYST_RATINGS]
    )


def _format_sec(filings: list) -> str:
    if not filings:
        return "  - No recent SEC filings."
    lines = []
    for f in filings:
        km = f.get("key_metrics", {})
        sentiment = km.get("sentiment", "")
        lines.append(
            f"  - {f['filing_type']} {f['filed_date']}: {f['summary']}"
            + (f" [sentiment={sentiment}]" if sentiment else "")
        )
    return "\n".join(lines)


def _build_prompt(context: dict) -> str:
    ticker           = context.get("ticker", "UNKNOWN")
    signal_direction = context.get("signal_direction", "LONG")
    factor_score     = context.get("factor_score", 0.5)
    regime           = context.get("regime", "NEUTRAL")
    search           = context.get("search", {})
    memory           = context.get("memory", {})

    rules_text = "\n".join(
        f"  - {r}" for r in memory.get("knowledge_rules", [])[:MAX_KNOWLEDGE_RULES]
    ) or "  - No specific rules found."

    event_risk = memory.get("event_risk_flag")
    risk_line  = f"  *** BINARY EVENT: {event_risk} — consider NO_POSITION ***" if event_risk else ""

    return f"""{FEW_SHOT}

─────────────────────────────────────
NOW DECIDE FOR THIS STOCK:
─────────────────────────────────────
Ticker          : {ticker}
Signal direction: {signal_direction}
Factor score    : {factor_score:.3f} (0=weakest, 1=strongest in universe)
Market regime   : {regime}
Regime hint     : {context.get('regime_hint', '')}
Holding hint    : {context.get('holding_period_hint', '')}

SEARCH AGENT FINDINGS:
  Catalyst type    : {search.get('catalyst_type', 'UNKNOWN')}
  Catalyst strength: {search.get('catalyst_strength', 'UNKNOWN')}
  News alignment   : {search.get('news_alignment', 'NEUTRAL')}
  Summary          : {search.get('summary', 'No summary.')}
  Risk flag        : {search.get('risk_flag', 'none')}

UPCOMING EVENTS (risk check):
{_format_events(memory.get('upcoming_events', []))}
{risk_line}

ANALYST RATINGS (last 30 days):
{_format_ratings(memory.get('analyst_ratings', []))}

SEC FILINGS (recent 8-K / 10-Q):
{_format_sec(memory.get('sec_filings', []))}

MEMORY AGENT FINDINGS:
  Context          : {memory.get('context_summary', 'No context.')}
  Knowledge rules  :
{rules_text}
  Similar past cases:
{_format_cases(memory.get('similar_cases', []))}

Start your Thought chain now, then output Action and final JSON."""


def run(context: dict) -> dict:
    """
    Run the Decision Agent for a single ticker.
    Retries up to MAX_DECISION_RETRIES times on schema validation failure.

    Args:
        context: output of merge.merge()

    Returns:
        Final decision dict with signal, confidence, reason, holding_period_days.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    ticker = context.get("ticker", "UNKNOWN")
    factor_score = context.get("factor_score", 0.5)

    prompt = _build_prompt(context)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_DECISION_RETRIES + 1):
        response = client.messages.create(
            model=ANALYST_MODEL,
            max_tokens=DECISION_AGENT_MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )

        raw = response.content[0].text
        result = _extract_json(raw)

        if result is None:
            errors = ["JSON parse failed"]
        else:
            errors = _validate(result)

        if not errors:
            break

        print(f"  [decision_agent] {ticker}: attempt {attempt+1} failed — {errors}")

        if attempt < MAX_DECISION_RETRIES:
            # Feed the error back so Claude can self-correct
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your output has validation errors: {errors}. "
                    "Output ONLY the corrected JSON, no other text."
                ),
            })
        else:
            print(f"  [decision_agent] {ticker}: all retries exhausted, using fallback")
            return _fallback(ticker)

    result["ticker"]           = ticker
    result["factor_score"]     = factor_score
    result["full_react_output"] = raw

    if result.get("signal") == "NO_POSITION":
        result["holding_period_days"] = 0

    print(
        f"  [decision_agent] {ticker}: "
        f"{result.get('signal')} "
        f"(conf={result.get('confidence')}%, "
        f"hold={result.get('holding_period_days')}d)"
    )

    return result


def _fallback(ticker: str) -> dict:
    """Safe default when Decision Agent fails entirely."""
    return {
        "ticker": ticker,
        "signal": "NEUTRAL",
        "confidence": 0,
        "news_alignment": "NEUTRAL",
        "reason": "Decision agent failed — no signal produced.",
        "risk_flag": "agent_error",
        "holding_period_days": 0,
        "react_summary": "Agent failure.",
        "factor_score": 0.0,
        "full_react_output": "",
    }


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from merge import merge

    # Mock Search Agent output
    mock_search = {
        "catalyst_type": "CONTRACT_WIN",
        "catalyst_strength": "STRONG",
        "news_alignment": "SUPPORTS",
        "sources": ["DoD awards BBAI $165M contract"],
        "summary": "Major government contract win directly supports momentum signal.",
        "risk_flag": "none",
        "search_count": 1,
    }

    # Mock Memory Agent output
    mock_memory = {
        "has_stats": True,
        "win_rate": 68.0,
        "avg_return": 4.2,
        "sample_size": 23,
        "query_level": "regime",
        "knowledge_rules": [
            "Government contracts are sticky — revenue visibility improves.",
            "DoD contracts often cause sustained momentum over 5–10 days.",
        ],
        "confidence_in_prior": "HIGH",
        "context_summary": "TRENDING regime momentum signals win 68% historically.",
    }

    context = merge(
        ticker="GS",
        signal_direction="LONG",
        factor_score=0.896,
        regime="NEUTRAL",
        search_result=mock_search,
        memory_result=mock_memory,
    )

    print("Testing Decision Agent...\n")
    result = run(context)

    print("\n── Decision Agent Result ──")
    for k, v in result.items():
        if k == "full_react_output":
            print(f"  full_react_output: [{len(v)} chars]")
            print("  " + "\n  ".join(v.split("\n")[:6]))
        else:
            print(f"  {k}: {v}")