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
import warnings

import anthropic

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

Return ONLY the JSON after your action. No markdown, no extra text."""


FEW_SHOT = """
EXAMPLES:

Example 1 — Strong buy, news confirms:
  Factor: LONG 0.89, Regime: TRENDING
  Search: CONTRACT_WIN, STRONG, SUPPORTS, "DoD contract $165M confirmed"
  Memory: win_rate=68%, n=23, HIGH confidence
  Thought: Factor very strong. News directly confirms with hard dollar amount.
           Historical win rate 68% with good sample size. Regime favors momentum.
           No binary risk events. High conviction.
  Action: decide
  Output: {"signal": "STRONG_BUY", "confidence": 87, "news_alignment": "SUPPORTS",
           "reason": "DoD contract win confirms momentum; 68% historical win rate supports.",
           "risk_flag": "none", "holding_period_days": 8,
           "react_summary": "Strong factor + confirmed catalyst + reliable history = high conviction long."}

Example 2 — No position, binary risk:
  Factor: SHORT 0.08, Regime: NEUTRAL
  Search: FDA_FAST_TRACK, MODERATE, CONTRADICTS, "FDA Fast Track designation announced"
  Memory: win_rate=52%, n=8, LOW confidence (sample too small)
  Thought: FDA Fast Track is NOT approval — price likely overstated the catalyst.
           BUT there may be follow-on FDA news. Short signal contradicted by positive news.
           Sample too small to trust historical stats. Binary event risk remains.
  Action: no_position(FDA binary risk — insufficient clarity to short or buy)
  Output: {"signal": "NO_POSITION", "confidence": 0, "news_alignment": "CONTRADICTS",
           "reason": "FDA Fast Track contradicts short; binary event risk too high.",
           "risk_flag": "FDA binary event within 48h", "holding_period_days": 0,
           "react_summary": "News contradicts signal and binary risk remains — pass."}

Example 3 — Volatile regime, weak signal:
  Factor: LONG 0.61, Regime: VOLATILE
  Search: NO_CATALYST, NONE, NEUTRAL, "No significant news found"
  Memory: knowledge rule: "Volatile regime without catalyst — reversal often fails"
  Thought: Regime is VOLATILE. Factor score 0.61 is moderate, not strong.
           No catalyst to explain the move. Knowledge rule warns against this setup.
           Volatile markets punish weak signals without catalysts.
  Action: no_position(weak signal in volatile regime with no catalyst)
  Output: {"signal": "NO_POSITION", "confidence": 0, "news_alignment": "NEUTRAL",
           "reason": "No catalyst in volatile regime; factor signal insufficient.",
           "risk_flag": "volatile regime, no catalyst", "holding_period_days": 0,
           "react_summary": "Volatile regime + no catalyst + moderate factor = not enough edge to trade."}
"""


def _extract_json(text: str) -> dict:
    """Extract JSON from final decision output."""
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except Exception:
        pass
    return {
        "signal": "NEUTRAL",
        "confidence": 0,
        "news_alignment": "NEUTRAL",
        "reason": "Decision agent parse error.",
        "risk_flag": "parse_error",
        "holding_period_days": 0,
        "react_summary": "Parse failed.",
    }


def run(context: dict) -> dict:
    """
    Run the Decision Agent for a single ticker.

    Args:
        context: output of merge.merge() — contains ticker, factor_score,
                 regime, search, memory, regime_hint, holding_period_hint

    Returns:
        Final decision dict with signal, confidence, reason, holding_period_days.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    ticker = context.get("ticker", "UNKNOWN")
    signal_direction = context.get("signal_direction", "LONG")
    factor_score = context.get("factor_score", 0.5)
    regime = context.get("regime", "NEUTRAL")
    search = context.get("search", {})
    memory = context.get("memory", {})

    # Format knowledge rules
    rules_text = "\n".join(
        f"  - {r}" for r in memory.get("knowledge_rules", [])
    ) or "  - No specific rules found."

    # Format memory stats
    if memory.get("win_rate") is not None:
        stats_text = (
            f"win_rate={memory['win_rate']:.0f}%, "
            f"avg_return={memory.get('avg_return', 0):+.1f}%, "
            f"n={memory.get('sample_size', 0)}, "
            f"query_level={memory.get('query_level', 'unknown')}"
        )
    else:
        stats_text = memory.get("stats_note", "No reliable stats available.")

    prompt = f"""{FEW_SHOT}

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

MEMORY AGENT FINDINGS:
  Historical stats : {stats_text}
  Confidence       : {memory.get('confidence_in_prior', 'NONE')}
  Context          : {memory.get('context_summary', 'No context.')}
  Knowledge rules  :
{rules_text}

Start your Thought chain now, then output Action and final JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    result = _extract_json(raw)

    result["ticker"] = ticker
    result["factor_score"] = factor_score
    result["full_react_output"] = raw

    # Override holding_period_days with 0 for NO_POSITION
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
