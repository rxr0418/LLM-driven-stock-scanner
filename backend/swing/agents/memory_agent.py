"""
agents/memory_agent.py - Memory Agent for Swing Trade Phase 2.

Responsibilities:
  - Query Supabase swing_stats for historical win rates
  - Query knowledge table for relevant trading rules
  - Use a ReAct loop to decide: is the data sufficient? should I
    fall back to a broader query? should I skip stats entirely?
  - Return structured memory context for the Decision Agent

Query key priority (to avoid cross-market error transfer):
  1. (regime, signal_type, factor)  — most specific
  2. (regime, signal_type)          — drop factor
  3. (signal_type)                  — drop regime
  4. knowledge rules only           — if sample_size < MIN_SAMPLE everywhere

Min sample threshold: 10 rows before trusting win rate stats.

Design note:
  Memory Agent is the only component that reads from Supabase
  during Phase 2. It also handles the write path after a signal
  is produced (called from main.py after Decision Agent returns).
"""

import json
import os
import sys
import warnings
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database import get_connection
from typing import Optional

import anthropic

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))

MIN_SAMPLE = 10

SYSTEM_PROMPT = """You are a memory retrieval agent for a swing trading system.

Your job is to find relevant historical data and trading rules from the database
to help the Decision Agent make a better judgment.

You MUST follow this exact ReAct format:

Thought: [what you know, what you need, whether current data is sufficient]
Action: query(sql) | stop
Observation: [query result - filled in by system]

Rules:
- Use Action: query(SELECT ...) to run SQL against Supabase
- Use Action: stop when you have enough context
- You have enough when:
  * You have win rate stats with sample_size >= 10, OR
  * You have confirmed no reliable stats exist and have knowledge rules
- Start specific, fall back to broader queries if sample_size < 10
- Maximum 4 queries allowed
- Always query the knowledge table for relevant rules

After Action: stop, output JSON in this exact format:
{
  "has_stats": true|false,
  "win_rate": float or null,
  "avg_return": float or null,
  "sample_size": integer or null,
  "query_level": "specific|regime|signal|none",
  "knowledge_rules": ["rule 1", "rule 2"],
  "confidence_in_prior": "HIGH|LOW|NONE",
  "context_summary": "one sentence summary of what history says"
}

Return ONLY the JSON after stop. No markdown, no extra text."""


def _run_query(sql: str) -> str:
    """
    Execute a SQL query against Supabase via psycopg2.
    Returns formatted string result.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        conn.close()

        if not rows:
            return "No rows returned."

        lines = [" | ".join(cols)]
        lines.append("-" * 60)
        for row in rows[:10]:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
        return "\n".join(lines)

    except Exception as e:
        return f"Query error: {e}"



def _query_knowledge(ticker: str, signal_type: str, regime: str) -> list:
    """
    Fetch relevant rules from the knowledge table.
    Used as a fallback when stats are insufficient.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT rule, context
            FROM knowledge
            WHERE is_active = true
            ORDER BY created_at DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [f"{r[0]} ({r[1]})" if r[1] else r[0] for r in rows]
    except Exception:
        return []


def _extract_json(text: str) -> dict:
    """Extract JSON from final LLM output."""
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except Exception:
        pass
    return {
        "has_stats": False,
        "win_rate": None,
        "avg_return": None,
        "sample_size": None,
        "query_level": "none",
        "knowledge_rules": [],
        "confidence_in_prior": "NONE",
        "context_summary": "Memory agent parse error — no historical context.",
    }


def _parse_action(text: str) -> tuple[str, Optional[str]]:
    """Parse Action line. Returns ('query', sql) or ('stop', None)."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            action = line[len("Action:"):].strip()
            if action.lower().startswith("query(") and action.endswith(")"):
                sql = action[len("query("):-1].strip()
                return "query", sql
            elif action.lower() == "stop":
                return "stop", None
    return "stop", None


def run(
    ticker: str,
    signal_direction: str,
    regime: str,
    factors_used: list,
) -> dict:
    """
    Run the Memory Agent ReAct loop for a single ticker.

    Args:
        ticker           : e.g. "AAPL"
        signal_direction : "LONG" or "SHORT"
        regime           : "TRENDING" | "VOLATILE" | "NEUTRAL"
        factors_used     : list of factor names from scanner

    Returns:
        Structured memory context dict for merge() and Decision Agent.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    factors_str = ", ".join(factors_used) if factors_used else "unknown"
    primary_factor = factors_used[0] if factors_used else "unknown"

    initial_message = f"""Retrieve historical context for this swing trade signal:

Ticker          : {ticker}
Signal direction: {signal_direction}
Market regime   : {regime}
Factors used    : {factors_str}

Database tables available:
  swing_stats  — columns: regime, signal, sample_size, avg_return_5d, win_rate_5d, avg_return_10d, win_rate_10d
  knowledge    — columns: rule, context, is_active

Start by querying the most specific level: regime='{regime}' AND signal='{signal_direction}'.
If sample_size < {MIN_SAMPLE}, fall back to broader queries.
Always query knowledge table for relevant rules."""

    messages = [{"role": "user", "content": initial_message}]
    query_count = 0
    react_trace = []
    MAX_QUERIES = 4

    print(f"  [memory_agent] {ticker}: starting ReAct loop")

    for turn in range(MAX_QUERIES + 2):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})
        react_trace.append({"turn": turn, "output": assistant_text})

        action_type, payload = _parse_action(assistant_text)

        if action_type == "stop":
            print(f"  [memory_agent] {ticker}: stopped after {query_count} queries")
            break

        if action_type == "query":
            if query_count >= MAX_QUERIES:
                messages.append({"role": "user", "content": "Max queries reached.\nAction: stop"})
                react_trace.append({"turn": turn, "forced": "max_queries"})
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                assistant_text = response.content[0].text
                messages.append({"role": "assistant", "content": assistant_text})
                react_trace.append({"turn": turn + 1, "output": assistant_text})
                break

            print(f"  [memory_agent] {ticker}: querying DB")
            result_str = _run_query(payload)
            query_count += 1

            messages.append({"role": "user", "content": f"Observation: {result_str}"})
            react_trace.append({"turn": turn, "query": payload[:80], "result_preview": result_str[:100]})

    # Extract final JSON
    final_text = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
    result = _extract_json(final_text)

    # Always append knowledge rules as a safety net
    if not result.get("knowledge_rules"):
        result["knowledge_rules"] = _query_knowledge(ticker, signal_direction, regime)

    result["ticker"] = ticker
    result["react_trace"] = react_trace

    return result


def write_decision_snapshot(
    signal_id: str,
    ticker: str,
    signal: str,
    confidence: int,
    regime: str,
    factors_used: list,
    holding_period_days: int,
    search_summary: dict,
    memory_context: dict,
    react_trace: list,
    price_at_scan: float,
) -> bool:
    """
    Write decision snapshot to swing_results after signal is produced.
    Called from main.py once Decision Agent returns.

    Returns True on success, False on failure.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO swing_results (
                signal_id, ticker, signal, confidence, regime,
                factors_used, holding_period_days,
                search_summary, memory_context, react_trace,
                price_at_scan, scan_date
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, NOW()::date
            )
            ON CONFLICT (signal_id) DO NOTHING
        """, (
            signal_id,
            ticker,
            signal,
            confidence,
            regime,
            json.dumps(factors_used),
            holding_period_days,
            json.dumps(search_summary),
            json.dumps(memory_context),
            json.dumps(react_trace),
            price_at_scan,
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"  [memory_agent] write_decision_snapshot failed for {ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = "GS"
    print(f"Testing Memory Agent on {ticker}...\n")

    result = run(
        ticker=ticker,
        signal_direction="LONG",
        regime="NEUTRAL",
        factors_used=["momentum_20d", "reversal_5d", "volume_spike"],
    )

    print("\n── Memory Agent Result ──")
    for k, v in result.items():
        if k == "react_trace":
            print(f"  react_trace: {len(v)} turns")
        elif k == "knowledge_rules":
            print(f"  knowledge_rules ({len(v)}):")
            for r in v:
                print(f"    - {r}")
        else:
            print(f"  {k}: {v}")
