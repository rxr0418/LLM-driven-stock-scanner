"""
agents/search_agent.py - Search Agent for Swing Trade Phase 2.

Responsibilities:
  - Fetch Yahoo Finance headlines for a ticker
  - Use Claude native tool_use (not hand-parsed ReAct) to decide
    whether to call Tavily for deeper research
  - Return a structured catalyst summary for the Decision Agent

Tool Registry:
  web_search — calls Tavily API, max MAX_TAVILY_SEARCHES times
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import anthropic

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SEARCH_AGENT_MODEL, SEARCH_AGENT_MAX_TOKENS, MAX_TAVILY_SEARCHES
from logger import get_logger
from resilience import with_retry

log = get_logger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Tool Registry — registered with Claude via tools=[]
# ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for recent financial news about a stock. "
            "Use when Yahoo headlines are insufficient to identify the catalyst. "
            f"You may call this at most {MAX_TAVILY_SEARCHES} times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'AAPL earnings Q2 2025 beat miss'",
                }
            },
            "required": ["query"],
        },
    }
]


SYSTEM_PROMPT = """You are a financial research agent analyzing stocks for swing trading.

Your job: given Yahoo Finance headlines and the option to search for more, identify the
most important catalyst driving this stock's price move and assess its strength.

Rules:
- Scan ALL provided headlines before deciding to search
- Use web_search only when headlines are insufficient to identify the catalyst type
- You have enough information when you know:
  * The catalyst type (earnings / contract / FDA / macro / none)
  * Whether the move is proportional to the catalyst
  * At least 2 independent sources, OR confirmed there is no news

CRITICAL — before classifying catalyst_type:
1. List ALL headlines you have seen
2. Identify which single event has the LARGEST price impact
3. Classify catalyst_type based on THAT event

When you have enough information, output ONLY this JSON (no markdown, no other text):
{
  "catalyst_type": "EARNINGS_BEAT|EARNINGS_MISS|CONTRACT_WIN|DEAL_WIN|FDA_APPROVAL|FDA_FAST_TRACK|MACRO|ANALYST_UPGRADE|ANALYST_DOWNGRADE|NO_CATALYST|OTHER",
  "catalyst_strength": "STRONG|MODERATE|WEAK|NONE",
  "news_alignment": "SUPPORTS|CONTRADICTS|NEUTRAL",
  "sources": ["headline 1", "headline 2"],
  "summary": "one sentence describing the key catalyst or lack thereof",
  "risk_flag": "none or specific risk",
  "search_count": integer
}"""


# ─────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────

def _execute_web_search(query: str) -> str:
    try:
        import requests
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "Tavily API key not set — web search unavailable."
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,
            },
            timeout=10,
        )
        data = resp.json()
        lines = []
        if data.get("answer"):
            lines.append(f"Summary: {data['answer']}")
        for r in data.get("results", [])[:3]:
            lines.append(
                f"  [{r.get('source', 'web')}] {r.get('title', '')} "
                f"— {r.get('content', '')[:120]}"
            )
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "web_search":
        return _execute_web_search(tool_input["query"])
    return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────

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


def _fallback_result(ticker: str, search_count: int) -> dict:
    return {
        "catalyst_type":     "OTHER",
        "catalyst_strength": "WEAK",
        "news_alignment":    "NEUTRAL",
        "sources":           [],
        "summary":           "Search agent parse error — analysis unavailable.",
        "risk_flag":         "parse_error",
        "search_count":      search_count,
        "ticker":            ticker,
        "react_trace":       [],
    }


# ─────────────────────────────────────────────────────────────
# Main agent loop (native tool_use)
# ─────────────────────────────────────────────────────────────

def _format_headlines(articles: list) -> str:
    if not articles:
        return "No headlines found from Yahoo Finance."
    return "\n".join(
        f"  [{a.get('publisher', 'Unknown')}] {a.get('title', '')}"
        for a in articles[:5]
    )


def run(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    yahoo_articles: list,
    recheck_questions: list | None = None,
) -> dict:
    """
    Run the Search Agent using Claude native tool_use.

    Args:
        ticker             : stock symbol
        signal_direction   : "LONG" or "SHORT"
        factor_score       : composite score from scanner (0-1)
        regime             : current market regime
        yahoo_articles     : list of {title, publisher} dicts from Yahoo
        recheck_questions  : if provided, this is a targeted recheck — focus
                             web_search on answering these specific questions

    Returns:
        Structured catalyst summary dict for merge() and Decision Agent.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    headlines_text = _format_headlines(yahoo_articles)

    if recheck_questions:
        recheck_block = (
            "\n\nRECHECK MODE — answer these specific questions before outputting JSON:\n"
            + "\n".join(f"  - {q}" for q in recheck_questions)
            + "\nUse web_search to find the answers."
        )
    else:
        recheck_block = ""

    messages = [{
        "role": "user",
        "content": (
            f"Analyze this swing trade candidate:\n\n"
            f"Ticker: {ticker} | Direction: {signal_direction} | "
            f"Score: {factor_score:.3f} | Regime: {regime}\n\n"
            f"Yahoo Finance headlines (assess ALL before deciding):\n"
            f"{headlines_text}\n\n"
            f"Identify the highest-impact catalyst, use web_search if needed, "
            f"then output the final JSON."
            f"{recheck_block}"
        ),
    }]

    search_count = 0
    react_trace  = []

    log.info("starting tool_use loop", extra={"ticker": ticker, "recheck": bool(recheck_questions)})

    @with_retry(label="search_agent/anthropic")
    def _create_message(**kwargs):
        return client.messages.create(**kwargs)

    # Agent loop: Claude decides when to call tools and when to stop
    for turn in range(MAX_TAVILY_SEARCHES + 3):
        response = _create_message(
            model=SEARCH_AGENT_MODEL,
            max_tokens=SEARCH_AGENT_MAX_TOKENS,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Claude finished — extract final JSON from text block
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
                    break

            react_trace.append({"turn": turn, "stop": "end_turn", "output": final_text[:200]})
            result = _extract_json(final_text)

            if result is None:
                log.warning("JSON parse failed, using fallback", extra={"ticker": ticker})
                return _fallback_result(ticker, search_count)

            result["search_count"] = search_count
            result["ticker"]       = ticker
            result["react_trace"]  = react_trace
            log.info("done", extra={"ticker": ticker, "searches": search_count,
                                    "catalyst": result.get("catalyst_type")})
            return result

        # Claude wants to call a tool
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                if search_count >= MAX_TAVILY_SEARCHES:
                    tool_output = f"Search limit ({MAX_TAVILY_SEARCHES}) reached. Use only what you have."
                    log.debug("search limit reached", extra={"ticker": ticker})
                else:
                    log.info("calling tool", extra={"ticker": ticker, "tool": tool_name,
                                                    "query": tool_input.get("query", "")})
                    tool_output  = _execute_tool(tool_name, tool_input)
                    search_count += 1

                react_trace.append({
                    "turn":   turn,
                    "tool":   tool_name,
                    "input":  tool_input,
                    "result": tool_output[:100],
                })
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     tool_output,
                })

            messages.append({"role": "user", "content": tool_results})

    log.warning("max turns exceeded, using fallback", extra={"ticker": ticker})
    return _fallback_result(ticker, search_count)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from swing.data import fetch_news

    ticker   = "GS"
    articles = fetch_news(ticker, max_articles=5)
    print(f"Yahoo headlines: {len(articles)}\n")

    result = run(
        ticker=ticker,
        signal_direction="LONG",
        factor_score=0.89,
        regime="NEUTRAL",
        yahoo_articles=articles,
    )

    print("\n── Search Agent Result ──")
    for k, v in result.items():
        if k == "react_trace":
            print(f"  react_trace: {len(v)} turns")
        else:
            print(f"  {k}: {v}")
