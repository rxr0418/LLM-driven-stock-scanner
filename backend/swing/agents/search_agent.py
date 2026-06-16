"""
agents/search_agent.py - Search Agent for Swing Trade Phase 2.

Responsibilities:
  - Fetch Yahoo Finance headlines for a ticker (free, no API key)
  - Run a ReAct loop: assess all headlines, decide whether to
    search for more via Tavily, stop when information is sufficient
  - Return a structured catalyst summary for the Decision Agent

ReAct loop:
  - Agent sees ALL Yahoo headlines upfront
  - Decides itself: sufficient? search? what query?
  - Max 2 Tavily searches to control cost
  - Always identifies highest-impact event before classifying
"""

import json
import os
import warnings
from typing import Optional

import anthropic

warnings.filterwarnings("ignore")

MAX_TAVILY_CALLS = 2

SYSTEM_PROMPT = """You are a financial research agent analyzing stocks for swing trading.

Your job is to gather sufficient information about a stock's recent catalyst and news.

You MUST follow this exact ReAct format for every step:

Thought: [your reasoning about what you know and what you still need]
Action: search(query) | stop
Observation: [result of action - filled in by the system]

Rules:
- Use Action: search(your query here) to search for more information
- Use Action: stop when you have enough information
- You have enough information when:
  * You know the catalyst type (earnings / contract / FDA / macro / none)
  * You have at least 2 independent sources OR confirmed there is no news
  * You understand if the price move is proportional to the catalyst
- Maximum 2 searches allowed - be efficient
- If no meaningful catalyst exists after searching, that IS useful information

CRITICAL — before classifying catalyst_type:
1. List ALL headlines you have seen (Yahoo + any Tavily results)
2. Identify which single event has the LARGEST price impact on this stock
3. Classify catalyst_type based on THAT event — not the first headline you saw

After Action: stop, output ONLY this JSON (no markdown):
{
  "catalyst_type": "EARNINGS_BEAT|EARNINGS_MISS|CONTRACT_WIN|DEAL_WIN|FDA_APPROVAL|FDA_FAST_TRACK|MACRO|ANALYST_UPGRADE|ANALYST_DOWNGRADE|NO_CATALYST|OTHER",
  "catalyst_strength": "STRONG|MODERATE|WEAK|NONE",
  "news_alignment": "SUPPORTS|CONTRADICTS|NEUTRAL",
  "sources": ["headline 1", "headline 2"],
  "summary": "one sentence describing the key catalyst or lack thereof",
  "risk_flag": "none or specific risk",
  "search_count": integer
}"""


def _format_headlines(articles: list) -> str:
    if not articles:
        return "No headlines found from Yahoo Finance."
    return "\n".join(
        f"  [{a.get('publisher', 'Unknown')}] {a.get('title', '')}"
        for a in articles[:5]
    )


def _call_tavily(query: str) -> str:
    try:
        import requests
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "Tavily API key not set — skipping web search."
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query,
                  "search_depth": "basic", "max_results": 3, "include_answer": True},
            timeout=10,
        )
        data = resp.json()
        lines = []
        if data.get("answer"):
            lines.append(f"Summary: {data['answer']}")
        for r in data.get("results", [])[:3]:
            lines.append(f"  [{r.get('source','web')}] {r.get('title','')} — {r.get('content','')[:120]}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def _parse_action(text: str) -> tuple[str, Optional[str]]:
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            action = line[len("Action:"):].strip()
            if action.lower().startswith("search(") and action.endswith(")"):
                query = action[len("search("):-1].strip().strip('"').strip("'")
                return "search", query
            elif action.lower() == "stop":
                return "stop", None
    return "stop", None


def _extract_json(text: str) -> dict:
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except Exception:
        pass
    return {
        "catalyst_type": "OTHER",
        "catalyst_strength": "WEAK",
        "news_alignment": "NEUTRAL",
        "sources": [],
        "summary": "Parse error — analysis unavailable.",
        "risk_flag": "parse_error",
        "search_count": 0,
    }


def run(
    ticker: str,
    signal_direction: str,
    factor_score: float,
    regime: str,
    yahoo_articles: list,
) -> dict:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    headlines_text = _format_headlines(yahoo_articles)

    initial_message = f"""Analyze this swing trade candidate:

Ticker: {ticker} | Direction: {signal_direction} | Score: {factor_score:.3f} | Regime: {regime}

Yahoo Finance headlines (all available — assess ALL before deciding):
{headlines_text}

Start your ReAct analysis. Scan all headlines, identify the highest-impact event, then decide if you need more information."""

    messages = [{"role": "user", "content": initial_message}]
    search_count = 0
    react_trace = []

    print(f"  [search_agent] {ticker}: starting ReAct loop")

    for turn in range(MAX_TAVILY_CALLS + 2):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})
        react_trace.append({"turn": turn, "output": assistant_text})

        action_type, query = _parse_action(assistant_text)

        if action_type == "stop":
            print(f"  [search_agent] {ticker}: stopped after {search_count} searches")
            break

        if action_type == "search":
            if search_count >= MAX_TAVILY_CALLS:
                # Force stop
                messages.append({"role": "user", "content": "Max searches reached. Action: stop"})
                react_trace.append({"turn": turn, "forced": "max_searches"})
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                assistant_text = response.content[0].text
                messages.append({"role": "assistant", "content": assistant_text})
                react_trace.append({"turn": turn + 1, "output": assistant_text})
                break

            print(f"  [search_agent] {ticker}: searching → {query}")
            search_result = _call_tavily(query)
            search_count += 1
            messages.append({"role": "user", "content": f"Observation: {search_result}"})
            react_trace.append({"turn": turn, "search": query, "result_preview": search_result[:100]})

    final_text = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
    result = _extract_json(final_text)
    result["search_count"] = search_count
    result["ticker"] = ticker
    result["react_trace"] = react_trace

    return result


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))
    from swing.data import fetch_news

    ticker = "GS"
    print(f"Testing Search Agent on {ticker}...\n")
    articles = fetch_news(ticker, max_articles=5)
    print(f"Yahoo headlines fetched: {len(articles)}\n")

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