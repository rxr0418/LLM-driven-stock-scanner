"""
premarket/catalyst.py - Agentic LLM catalyst analysis using MCP servers.

MCP servers used:
  - Tavily MCP  : real-time web search for breaking news
  - PostgreSQL MCP : direct SQL queries on Supabase

Claude autonomously decides which tools to call and how many times.
Final output: TRADE / WATCH / AVOID with entry + exit timing.
"""

import json
import os
import warnings
from datetime import datetime

import anthropic

warnings.filterwarnings("ignore")

# Langfuse tracing v4.x — uses direct trace/span API, no decorators needed
try:
    from langfuse import Langfuse
    _langfuse = Langfuse(
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host       = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )
    LANGFUSE_AVAILABLE = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and
        os.environ.get("LANGFUSE_SECRET_KEY")
    )
except Exception:
    LANGFUSE_AVAILABLE = False
    _langfuse = None

# Import RAG functions as fallback if MCP unavailable
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database import get_all_catalyst_stats, get_relevant_knowledge
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Catalyst definitions
# ─────────────────────────────────────────────────────────────

CATALYST_TYPES = {
    "FDA_APPROVAL":    "FDA drug approval (full NDA/BLA approval, not Fast Track)",
    "FDA_FAST_TRACK":  "FDA Fast Track / Breakthrough designation (NOT approval)",
    "CLINICAL_TRIAL":  "Positive clinical trial results",
    "EARNINGS_BEAT":   "Quarterly earnings beat estimates",
    "EARNINGS_MISS":   "Quarterly earnings miss — explains gap down",
    "MA":              "Merger, acquisition, or buyout announcement",
    "CONTRACT_WIN":    "Major contract or partnership win",
    "GUIDANCE_RAISE":  "Management raises forward guidance",
    "DILUTION":        "Share offering or dilution — explains gap down",
    "SHORT_SQUEEZE":   "Short squeeze dynamics detected",
    "ANALYST_UPGRADE": "Analyst rating upgrade or price target raise",
    "SECTOR_MOVE":     "Riding broader sector momentum (crypto, biotech, EV)",
    "UNKNOWN":         "No clear catalyst — high manipulation risk",
}

FEW_SHOT_EXAMPLES = """
EXAMPLES:

Example 1 — Real FDA approval (strong, proportional):
  Ticker: SAVA | Change: +28% | RVOL: 15.2x | Float: 8M shares
  {
    "catalyst_type": "FDA_APPROVAL",
    "catalyst_strength": "STRONG",
    "proportionality": "FAIR",
    "manipulation_risk": "LOW",
    "signal": "TRADE",
    "confidence": 88,
    "reason": "Full FDA approval for lead drug is a company-changing binary event.",
    "risk": "Profit-taking in first 5 minutes is common; wait for first green candle.",
    "entry_timing": "Wait for first 1-minute candle to close green after 9:30 open.",
    "exit_timing": "Exit by 10:30 AM or at +20% gain; stop loss at -8% from entry."
  }

Example 2 — FDA Fast Track (weaker than it looks):
  Ticker: OCGN | Change: +22% | RVOL: 6.8x | Float: 12M shares
  {
    "catalyst_type": "FDA_FAST_TRACK",
    "catalyst_strength": "MODERATE",
    "proportionality": "OVER",
    "manipulation_risk": "MEDIUM",
    "signal": "WATCH",
    "confidence": 45,
    "reason": "Fast Track is NOT approval — only speeds review. +22% likely overreaction.",
    "risk": "Gap fill to +8-12% likely at open.",
    "entry_timing": "Only enter if holds above +15% in first 5 minutes.",
    "exit_timing": "If entered, exit within 30 minutes. Do not hold past 10:00 AM."
  }

Example 3 — No catalyst (manipulation risk):
  Ticker: MDJH | Change: +35% | RVOL: 25x | Float: 2M shares
  {
    "catalyst_type": "UNKNOWN",
    "catalyst_strength": "NONE",
    "proportionality": "OVER",
    "manipulation_risk": "HIGH",
    "signal": "AVOID",
    "confidence": 15,
    "reason": "No catalyst with 2M float and 25x RVOL is a classic pump-and-dump.",
    "risk": "Will likely reverse 80%+ of gains within 30 minutes of open.",
    "entry_timing": "Do not enter.",
    "exit_timing": "N/A — do not trade."
  }
"""

CONFIDENCE_RUBRIC = """
CONFIDENCE SCORING:
  85-100: Hard binary catalyst confirmed (FDA approval, signed acquisition, earnings release)
  70-84 : Strong fundamental catalyst, clear and company-specific
  50-69 : Moderate catalyst, some uncertainty about follow-through
  30-49 : Weak catalyst or proportionality mismatch
  10-29 : No real catalyst, or move clearly disproportionate to news
   0-9  : Active manipulation signals
"""


# ─────────────────────────────────────────────────────────────
# MCP server config
# ─────────────────────────────────────────────────────────────

def get_mcp_servers() -> list:
    """
    Build MCP server list from environment variables.
    Both servers are optional — falls back gracefully if keys missing.
    """
    servers = []

    # Tavily MCP — real-time web search
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        servers.append({
            "type": "url",
            "url":  f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_key}",
            "name": "tavily",
        })
    else:
        print("[mcp] TAVILY_API_KEY not set — search unavailable")

    # Supabase MCP — direct SQL queries on your database
    supabase_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "")
    if supabase_token:
        servers.append({
            "type": "url",
            "url":  "https://mcp.supabase.com/mcp?project_ref=xoedhlcucobyuvlxtyfi",
            "name": "supabase",
            "headers": {
                "Authorization": f"Bearer {supabase_token}",
            },
        })
    else:
        print("[mcp] SUPABASE_ACCESS_TOKEN not set — database MCP unavailable")

    return servers


# ─────────────────────────────────────────────────────────────
# Agentic analysis with MCP
# ─────────────────────────────────────────────────────────────

def analyze_catalyst_agentic(
    ticker: str,
    premarket_change_pct: float,
    rvol: float,
    float_shares: float,
    market_cap: float,
    news_items: list,
    lang: str = "en",
    client: anthropic.Anthropic = None,
    max_tokens: int = 1500,
) -> dict:
    """
    Agentic catalyst analysis using Tavily MCP + PostgreSQL MCP.

    Claude autonomously:
      1. Reviews Finnhub news
      2. Uses Tavily MCP to search for more news if needed
      3. Uses PostgreSQL MCP to query catalyst_stats and knowledge tables
      4. Outputs final TRADE/WATCH/AVOID signal
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client  = anthropic.Anthropic(api_key=api_key)

    # Format Finnhub news
    if news_items:
        news_text = "\n".join(
            f"  [{n.get('source', '?')}] {n.get('headline', '')}\n"
            f"  {n.get('summary', '')[:150]}"
            for n in news_items[:5]
        )
    else:
        news_text = "  No news found from Finnhub in last 48 hours."

    float_m      = float_shares / 1e6 if float_shares else 0
    cap_m        = market_cap / 1e6 if market_cap else 0
    now_et       = datetime.now()
    open_et      = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    mins_to_open = max(0, int((open_et - now_et).total_seconds() / 60))

    lang_instruction = (
        "请用中文回答，reason/risk/entry_timing/exit_timing各控制在30字以内。"
        if lang == "zh"
        else "Answer in English. Keep reason, risk, entry_timing, exit_timing under 25 words each."
    )

    # RAG fallback context (used when MCP DB unavailable)
    rag_fallback = ""
    if DB_AVAILABLE:
        stats = get_all_catalyst_stats()
        rules = get_relevant_knowledge(
            catalyst_type="",
            keywords=["catalyst", "pump", "float", "fda", "earnings"]
        )
        if stats:
            rag_fallback += f"\n{stats}\n"
        if rules:
            rag_fallback += f"\n{rules}\n"

    mcp_servers  = get_mcp_servers()
    has_tavily   = any(s["name"] == "tavily"   for s in mcp_servers)
    has_supabase = any(s["name"] == "supabase" for s in mcp_servers)

    system_prompt = f"""You are an experienced small-cap day trader with access to tools.
Market opens in {mins_to_open} minutes. Analyze whether this premarket mover is worth trading.

Your workflow:
1. Review the Finnhub news provided
2. If news is unclear or missing, use tavily-search to find more information
3. Query the database to check historical catalyst performance:
   - Table: catalyst_stats (columns: catalyst, sample_size, avg_open_return, win_rate_pct)
   - Table: knowledge (columns: category, content, confidence)
4. Output your final JSON analysis

Available MCP tools:
{"- tavily-search: real-time web search" if has_tavily else "- tavily-search: NOT AVAILABLE"}
{"- PostgreSQL queries on Supabase (catalyst_stats, knowledge tables)" if has_supabase else "- Database: NOT AVAILABLE"}

CRITICAL DISTINCTIONS:
- FDA APPROVAL (NDA/BLA granted) → STRONG catalyst
- FDA Fast Track / Breakthrough → NOT approval, often overreacted
- No news + tiny float + extreme RVOL → pump and dump, AVOID

{FEW_SHOT_EXAMPLES}

{CONFIDENCE_RUBRIC}

{rag_fallback}

{lang_instruction}

After gathering enough information, return ONLY this JSON (no markdown):
{{
  "catalyst_type": "one of: {', '.join(CATALYST_TYPES.keys())}",
  "catalyst_strength": "STRONG or MODERATE or WEAK or NONE",
  "proportionality": "OVER or FAIR or UNDER",
  "manipulation_risk": "HIGH or MEDIUM or LOW",
  "signal": "TRADE or WATCH or AVOID",
  "confidence": integer 0-100,
  "reason": "one sentence",
  "risk": "one sentence",
  "entry_timing": "one sentence",
  "exit_timing": "one sentence"
}}"""

    user_message = f"""Analyze this premarket stock:

Ticker           : {ticker}
Premarket change : {premarket_change_pct:+.1f}%
RVOL             : {rvol:.1f}x
Float            : {float_m:.1f}M shares
Market cap       : ${cap_m:.0f}M
Minutes to open  : {mins_to_open}

Finnhub news (last 48h):
{news_text}

Use your tools to gather more information if needed, then provide your final JSON analysis."""

    try:
        kwargs = {
            "model":      "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "system":     system_prompt,
            "messages":   [{"role": "user", "content": user_message}],
        }
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers

        response = client.messages.create(**kwargs)

        # Extract JSON from response
        for block in response.content:
            if hasattr(block, "text"):
                raw = block.text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(raw)
                    final = {
                        "ticker":            ticker,
                        "catalyst_type":     result.get("catalyst_type",     "UNKNOWN"),
                        "catalyst_strength": result.get("catalyst_strength", "NONE"),
                        "proportionality":   result.get("proportionality",   "FAIR"),
                        "manipulation_risk": result.get("manipulation_risk", "MEDIUM"),
                        "signal":            result.get("signal",            "AVOID"),
                        "confidence":        result.get("confidence",        0),
                        "reason":            result.get("reason",            ""),
                        "risk":              result.get("risk",              ""),
                        "entry_timing":      result.get("entry_timing",      ""),
                        "exit_timing":       result.get("exit_timing",       ""),
                    }
                    # Log to Langfuse
                    if LANGFUSE_AVAILABLE and _langfuse:
                        try:
                            _langfuse.create_event(
                                name     = f"catalyst_{ticker}",
                                input    = {
                                    "ticker": ticker,
                                    "change": premarket_change_pct,
                                    "rvol":   rvol,
                                    "news":   [n.get("headline", "") for n in news_items[:3]],
                                },
                                output   = final,
                                metadata = {
                                    "signal":        final["signal"],
                                    "catalyst_type": final["catalyst_type"],
                                    "confidence":    final["confidence"],
                                    "mcp_used":      [s["name"] for s in mcp_servers],
                                },
                            )
                            _langfuse.flush()
                        except Exception as lf_err:
                            print(f"[langfuse] logging failed: {lf_err}")
                    return final
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"[catalyst] MCP call failed for {ticker}: {e}")

    # Fallback
    return {
        "ticker":            ticker,
        "catalyst_type":     "UNKNOWN",
        "catalyst_strength": "NONE",
        "proportionality":   "FAIR",
        "manipulation_risk": "MEDIUM",
        "signal":            "AVOID",
        "confidence":        0,
        "reason":            "Analysis unavailable",
        "risk":              "Unknown",
        "entry_timing":      "Do not enter without analysis",
        "exit_timing":       "N/A",
    }


# ─────────────────────────────────────────────────────────────
# Batch analysis
# ─────────────────────────────────────────────────────────────

def analyze_candidates_batch(
    candidates: list,
    lang: str = "en",
) -> list:
    """Run agentic MCP catalyst analysis for all premarket candidates."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)

    mcp_servers = get_mcp_servers()
    print(f"[catalyst] Analyzing {len(candidates)} candidates "
          f"(MCP servers: {[s['name'] for s in mcp_servers]})...")

    for candidate in candidates:
        ticker     = candidate["ticker"]
        change     = candidate.get("premarket_change_pct", 0)
        rvol       = candidate.get("rvol", 0)
        float_sh   = candidate.get("float", 0)
        market_cap = candidate.get("market_cap", 0)
        news       = candidate.get("news", [])

        print(f"  → {ticker} ({change:+.1f}%, RVOL={rvol:.1f}x)")

        result = analyze_catalyst_agentic(
            ticker               = ticker,
            premarket_change_pct = change,
            rvol                 = rvol,
            float_shares         = float_sh,
            market_cap           = market_cap,
            news_items           = news,
            lang                 = lang,
            client               = client,
        )
        candidate.update(result)

        print(f"     → {result['signal']} ({result['confidence']}%) "
              f"[{result['catalyst_type']}] risk={result['manipulation_risk']}")

    return candidates


# ─────────────────────────────────────────────────────────────
# Mode-controlled analysis (for eval)
# ─────────────────────────────────────────────────────────────

def analyze_catalyst_with_mode(
    ticker: str,
    premarket_change_pct: float,
    rvol: float,
    float_shares: float,
    market_cap: float,
    news_items: list,
    mode: str = "full",
    lang: str = "en",
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Mode-controlled version of analyze_catalyst_agentic for eval.

    mode:
      "baseline"  — no RAG, no MCP (plain prompt only)
      "rag_only"  — RAG injected, no MCP
      "full"      — RAG + Tavily MCP + Supabase MCP (production)
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client  = anthropic.Anthropic(api_key=api_key)

    # Format news
    if news_items:
        news_text = "\n".join(
            f"  [{n.get('source', '?')}] {n.get('headline', '')}\n"
            f"  {n.get('summary', '')[:150]}"
            for n in news_items[:5]
        )
    else:
        news_text = "  No news found from Finnhub in last 48 hours."

    float_m      = float_shares / 1e6 if float_shares else 0
    cap_m        = market_cap / 1e6 if market_cap else 0
    now_et       = datetime.now()
    open_et      = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    mins_to_open = max(0, int((open_et - now_et).total_seconds() / 60))

    # RAG context — only injected in rag_only and full modes
    rag_context = ""
    if mode in ("rag_only", "full") and DB_AVAILABLE:
        stats = get_all_catalyst_stats()
        rules = get_relevant_knowledge(
            catalyst_type="",
            keywords=["catalyst", "pump", "float", "fda", "earnings"]
        )
        if stats:
            rag_context += f"\n{stats}\n"
        if rules:
            rag_context += f"\n{rules}\n"

    # MCP servers — only in full mode
    mcp_servers = []
    if mode == "full":
        mcp_servers = get_mcp_servers()

    system_prompt = f"""You are an experienced small-cap day trader.
Market opens in {mins_to_open} minutes. Analyze this premarket mover.

CRITICAL DISTINCTIONS:
- FDA APPROVAL (NDA/BLA granted) → STRONG catalyst
- FDA Fast Track / Breakthrough → NOT approval, often overreacted
- No news + tiny float + extreme RVOL → pump and dump, AVOID

{FEW_SHOT_EXAMPLES}

{CONFIDENCE_RUBRIC}

{rag_context}

Return ONLY this JSON (no markdown):
{{
  "catalyst_type": "one of: {', '.join(CATALYST_TYPES.keys())}",
  "catalyst_strength": "STRONG or MODERATE or WEAK or NONE",
  "proportionality": "OVER or FAIR or UNDER",
  "manipulation_risk": "HIGH or MEDIUM or LOW",
  "signal": "TRADE or WATCH or AVOID",
  "confidence": integer 0-100,
  "reason": "one sentence",
  "risk": "one sentence",
  "entry_timing": "one sentence",
  "exit_timing": "one sentence"
}}"""

    user_message = f"""Analyze this stock:

Ticker           : {ticker}
Premarket change : {premarket_change_pct:+.1f}%
RVOL             : {rvol:.1f}x
Float            : {float_m:.1f}M shares
Market cap       : ${cap_m:.0f}M
Minutes to open  : {mins_to_open}

News:
{news_text}"""

    # full mode: use the full agentic loop which handles MCP tool calls properly
    if mode == "full":
        result = analyze_catalyst_agentic(
            ticker               = ticker,
            premarket_change_pct = premarket_change_pct,
            rvol                 = rvol,
            float_shares         = float_shares,
            market_cap           = market_cap,
            news_items           = news_items,
            lang                 = lang,
            client               = client,
        )
        result["eval_mode"] = mode
        return result

    # baseline and rag_only: single call, no MCP loop needed
    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 600,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_message}],
        )

        for block in response.content:
            if hasattr(block, "text"):
                raw = block.text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(raw)
                    final = {
                        "ticker":            ticker,
                        "catalyst_type":     result.get("catalyst_type",     "UNKNOWN"),
                        "catalyst_strength": result.get("catalyst_strength", "NONE"),
                        "proportionality":   result.get("proportionality",   "FAIR"),
                        "manipulation_risk": result.get("manipulation_risk", "MEDIUM"),
                        "signal":            result.get("signal",            "AVOID"),
                        "confidence":        result.get("confidence",        0),
                        "reason":            result.get("reason",            ""),
                        "risk":              result.get("risk",              ""),
                        "entry_timing":      result.get("entry_timing",      ""),
                        "exit_timing":       result.get("exit_timing",       ""),
                        "eval_mode":         mode,
                    }
                    if LANGFUSE_AVAILABLE and _langfuse:
                        try:
                            _langfuse.create_event(
                                name     = f"catalyst_{ticker}_{mode}",
                                input    = {
                                    "ticker": ticker,
                                    "change": premarket_change_pct,
                                    "rvol":   rvol,
                                    "news":   [n.get("headline", "") for n in news_items[:3]],
                                    "mode":   mode,
                                },
                                output   = final,
                                metadata = {
                                    "signal":        final["signal"],
                                    "catalyst_type": final["catalyst_type"],
                                    "confidence":    final["confidence"],
                                },
                            )
                            _langfuse.flush()
                        except Exception as lf_err:
                            print(f"[langfuse] logging failed: {lf_err}")
                    return final
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"[eval] Failed for {ticker} mode={mode}: {e}")

    return {
        "ticker":            ticker,
        "catalyst_type":     "UNKNOWN",
        "catalyst_strength": "NONE",
        "proportionality":   "FAIR",
        "manipulation_risk": "MEDIUM",
        "signal":            "AVOID",
        "confidence":        0,
        "reason":            "Analysis failed",
        "risk":              "Unknown",
        "entry_timing":      "Do not enter",
        "exit_timing":       "N/A",
        "eval_mode":         mode,
    }