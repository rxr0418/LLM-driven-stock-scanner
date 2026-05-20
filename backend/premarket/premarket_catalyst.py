"""
premarket/catalyst.py - LLM catalyst quality analysis for day trading.

Analyzes premarket movers to determine:
  - Catalyst type and authenticity
  - Whether the premarket move is proportional to the catalyst
  - Manipulation/pump-and-dump risk
  - Entry timing suggestion
  - Final signal: TRADE / WATCH / AVOID
"""

import json
import os
import warnings

import anthropic

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Catalyst definitions
# ─────────────────────────────────────────────────────────────

CATALYST_TYPES = {
    "FDA_APPROVAL":     "FDA drug approval (full NDA/BLA approval, not Fast Track)",
    "FDA_FAST_TRACK":   "FDA Fast Track / Breakthrough designation (NOT approval)",
    "CLINICAL_TRIAL":   "Positive clinical trial results",
    "EARNINGS_BEAT":    "Quarterly earnings beat estimates",
    "EARNINGS_MISS":    "Quarterly earnings miss — explains gap down",
    "MA":               "Merger, acquisition, or buyout announcement",
    "CONTRACT_WIN":     "Major contract or partnership win",
    "GUIDANCE_RAISE":   "Management raises forward guidance",
    "DILUTION":         "Share offering or dilution — explains gap down",
    "SHORT_SQUEEZE":    "Short squeeze dynamics detected",
    "ANALYST_UPGRADE":  "Analyst rating upgrade or price target raise",
    "SECTOR_MOVE":      "Riding broader sector momentum (crypto, biotech, EV)",
    "UNKNOWN":          "No clear catalyst — high manipulation risk",
}

# ─────────────────────────────────────────────────────────────
# Few-shot examples
# ─────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLES:

Example 1 — Real FDA approval (strong, proportional):
  Ticker: SAVA | Change: +28% | RVOL: 15.2x | Float: 8M shares
  Minutes to open: 45
  News: "Cassava Sciences receives FDA approval for simufilam in Alzheimer's disease"
  Analysis:
  {
    "catalyst_type": "FDA_APPROVAL",
    "catalyst_strength": "STRONG",
    "proportionality": "FAIR",
    "manipulation_risk": "LOW",
    "signal": "TRADE",
    "confidence": 88,
    "reason": "Full FDA approval for lead drug is a company-changing binary event with sustained follow-through.",
    "risk": "Profit-taking in first 5 minutes is common; wait for first green candle to confirm.",
    "entry_timing": "Wait for first 1-minute candle to close green after 9:30 open."
  }

Example 2 — FDA Fast Track (weaker than it looks):
  Ticker: OCGN | Change: +22% | RVOL: 6.8x | Float: 12M shares
  Minutes to open: 30
  News: "Ocugen receives FDA Fast Track designation for vaccine candidate"
  Analysis:
  {
    "catalyst_type": "FDA_FAST_TRACK",
    "catalyst_strength": "MODERATE",
    "proportionality": "OVER",
    "manipulation_risk": "MEDIUM",
    "signal": "WATCH",
    "confidence": 45,
    "reason": "Fast Track is NOT approval — only speeds review. +22% likely overreaction.",
    "risk": "Gap fill to +8-12% is likely at open as retail traders realize this is not an approval.",
    "entry_timing": "Only enter if holds above +15% in first 5 minutes. Otherwise avoid."
  }

Example 3 — Earnings beat with guidance raise:
  Ticker: BBAI | Change: +18% | RVOL: 8.5x | Float: 45M shares
  Minutes to open: 60
  News: "BigBear.ai Q4 revenue $45M beats $38M estimate; raises FY guidance to $195M"
  Analysis:
  {
    "catalyst_type": "EARNINGS_BEAT",
    "catalyst_strength": "STRONG",
    "proportionality": "FAIR",
    "manipulation_risk": "LOW",
    "signal": "TRADE",
    "confidence": 78,
    "reason": "Revenue beat + guidance raise is double catalyst; institutional buying expected at open.",
    "risk": "Float is 45M — larger than ideal for small-cap momentum; move may be slower.",
    "entry_timing": "Buy on first pullback after open, not at the open price itself."
  }

Example 4 — No catalyst (manipulation risk):
  Ticker: MDJH | Change: +35% | RVOL: 25x | Float: 2M shares
  Minutes to open: 20
  News: No relevant news found in last 48 hours.
  Analysis:
  {
    "catalyst_type": "UNKNOWN",
    "catalyst_strength": "NONE",
    "proportionality": "OVER",
    "manipulation_risk": "HIGH",
    "signal": "AVOID",
    "confidence": 15,
    "reason": "No identifiable catalyst with 2M float and 25x RVOL is a classic pump-and-dump pattern.",
    "risk": "Will likely reverse 80%+ of gains within 30 minutes of open. Do not chase.",
    "entry_timing": "Do not enter. Short only if you have Level 2 access and see clear exhaustion."
  }

Example 5 — Sector move (Bitcoin miners):
  Ticker: MARA | Change: +12% | RVOL: 3.2x | Float: 95M shares
  Minutes to open: 50
  News: "Bitcoin surges 5% overnight to $72,000 on ETF inflow data"
  Analysis:
  {
    "catalyst_type": "SECTOR_MOVE",
    "catalyst_strength": "MODERATE",
    "proportionality": "FAIR",
    "manipulation_risk": "LOW",
    "signal": "WATCH",
    "confidence": 52,
    "reason": "BTC move is real but MARA's 95M float means slower price action than pure small-caps.",
    "risk": "If BTC pulls back at open, MARA will drop faster. Sector moves can reverse quickly.",
    "entry_timing": "Only enter if BTC holds gains at 9:30. Watch BTC price as leading indicator."
  }
"""

# ─────────────────────────────────────────────────────────────
# Confidence rubric
# ─────────────────────────────────────────────────────────────

CONFIDENCE_RUBRIC = """
CONFIDENCE SCORING:
  85-100: Hard binary catalyst confirmed (FDA approval text, signed acquisition, earnings release)
  70-84 : Strong fundamental catalyst, clear and company-specific
  50-69 : Moderate catalyst, some uncertainty about follow-through
  30-49 : Weak catalyst (analyst upgrade, vague partnership) or proportionality mismatch
  10-29 : No real catalyst, or move is clearly disproportionate to news
   0-9  : Active manipulation signals (no news + extreme float + extreme RVOL)
"""


# ─────────────────────────────────────────────────────────────
# Core analysis function
# ─────────────────────────────────────────────────────────────

def analyze_catalyst(
    ticker: str,
    premarket_change_pct: float,
    rvol: float,
    float_shares: float,
    market_cap: float,
    news_items: list,
    lang: str = "en",
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Use Claude to analyze catalyst quality for a premarket mover.

    Key differences from swing trade analysis:
      - Focuses on TODAY's specific catalyst, not weekly news
      - Judges proportionality (is +30% justified by the news?)
      - Assesses manipulation risk explicitly
      - Gives specific entry timing advice
      - Distinguishes FDA approval vs Fast Track (critical difference)

    Args:
        ticker               : stock symbol
        premarket_change_pct : premarket price change %
        rvol                 : relative volume
        float_shares         : float shares (raw shares)
        market_cap           : market cap in USD
        news_items           : list of news dicts {headline, summary, source}
        lang                 : "en" or "zh"
        client               : Anthropic client

    Returns:
        dict with full catalyst analysis
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client  = anthropic.Anthropic(api_key=api_key)

    # Format news
    if news_items:
        news_text = "\n".join(
            f"  [{n.get('source', '?')}] {n.get('headline', '')}\n"
            f"  Summary: {n.get('summary', 'N/A')[:150]}"
            for n in news_items[:5]
        )
    else:
        news_text = "  No relevant news found in the last 48 hours."

    float_m    = float_shares / 1e6 if float_shares else 0
    cap_m      = market_cap / 1e6 if market_cap else 0

    # Compute minutes to market open (ET)
    from datetime import datetime
    now_et     = datetime.now()
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    mins_to_open = max(0, int((market_open - now_et).total_seconds() / 60))

    lang_instruction = (
        "请用中文回答，保持专业交易分析风格，reason/risk/entry_timing各控制在30字以内。"
        if lang == "zh"
        else "Answer in English. Keep reason, risk, and entry_timing under 25 words each."
    )

    prompt = f"""{lang_instruction}

You are an experienced small-cap day trader. Market opens in {mins_to_open} minutes.
Your job: determine if this premarket move is worth trading at the open.

CRITICAL DISTINCTIONS YOU MUST MAKE:
- FDA APPROVAL (NDA/BLA granted) → company-changing, STRONG catalyst
- FDA Fast Track / Breakthrough Designation → speeds review only, NOT approval, often OVERREACTED
- Earnings beat alone vs beat + guidance raise → very different strength
- No news + extreme move + tiny float → almost always pump and dump, AVOID

{FEW_SHOT_EXAMPLES}

{CONFIDENCE_RUBRIC}

─────────────────────────────────────────
ANALYZE THIS STOCK:
─────────────────────────────────────────
Ticker           : {ticker}
Premarket change : {premarket_change_pct:+.1f}%
RVOL             : {rvol:.1f}x
Float            : {float_m:.1f}M shares
Market cap       : ${cap_m:.0f}M
Minutes to open  : {mins_to_open}

News (last 48 hours):
{news_text}

─────────────────────────────────────────
Return ONLY valid JSON, no markdown, no extra text:
{{
  "catalyst_type": "one of: {', '.join(CATALYST_TYPES.keys())}",
  "catalyst_strength": "STRONG or MODERATE or WEAK or NONE",
  "proportionality": "OVER (move > catalyst) or FAIR or UNDER (move < catalyst)",
  "manipulation_risk": "HIGH or MEDIUM or LOW",
  "signal": "TRADE or WATCH or AVOID",
  "confidence": integer 0-100,
  "reason": "one sentence — what is the catalyst and why does it matter",
  "risk": "one sentence — main risk at open",
  "entry_timing": "one sentence — when exactly should a day trader enter"
}}"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}],
        )

        raw    = response.content[0].text.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return {
            "ticker":             ticker,
            "catalyst_type":      result.get("catalyst_type",     "UNKNOWN"),
            "catalyst_strength":  result.get("catalyst_strength", "NONE"),
            "proportionality":    result.get("proportionality",   "FAIR"),
            "manipulation_risk":  result.get("manipulation_risk", "MEDIUM"),
            "signal":             result.get("signal",            "AVOID"),
            "confidence":         result.get("confidence",        0),
            "reason":             result.get("reason",            ""),
            "risk":               result.get("risk",              ""),
            "entry_timing":       result.get("entry_timing",      ""),
        }

    except Exception as e:
        print(f"[catalyst] LLM failed for {ticker}: {e}")
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
        }


# ─────────────────────────────────────────────────────────────
# Batch analysis
# ─────────────────────────────────────────────────────────────

def analyze_candidates_batch(
    candidates: list,
    lang: str = "en",
) -> list:
    """Run catalyst analysis for all premarket candidates."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)

    print(f"[catalyst] Analyzing {len(candidates)} candidates...")

    for candidate in candidates:
        ticker     = candidate["ticker"]
        change     = candidate.get("premarket_change_pct", 0)
        rvol       = candidate.get("rvol", 0)
        float_sh   = candidate.get("float", 0)
        market_cap = candidate.get("market_cap", 0)
        news       = candidate.get("news", [])

        print(f"  → {ticker} ({change:+.1f}%, RVOL={rvol:.1f}x)", end=" ")

        result = analyze_catalyst(
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

        print(f"| {result['signal']} ({result['confidence']}%) "
              f"[{result['catalyst_type']}] risk={result['manipulation_risk']}")

    return candidates
