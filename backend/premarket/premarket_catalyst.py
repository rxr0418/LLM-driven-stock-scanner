"""
premarket/catalyst.py - LLM catalyst quality analysis.

Analyzes news headlines to determine:
  - Catalyst type (FDA, earnings, M&A, contract, etc.)
  - Catalyst strength (STRONG / MODERATE / WEAK / NONE)
  - Whether the move is sustainable or likely to fade
  - Final premarket signal recommendation
"""

import json
import os
import warnings

import anthropic

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Catalyst type definitions
# ─────────────────────────────────────────────────────────────

CATALYST_TYPES = {
    "FDA_APPROVAL":     "FDA drug approval or breakthrough designation",
    "CLINICAL_TRIAL":   "Positive clinical trial results",
    "EARNINGS_BEAT":    "Quarterly earnings beat estimates",
    "MA":               "Merger, acquisition, or buyout announcement",
    "CONTRACT":         "Major contract or partnership win",
    "GUIDANCE_RAISE":   "Management raises forward guidance",
    "SHORT_SQUEEZE":    "Short squeeze dynamics detected",
    "ANALYST_UPGRADE":  "Analyst rating upgrade or price target raise",
    "SECTOR_MOMENTUM":  "Riding broader sector momentum",
    "UNKNOWN":          "No clear catalyst identified",
}

# Catalyst strength mapping
CATALYST_STRENGTH = {
    "FDA_APPROVAL":    "STRONG",
    "CLINICAL_TRIAL":  "STRONG",
    "EARNINGS_BEAT":   "STRONG",
    "MA":              "STRONG",
    "CONTRACT":        "MODERATE",
    "GUIDANCE_RAISE":  "MODERATE",
    "SHORT_SQUEEZE":   "MODERATE",
    "ANALYST_UPGRADE": "WEAK",
    "SECTOR_MOMENTUM": "WEAK",
    "UNKNOWN":         "NONE",
}

# Few-shot examples for catalyst analysis
FEW_SHOT_EXAMPLES = """
EXAMPLES:

Example 1 — Strong FDA catalyst:
  Ticker: SAVA | Premarket: +28% | RVOL: 15.2
  News: "Cassava Sciences receives FDA Breakthrough Therapy designation for simufilam"
  Output:
  {
    "catalyst_type": "FDA_APPROVAL",
    "catalyst_strength": "STRONG",
    "signal": "TRADE",
    "confidence": 88,
    "reason": "FDA Breakthrough designation is a binary catalyst with sustained follow-through historically.",
    "risk": "Gap fills are common after FDA news; watch for profit-taking at open."
  }

Example 2 — Earnings beat:
  Ticker: BBAI | Premarket: +18% | RVOL: 8.5
  News: "BigBear.ai reports Q4 revenue $45M, beats consensus of $38M; raises full-year guidance"
  Output:
  {
    "catalyst_type": "EARNINGS_BEAT",
    "catalyst_strength": "STRONG",
    "signal": "TRADE",
    "confidence": 78,
    "reason": "Revenue beat + guidance raise is a double catalyst; institutional buying likely on open.",
    "risk": "Already up 18% premarket; wait for first 5-minute candle to confirm direction."
  }

Example 3 — Weak analyst upgrade:
  Ticker: MARA | Premarket: +12% | RVOL: 3.1
  News: "Analyst at Roth Capital raises MARA price target from $12 to $16"
  Output:
  {
    "catalyst_type": "ANALYST_UPGRADE",
    "catalyst_strength": "WEAK",
    "signal": "WATCH",
    "confidence": 42,
    "reason": "Analyst upgrades rarely sustain premarket moves; often fade at open.",
    "risk": "High probability of gap fill. Only trade if RVOL stays above 3 after open."
  }

Example 4 — No catalyst:
  Ticker: OCGN | Premarket: +9% | RVOL: 2.3
  News: No relevant news found in the last 48 hours.
  Output:
  {
    "catalyst_type": "UNKNOWN",
    "catalyst_strength": "NONE",
    "signal": "AVOID",
    "confidence": 20,
    "reason": "No identifiable catalyst. Sympathy play or manipulation risk is high.",
    "risk": "Moves without catalysts frequently reverse hard at open. Avoid."
  }
"""

CONFIDENCE_RUBRIC = """
CONFIDENCE SCORING GUIDE:
  85-100: Binary catalyst with clear outcome (FDA approval, confirmed acquisition)
  70-84 : Strong fundamental catalyst (earnings beat + guidance, major contract)
  50-69 : Moderate catalyst with uncertainty (clinical results, guidance only)
  30-49 : Weak catalyst (analyst upgrade, sector momentum)
  10-29 : No clear catalyst or likely manipulation
"""


# ─────────────────────────────────────────────────────────────
# LLM analysis
# ─────────────────────────────────────────────────────────────

def analyze_catalyst(
    ticker: str,
    premarket_change_pct: float,
    rvol: float,
    news_items: list,
    lang: str = "en",
    client: anthropic.Anthropic = None,
) -> dict:
    """
    Use Claude to analyze the catalyst quality for a premarket mover.

    Args:
        ticker               : stock symbol
        premarket_change_pct : premarket price change in %
        rvol                 : relative volume ratio
        news_items           : list of news dicts with headline and summary
        lang                 : "en" or "zh"
        client               : Anthropic client (created if not provided)

    Returns:
        dict with catalyst_type, catalyst_strength, signal, confidence, reason, risk
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client  = anthropic.Anthropic(api_key=api_key)

    # Format news
    if news_items:
        news_text = "\n".join(
            f"  - [{n.get('source', 'Unknown')}] {n.get('headline', '')}"
            for n in news_items[:5]
        )
    else:
        news_text = "  - No relevant news found in the last 48 hours."

    lang_instruction = (
        "请用中文回答，保持专业交易分析风格，reason和risk控制在25个字以内。"
        if lang == "zh"
        else "Answer in English. Keep reason and risk under 20 words each."
    )

    prompt = f"""{lang_instruction}

You are an experienced day trader analyzing a premarket gap-up stock.
Your job is to identify the catalyst, assess its quality, and decide 
whether this is worth trading at the open.

{FEW_SHOT_EXAMPLES}

{CONFIDENCE_RUBRIC}

─────────────────────────────────────────
NOW ANALYZE THIS STOCK:
─────────────────────────────────────────
Ticker           : {ticker}
Premarket change : {premarket_change_pct:+.1f}%
RVOL             : {rvol:.1f}x

Recent news (last 48 hours):
{news_text}

─────────────────────────────────────────
Catalyst types to choose from:
{', '.join(CATALYST_TYPES.keys())}

Signal options:
  TRADE → Strong catalyst, worth taking a position at open
  WATCH → Moderate catalyst, wait for confirmation after open
  AVOID → Weak/no catalyst, high risk of reversal

Return ONLY valid JSON, no markdown, no extra text:
{{
  "catalyst_type": "one of the types above",
  "catalyst_strength": "STRONG" or "MODERATE" or "WEAK" or "NONE",
  "signal": "TRADE" or "WATCH" or "AVOID",
  "confidence": integer 0-100,
  "reason": "one sentence explanation",
  "risk": "one sentence about main risk"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        raw    = response.content[0].text.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return {
            "ticker":             ticker,
            "catalyst_type":      result.get("catalyst_type", "UNKNOWN"),
            "catalyst_strength":  result.get("catalyst_strength", "NONE"),
            "signal":             result.get("signal", "AVOID"),
            "confidence":         result.get("confidence", 0),
            "reason":             result.get("reason", ""),
            "risk":               result.get("risk", ""),
        }

    except Exception as e:
        print(f"[catalyst] LLM failed for {ticker}: {e}")
        return {
            "ticker":            ticker,
            "catalyst_type":     "UNKNOWN",
            "catalyst_strength": "NONE",
            "signal":            "AVOID",
            "confidence":        0,
            "reason":            "Analysis unavailable",
            "risk":              "Unknown",
        }


def analyze_candidates_batch(
    candidates: list,
    lang: str = "en",
) -> list:
    """
    Run catalyst analysis for all premarket candidates.

    Args:
        candidates : list of candidate dicts from data.py
        lang       : "en" or "zh"

    Returns:
        candidates with catalyst analysis added
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)

    print(f"[catalyst] Analyzing {len(candidates)} candidates...")

    for candidate in candidates:
        ticker  = candidate["ticker"]
        change  = candidate.get("premarket_change_pct", 0)
        rvol    = candidate.get("rvol", 0)
        news    = candidate.get("news", [])

        print(f"  → {ticker} ({change:+.1f}%, RVOL={rvol:.1f}x)", end=" ")

        result = analyze_catalyst(ticker, change, rvol, news, lang=lang, client=client)
        candidate.update(result)

        print(f"| {result['signal']} ({result['confidence']}%) [{result['catalyst_type']}]")

    return candidates
