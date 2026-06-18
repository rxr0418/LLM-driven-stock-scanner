"""
llm_analyst.py - LLM-powered news analysis module.

Prompt engineering techniques used:
  - Role prompting       : Claude acts as a quantitative analyst
  - Structured context   : factor score, regime, news organized clearly
  - Task decomposition   : three explicit sub-questions to answer
  - Few-shot examples    : two examples guide output quality and format
  - Confidence calibration: explicit scoring rubric prevents clustering
  - Forced JSON output   : strict format constraint for reliable parsing
  - Negative constraints : "no other text" prevents markdown wrapping
"""

import json
import os
import warnings

import anthropic

warnings.filterwarnings("ignore")

ANALYST_MODEL      = "claude-sonnet-4-6"
ANALYST_MAX_TOKENS = 400


# ─────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────

def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Run: export ANTHROPIC_API_KEY='your-key'"
        )
    return anthropic.Anthropic(api_key=api_key)


# ─────────────────────────────────────────────────────────────
# Few-shot examples (injected into every prompt)
# ─────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLES OF GOOD ANALYSIS:

Example 1 — News strongly supports a LONG signal:
  Stock: AAPL | Signal: LONG | Score: 0.88 | Regime: TRENDING
  News:
    - [Bloomberg] Apple reports record iPhone sales, raises full-year guidance
    - [Reuters] Apple Services revenue hits all-time high, beats estimates by 12%
  Output:
  {
    "signal": "STRONG_BUY",
    "confidence": 88,
    "news_alignment": "SUPPORTS",
    "reason": "Record sales and raised guidance directly confirm momentum signal.",
    "risk_flag": "none"
  }

Example 2 — News contradicts a LONG signal (override):
  Stock: META | Signal: LONG | Score: 0.79 | Regime: NEUTRAL
  News:
    - [WSJ] Meta faces FTC antitrust lawsuit, breakup of Instagram possible
    - [FT] Advertisers pause Meta spending amid regulatory uncertainty
  Output:
  {
    "signal": "AVOID",
    "confidence": 78,
    "news_alignment": "CONTRADICTS",
    "reason": "Major antitrust risk overrides momentum; advertiser pullback threatens revenue.",
    "risk_flag": "FTC antitrust lawsuit — potential Instagram breakup"
  }

Example 3 — Earnings risk near a SHORT signal:
  Stock: NVDA | Signal: SHORT | Score: 0.12 | Regime: NEUTRAL
  News:
    - [CNBC] Nvidia earnings report due Wednesday after close
    - [Barron's] Analysts raise NVDA price targets ahead of results
  Output:
  {
    "signal": "NEUTRAL",
    "confidence": 45,
    "news_alignment": "CONTRADICTS",
    "reason": "Imminent earnings with bullish analyst sentiment contradicts short signal; wait for report.",
    "risk_flag": "Earnings Wednesday after close — high binary risk, avoid directional bets"
  }
"""

# ─────────────────────────────────────────────────────────────
# Confidence calibration rubric (injected into every prompt)
# ─────────────────────────────────────────────────────────────

CONFIDENCE_RUBRIC = """
CONFIDENCE SCORING GUIDE (use the full 0-100 range):
  90-100 : News directly confirms signal with hard data (earnings beat, guidance raise, major contract)
  75-89  : News generally supports signal with clear positive/negative catalysts
  55-74  : Mixed news or signal supported by factor alone with neutral news
  35-54  : News somewhat contradicts signal or major uncertainty present
  15-34  : News clearly contradicts signal or major risk event imminent
  0-14   : News strongly contradicts signal (e.g. fraud allegation, CEO departure, regulatory shutdown)
"""


# ─────────────────────────────────────────────────────────────
# Single stock analysis
# ─────────────────────────────────────────────────────────────

def analyze_stock(
    client: anthropic.Anthropic,
    ticker: str,
    factor_score: float,
    signal_direction: str,
    regime: str,
    news_articles: list,
    lang: str = "en",
) -> dict:
    """
    Ask Claude to analyze whether recent news supports the factor signal.

    Args:
        ticker           : stock symbol e.g. "AAPL"
        factor_score     : composite score from scanner (0 to 1)
        signal_direction : "LONG" or "SHORT"
        regime           : current market regime label
        news_articles    : list of dicts with keys: title, publisher

    Returns:
        dict with signal, confidence, reason, news_alignment, risk_flag
    """
    # Format news headlines for the prompt
    if news_articles:
        news_text = "\n".join(
            f"    - [{a.get('publisher', 'Unknown')}] {a.get('title', '')}"
            for a in news_articles[:5]
        )
    else:
        news_text = "    - No recent news found."

    # Determine factor type based on regime
    if regime == "TRENDING":
        factor_type = "momentum (winners keep winning)"
    elif regime == "VOLATILE":
        factor_type = "mean-reversion (oversold stocks bounce back)"
    else:
        factor_type = "combined momentum + reversal + volume"

    lang_instruction = (
    "请用中文回答，保持专业量化分析风格，reason控制在20个字以内。"
    if lang == "zh"
    else "Answer in English. Keep reason under 20 words."
)

    prompt = f"""{lang_instruction}You are a quantitative analyst reviewing a stock scanner signal. \
Your job is to determine whether recent news supports or contradicts the factor-based signal, \
and produce a final trading recommendation.

{FEW_SHOT_EXAMPLES}

{CONFIDENCE_RUBRIC}

─────────────────────────────────────────
NOW ANALYZE THIS STOCK:
─────────────────────────────────────────
Stock          : {ticker}
Market regime  : {regime}
Signal direction: {signal_direction}
Factor score   : {factor_score:.2f} out of 1.0
Factor type    : {factor_type}

Score interpretation:
  - LONG  signal: score near 1.0 = strongest bullish factor signal in the universe
  - SHORT signal: score near 0.0 = weakest / most bearish factor signal in the universe

Recent news headlines:
{news_text}

─────────────────────────────────────────
YOUR TASK:
1. Does the news SUPPORT, CONTRADICT, or is NEUTRAL to the {signal_direction} signal?
2. Are there major risk events (earnings, lawsuits, regulatory, macro) that could override the signal?
3. Produce a final recommendation, using the examples and confidence rubric above.

Return ONLY valid JSON, no markdown, no explanation outside the JSON:
{{
  "signal": "STRONG_BUY" or "BUY" or "NEUTRAL" or "AVOID",
  "confidence": integer 0-100 (use the full range per the rubric above),
  "news_alignment": "SUPPORTS" or "CONTRADICTS" or "NEUTRAL",
  "reason": "one sentence, max 20 words, specific to this stock",
  "risk_flag": "none" or specific risk description
}}"""

    try:
        response = client.messages.create(
            model=ANALYST_MODEL,
            max_tokens=ANALYST_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        return {
            "ticker":         ticker,
            "factor_score":   round(factor_score, 4),
            "signal":         result.get("signal", "NEUTRAL"),
            "confidence":     result.get("confidence", 50),
            "news_alignment": result.get("news_alignment", "NEUTRAL"),
            "reason":         result.get("reason", "No analysis available."),
            "risk_flag":      result.get("risk_flag", "none"),
            "news_titles":    [a.get("title", "") for a in news_articles[:5]],
        }

    except json.JSONDecodeError as e:
        print(f"  [llm] JSON parse error for {ticker}: {e}")
        return _fallback_result(ticker, factor_score, signal_direction)
    except Exception as e:
        print(f"  [llm] Analysis failed for {ticker}: {e}")
        return _fallback_result(ticker, factor_score, signal_direction)


def _fallback_result(ticker: str, factor_score: float, direction: str) -> dict:
    """Return a neutral placeholder when LLM analysis fails."""
    return {
        "ticker":         ticker,
        "factor_score":   round(factor_score, 4),
        "signal":         "NEUTRAL",
        "confidence":     0,
        "news_alignment": "NEUTRAL",
        "reason":         "LLM analysis unavailable — factor signal only.",
        "risk_flag":      "analysis_failed",
        "news_titles":    [],
    }


# ─────────────────────────────────────────────────────────────
# Batch analysis
# ─────────────────────────────────────────────────────────────

def analyze_watchlist(
    scan_results: dict,
    news_data: dict,
    top_n: int = 10,
    lang: str = "en",
) -> dict:
    """
    Run LLM analysis on the top long and short candidates.

    Args:
        scan_results : output of scanner.run_scan()
        news_data    : output of data.fetch_news_batch()
        top_n        : number of candidates to analyze per side

    Returns:
        dict with analyzed long_watchlist and short_watchlist
    """
    client  = get_client()
    regime  = scan_results.get("regime", "NEUTRAL")

    long_candidates  = scan_results.get("long_candidates", [])[:top_n]
    short_candidates = scan_results.get("short_candidates", [])[:top_n]

    print(f"\n[llm] Analyzing {len(long_candidates)} long candidates...")
    long_watchlist = []
    for item in long_candidates:
        ticker = item["ticker"]
        score  = item["score"]
        news   = news_data.get(ticker, [])
        print(f"  → {ticker} (score={score:.4f})", end=" ")
        result = analyze_stock(client, ticker, score, "LONG", regime, news, lang=lang)
        print(f"| {result['signal']} ({result['confidence']}%)")
        long_watchlist.append(result)

    print(f"\n[llm] Analyzing {len(short_candidates)} short candidates...")
    short_watchlist = []
    for item in short_candidates:
        ticker = item["ticker"]
        score  = item["score"]
        news   = news_data.get(ticker, [])
        print(f"  → {ticker} (score={score:.4f})", end=" ")
        result = analyze_stock(client, ticker, score, "SHORT", regime, news, lang=lang)
        print(f"| {result['signal']} ({result['confidence']}%)")
        short_watchlist.append(result)

    return {
        "regime":          regime,
        "description":     scan_results.get("description", ""),
        "factors_used":    scan_results.get("factors_used", []),
        "long_watchlist":  long_watchlist,
        "short_watchlist": short_watchlist,
        "timestamp":       scan_results.get("timestamp", ""),
    }


# ─────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────

def print_watchlist(watchlist: dict) -> None:
    """Print the final watchlist in a readable format."""

    signal_rank = {"STRONG_BUY": 0, "BUY": 1, "NEUTRAL": 2, "AVOID": 3}

    print("\n" + "=" * 70)
    print("DAILY WATCHLIST")
    print("=" * 70)
    print(f"Regime    : {watchlist['regime']}")
    print(f"Factors   : {watchlist['factors_used']}")
    print(f"Timestamp : {watchlist['timestamp']}")
    print(f"\n{watchlist['description']}")

    print("\n" + "─" * 70)
    print("LONG CANDIDATES")
    print("─" * 70)
    long_list = sorted(
        watchlist["long_watchlist"],
        key=lambda x: (signal_rank.get(x["signal"], 99), -x["confidence"])
    )
    for item in long_list:
        flag = "⚠ " if item["risk_flag"] != "none" else "  "
        print(
            f"{flag}{item['ticker']:<6} "
            f"| {item['signal']:<11} "
            f"| conf={item['confidence']:>3}% "
            f"| {item['reason']}"
        )
        if item["risk_flag"] != "none":
            print(f"         ↳ Risk: {item['risk_flag']}")

    print("\n" + "─" * 70)
    print("SHORT CANDIDATES")
    print("─" * 70)
    short_list = sorted(
        watchlist["short_watchlist"],
        key=lambda x: (signal_rank.get(x["signal"], 99), -x["confidence"])
    )
    for item in short_list:
        flag = "⚠ " if item["risk_flag"] != "none" else "  "
        print(
            f"{flag}{item['ticker']:<6} "
            f"| {item['signal']:<11} "
            f"| conf={item['confidence']:>3}% "
            f"| {item['reason']}"
        )
        if item["risk_flag"] != "none":
            print(f"         ↳ Risk: {item['risk_flag']}")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data    import fetch_price_data, fetch_news_batch, UNIVERSE
    from regime  import detect_regime
    from scanner import run_scan

    print("Testing llm_analyst.py...\n")

    test_tickers = UNIVERSE[:20]
    top_n        = 3  # keep small to save API cost during testing

    print("1. Fetching price data...")
    price_data = fetch_price_data(test_tickers, lookback_days=90)

    print("2. Detecting regime...")
    regime_result = detect_regime()

    print("3. Running scan...")
    scan_results = run_scan(price_data, regime_result, top_n=top_n)

    print("4. Fetching news...")
    all_candidates = (
        [item["ticker"] for item in scan_results["long_candidates"]] +
        [item["ticker"] for item in scan_results["short_candidates"]]
    )
    news_data = fetch_news_batch(all_candidates, max_articles=5)

    print("5. Running LLM analysis...")
    watchlist = analyze_watchlist(scan_results, news_data, top_n=top_n)

    print_watchlist(watchlist)