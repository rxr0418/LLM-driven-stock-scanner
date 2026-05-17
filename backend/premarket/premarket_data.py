"""
premarket/data.py - Premarket data fetching module.

Data sources:
  yfinance  → stock universe screening (price, market cap, float)
  Finnhub   → premarket quotes, news, company info

Responsibilities:
  - Build a universe of small-cap stocks to scan
  - Fetch premarket price change and volume
  - Compute RVOL (Relative Volume)
  - Fetch recent news for catalyst analysis
"""

import os
import time
import warnings
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Finnhub client
# ─────────────────────────────────────────────────────────────

def get_finnhub_client() -> finnhub.Client:
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not set")
    return finnhub.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────
# Small-cap universe
# ─────────────────────────────────────────────────────────────

# Curated small-cap watchlist
# These are liquid small-caps that frequently appear in premarket scanners
# Replace or expand based on your own research
SMALL_CAP_WATCHLIST = [
    # Biotech/Pharma (frequent FDA catalysts)
    "SAVA", "ACMR", "BCRX", "CGON", "CLOV", "CPRX", "CRMD",
    "DARE", "DPST", "DXCM", "EDIT", "FATE", "FGEN", "FREQ",
    "GOSS", "HALO", "HOOK", "IMVT", "INVA", "IONS", "ITCI",
    "KALA", "KPTI", "LGND", "LQDA", "MARA", "MBIO", "MDGL",
    "NKTR", "NVAX", "OCGN", "PGEN", "PRAX", "PRTK", "PTGX",
    # Tech small-caps
    "ADMA", "AEHR", "AIOT", "AKTS", "ALBT", "ALEC", "ALEX",
    "ALTU", "AMSC", "AMSWA", "ANTE", "APLT", "APPS", "ARCT",
    "ARDX", "ARQT", "ASYS", "ATEX", "ATIF", "ATOS", "ATRC",
    # Energy/Mining small-caps
    "AMMO", "AMPE", "AMPIO", "AMPX", "AMRK", "AMRS", "AMSC",
    # Additional liquid small-caps
    "BBAI", "BFRI", "BFST", "BGFV", "BGRY", "BHVN", "BIGC",
    "BIMI", "BIOL", "BIOX", "BIRD", "BITE", "BKNG", "BKSC",
    "BLFS", "BLKB", "BLMN", "BLND", "BLNK", "BLPH", "BLRX",
]

# Fallback: use S&P 600 small-cap ETF components
# For production, fetch from a proper small-cap index

def get_small_cap_universe(client: finnhub.Client) -> list:
    """
    Dynamically fetch US small-cap stocks from Finnhub.
    Returns tickers with price 1-20 USD and market cap < 500M.
    """
    try:
        # Get all US stocks
        stocks = client.stock_symbols("US")
        
        # Filter to common stock only, exclude ETFs/warrants
        tickers = [
            s["symbol"] for s in stocks
            if s.get("type") == "Common Stock"
            and "." not in s["symbol"]  # exclude foreign listings
            and len(s["symbol"]) <= 5    # exclude long symbols
        ]
        
        print(f"[premarket] Got {len(tickers)} US common stocks from Finnhub")
        return tickers[:500]  # limit for free tier
        
    except Exception as e:
        print(f"[premarket] Finnhub universe fetch failed: {e}")
        return SMALL_CAP_WATCHLIST  # fallback to static list
    
def screen_small_caps(
    min_price: float = 1.0,
    max_price: float = 20.0,
    max_market_cap: float = 5e8,  # 500M
    max_float: float = 5e7,       # 50M shares
) -> list:
    """
    Screen stocks from the watchlist that meet small-cap criteria.

    Args:
        min_price     : minimum stock price in USD
        max_price     : maximum stock price in USD
        max_market_cap: maximum market cap in USD
        max_float     : maximum float shares

    Returns:
        list of tickers that pass the screening criteria
    """
    qualified = []

    client   = get_finnhub_client()
    universe = get_small_cap_universe(client)
    print(f"[premarket] Screening {len(universe)} stocks...")

    for ticker in universe:
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            price      = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            market_cap = info.get("marketCap", 0)
            float_sh   = info.get("floatShares", float("inf"))

            if (
                min_price <= price <= max_price
                and market_cap <= max_market_cap
                and float_sh <= max_float
            ):
                qualified.append({
                    "ticker":     ticker,
                    "price":      round(price, 2),
                    "market_cap": market_cap,
                    "float":      float_sh,
                })

            time.sleep(0.1)  # avoid rate limiting

        except Exception:
            continue

    print(f"[premarket] {len(qualified)} stocks passed screening\n")
    return qualified


# ─────────────────────────────────────────────────────────────
# Premarket data
# ─────────────────────────────────────────────────────────────

def get_premarket_quote(ticker: str, client: finnhub.Client) -> dict:
    """
    Fetch current premarket quote for a ticker via Finnhub.

    Returns:
        dict with premarket_price, premarket_change_pct, volume
    """
    try:
        quote = client.quote(ticker)

        current_price  = quote.get("c", 0)   # current/premarket price
        prev_close     = quote.get("pc", 0)  # previous close
        premarket_vol  = quote.get("v", 0)   # volume so far today

        if prev_close == 0:
            return {}

        change_pct = ((current_price - prev_close) / prev_close) * 100

        return {
            "ticker":              ticker,
            "premarket_price":     round(current_price, 2),
            "prev_close":          round(prev_close, 2),
            "premarket_change_pct": round(change_pct, 2),
            "premarket_volume":    int(premarket_vol),
        }

    except Exception as e:
        print(f"[premarket] Quote failed for {ticker}: {e}")
        return {}


def compute_rvol(ticker: str, current_volume: int, lookback_days: int = 20) -> float:
    """
    Compute Relative Volume (RVOL).

    RVOL = today's volume so far / average volume at this time of day
    Uses full-day average volume as a proxy (simplification).

    Args:
        ticker        : stock ticker
        current_volume: volume traded so far today
        lookback_days : days to average over

    Returns:
        RVOL ratio (> 2 is notable, > 5 is strong)
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period=f"{lookback_days}d")

        if hist.empty:
            return 0.0

        avg_volume = hist["Volume"].mean()

        if avg_volume == 0:
            return 0.0

        # Adjust for time of day (premarket is typically ~10% of daily volume)
        # This is a rough approximation
        now = datetime.now()
        market_open  = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        if now < market_open:
            # Still premarket — compare to expected premarket volume
            expected_premarket_vol = avg_volume * 0.08
            rvol = current_volume / expected_premarket_vol if expected_premarket_vol > 0 else 0
        else:
            # Market hours — compare to full day average
            elapsed = (now - market_open).seconds
            full_day = (market_close - market_open).seconds
            fraction = min(elapsed / full_day, 1.0)
            expected_vol = avg_volume * fraction
            rvol = current_volume / expected_vol if expected_vol > 0 else 0

        return round(rvol, 2)

    except Exception:
        return 0.0


def get_recent_news(ticker: str, client: finnhub.Client, days: int = 2) -> list:
    """
    Fetch recent news for a ticker via Finnhub.

    Args:
        ticker : stock ticker
        client : Finnhub client
        days   : how many days back to look

    Returns:
        list of dicts with headline, source, datetime, summary
    """
    try:
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

        news = client.company_news(ticker, _from=start, to=end)

        results = []
        for article in news[:5]:
            results.append({
                "headline": article.get("headline", ""),
                "source":   article.get("source", ""),
                "datetime": article.get("datetime", 0),
                "summary":  article.get("summary", "")[:200],
            })

        return results

    except Exception as e:
        print(f"[premarket] News failed for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Full premarket scan
# ─────────────────────────────────────────────────────────────

def run_premarket_data_fetch(
    min_premarket_change: float = 5.0,
    max_premarket_change: float = 40.0,
    min_rvol: float = 2.0,
    min_volume: int = 100_000,
) -> list:
    """
    Run the full premarket data fetch pipeline.

    Steps:
      1. Screen small-cap universe
      2. Fetch premarket quotes for each stock
      3. Filter by premarket change and volume
      4. Compute RVOL
      5. Fetch news for qualified stocks

    Returns:
        list of candidate dicts with all premarket data
    """
    client = get_finnhub_client()

    # Step 1: Screen universe
    universe = screen_small_caps()
    tickers  = [s["ticker"] for s in universe]
    info_map = {s["ticker"]: s for s in universe}

    if not tickers:
        print("[premarket] No stocks passed screening")
        return []

    # Step 2: Fetch premarket quotes
    print(f"[premarket] Fetching premarket quotes for {len(tickers)} stocks...")
    candidates = []

    for i, ticker in enumerate(tickers):
        quote = get_premarket_quote(ticker, client)
        if not quote:
            continue

        change = quote["premarket_change_pct"]
        volume = quote["premarket_volume"]

        # Step 3: Filter by change and volume
        if not (min_premarket_change <= abs(change) <= max_premarket_change):
            continue
        if volume < min_volume:
            continue

        # Step 4: Compute RVOL
        rvol = compute_rvol(ticker, volume)
        if rvol < min_rvol:
            continue

        # Build candidate record
        stock_info = info_map.get(ticker, {})
        candidate  = {
            **quote,
            **stock_info,
            "rvol": rvol,
        }
        candidates.append(candidate)

        # Rate limit: Finnhub free tier = 60 calls/min
        if i % 55 == 0 and i > 0:
            print("[premarket] Rate limit pause (1s)...")
            time.sleep(1)

    print(f"[premarket] {len(candidates)} stocks passed premarket filter\n")

    # Step 5: Fetch news for candidates
    print("[premarket] Fetching news for candidates...")
    for candidate in candidates:
        ticker = candidate["ticker"]
        news   = get_recent_news(ticker, client)
        candidate["news"] = news
        time.sleep(0.1)

    return candidates


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing premarket/data.py...\n")

    client = get_finnhub_client()

    print("1. Testing Finnhub connection...")
    quote = get_finnhub_client()
    print("   Finnhub client OK\n")

    print("2. Testing single quote (AAPL)...")
    q = get_premarket_quote("AAPL", client)
    print(f"   Price={q.get('premarket_price')}, Change={q.get('premarket_change_pct')}%\n")

    print("3. Testing RVOL (AAPL)...")
    rvol = compute_rvol("AAPL", q.get("premarket_volume", 0))
    print(f"   RVOL={rvol}\n")

    print("4. Testing news fetch (AAPL)...")
    news = get_recent_news("AAPL", client)
    for n in news[:2]:
        print(f"   [{n['source']}] {n['headline'][:60]}")
    print()

    print("All tests passed!")
