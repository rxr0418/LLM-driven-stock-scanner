"""
data.py - Data fetching module for the stock scanner.

Responsibilities:
  - Download OHLCV price data via yfinance
  - Fetch recent news headlines via Yahoo Finance (free, no API key needed)
  - Define the stock universe to scan
"""
import json
from pathlib import Path
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Stock universe
# ─────────────────────────────────────────────────────────────

# S&P 500 representative sample across sectors

def load_sp500() -> list:
    cache_file = Path(__file__).parent / "sp500_tickers.json"
    if cache_file.exists():
        with open(cache_file) as f:
            tickers = json.load(f)
        print(f"[swing] Loaded {len(tickers)} tickers from cache")
        return tickers

    # 下面这行只有缓存不存在时才跑，现在应该永远跑不到
    tickers = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    with open(cache_file, "w") as f:
        json.dump(tickers, f)
    return tickers
UNIVERSE = load_sp500()


# ─────────────────────────────────────────────────────────────
# Price data
# ─────────────────────────────────────────────────────────────

def fetch_price_data(tickers: list, lookback_days: int = 60) -> dict:
    """
    Download OHLCV data for a list of tickers.

    Args:
        tickers      : list of ticker symbols
        lookback_days: number of calendar days to look back

    Returns:
        dict with keys: close, open, high, low, volume (each a DataFrame)
        Returns empty dict if download fails.
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # Handle single-ticker edge case (yfinance returns flat columns)
        if isinstance(raw.columns, pd.Index) and not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

        return {
            "close":  raw["Close"].dropna(how="all"),
            "open":   raw["Open"].dropna(how="all"),
            "high":   raw["High"].dropna(how="all"),
            "low":    raw["Low"].dropna(how="all"),
            "volume": raw["Volume"].dropna(how="all"),
        }

    except Exception as e:
        print(f"[data] Price download failed: {e}")
        return {}


def fetch_single_ticker(ticker: str, lookback_days: int = 60) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        return df
    except Exception as e:
        print(f"[data] Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# News data
# ─────────────────────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 5) -> list:
    """
    Fetch recent news headlines for a ticker via Yahoo Finance.
    Free, no API key required.

    Args:
        ticker      : stock ticker symbol
        max_articles: max number of articles to return

    Returns:
        list of dicts with keys: title, publisher, link, published
        Returns empty list if fetch fails.
    """
    try:
        t    = yf.Ticker(ticker)
        news = t.news or []

        results = []
        for article in news[:max_articles]:
            content = article.get("content", {})
            results.append({
                "title":     content.get("title", "No title"),
                "publisher": content.get("provider", {}).get("displayName", "Unknown"),
                "link":      content.get("canonicalUrl", {}).get("url", ""),
                "published": content.get("pubDate", ""),
            })

        return results

    except Exception as e:
        print(f"[data] News fetch failed for {ticker}: {e}")
        return []


def fetch_news_batch(tickers: list, max_articles: int = 3) -> dict:
    """
    Fetch news for multiple tickers.

    Returns:
        dict mapping ticker -> list of articles
    """
    results = {}
    for ticker in tickers:
        results[ticker] = fetch_news(ticker, max_articles)
    return results


# ─────────────────────────────────────────────────────────────
# Market overview
# ─────────────────────────────────────────────────────────────

def fetch_market_overview() -> dict:
    """
    Fetch key market benchmarks for regime detection.

    Returns:
        dict with SPY, QQQ, VIX recent close prices and returns
    """
    benchmarks = ["SPY", "QQQ", "^VIX"]

    try:
        raw = yf.download(
            benchmarks,
            period="30d",
            auto_adjust=True,
            progress=False,
        )
        close = raw["Close"]

        overview = {}
        for ticker in benchmarks:
            col = ticker
            if col in close.columns:
                series = close[col].dropna()
                overview[ticker] = {
                    "latest":      round(float(series.iloc[-1]), 2),
                    "return_5d":   round(float(series.pct_change(5).iloc[-1]), 4),
                    "return_20d":  round(float(series.pct_change(20).iloc[-1]), 4),
                }

        return overview

    except Exception as e:
        print(f"[data] Market overview fetch failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data.py...\n")

    # Test price data
    print("1. Fetching price data for 5 tickers...")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
    data = fetch_price_data(test_tickers, lookback_days=30)
    if data:
        print(f"   Close shape: {data['close'].shape}")
        print(f"   Latest close:\n{data['close'].iloc[-1].round(2)}\n")
    else:
        print("   Failed\n")

    # Test news
    print("2. Fetching news for AAPL...")
    news = fetch_news("AAPL", max_articles=3)
    for article in news:
        print(f"   [{article['publisher']}] {article['title']}")
    print()

    # Test market overview
    print("3. Fetching market overview...")
    overview = fetch_market_overview()
    for ticker, metrics in overview.items():
        print(f"   {ticker}: latest={metrics['latest']}, 5d={metrics['return_5d']:.2%}")
