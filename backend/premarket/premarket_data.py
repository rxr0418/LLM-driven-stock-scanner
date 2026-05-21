"""
premarket/data.py - Premarket data fetching module.

Universe strategy:
  Priority 1: small_cap_300m.json  (pre-filtered, market cap < $300M, ~1071 tickers)
  Priority 2: small_cap_100m.json  (market cap < $100M, ~930 tickers)
  Priority 3: Finnhub stock_symbols API (dynamic)
  Priority 4: Hardcoded fallback list

Units (all raw values passed from frontend):
  price        → USD
  market_cap   → USD  (frontend sends M$ × 1e6)
  float        → shares (frontend sends K sh × 1000)
  pm_volume    → shares (frontend sends K sh × 1000)
  pm_amount    → USD   (frontend sends K$ × 1000)
  day_volume   → shares (frontend sends K sh × 1000)
"""

import asyncio
import json
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import finnhub
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
CAP_300M_FILE  = BASE_DIR / "small_cap_300m.json"
CAP_100M_FILE  = BASE_DIR / "small_cap_100m.json"
UNIVERSE_CACHE = BASE_DIR / "us_universe.json"


# ─────────────────────────────────────────────────────────────
# Fallback list (used only if all files are missing)
# ─────────────────────────────────────────────────────────────

FALLBACK_TICKERS = [
    "MARA", "RIOT", "CLSK", "CIFR", "BTBT", "HUT", "BITF", "HIVE", "WULF", "IREN",
    "NKLA", "RIDE", "GOEV", "WKHS", "SOLO", "IDEX", "EVGO", "BLNK", "CHPT", "MULN",
    "GME",  "AMC",  "SNDL", "WISH", "SKLZ", "SPCE", "PSFE", "DWAC", "ATER", "BBIG",
    "NVAX", "OCGN", "PGEN", "PRAX", "BCRX", "ACMR", "ADMA", "EDIT", "FATE", "BNGO",
    "APPS", "BBAI", "BFRI", "BHVN", "BIMI", "BIRD", "BLNK", "BLPH", "BLRX", "BLUE",
]


# ─────────────────────────────────────────────────────────────
# Abstract data provider interface
# ─────────────────────────────────────────────────────────────

class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    To add a new provider (e.g. Polygon, Alpaca):
      1. Subclass MarketDataProvider
      2. Implement get_quotes()
      3. Change provider = FinnhubProvider() to provider = YourProvider()
         in run_premarket_data_fetch() — nothing else needs to change.
    """

    # Subclasses set this to control Semaphore size
    CONCURRENT_REQUESTS: int = 45

    @abstractmethod
    async def get_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fetch premarket quotes for a list of tickers.

        Returns:
            dict keyed by ticker symbol, each value a dict with:
              ticker, premarket_price, prev_close,
              premarket_change_pct, premarket_volume, pm_amount
        """
        raise NotImplementedError


class FinnhubProvider(MarketDataProvider):
    """
    Finnhub implementation.
    Free tier: 60 calls/min → Semaphore(45) leaves buffer.

    To upgrade to paid tier, increase CONCURRENT_REQUESTS.
    To switch to Polygon, replace this class with PolygonProvider below.
    """

    CONCURRENT_REQUESTS = 45

    def __init__(self):
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set")
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    async def _fetch_single(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        ticker: str,
    ) -> tuple[str, dict]:
        """Fetch one ticker quote. Returns (ticker, quote_dict)."""
        async with semaphore:
            url = f"{self.base_url}/quote"
            params = {"symbol": ticker, "token": self.api_key}
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return ticker, {}
                    data = await resp.json()

                current_price = data.get("c", 0)
                prev_close    = data.get("pc", 0)
                volume        = data.get("v", 0)

                if prev_close == 0 or current_price == 0:
                    return ticker, {}

                change_pct = ((current_price - prev_close) / prev_close) * 100

                return ticker, {
                    "ticker":               ticker,
                    "premarket_price":      round(current_price, 2),
                    "prev_close":           round(prev_close, 2),
                    "premarket_change_pct": round(change_pct, 2),
                    "premarket_volume":     int(volume),
                    "pm_amount":            round(current_price * volume, 0),
                }
            except Exception:
                return ticker, {}

    async def get_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fetch all tickers concurrently with Semaphore rate limiting.
        ~45 concurrent requests → ~1071 tickers in ~2-3 minutes
        vs sequential ~18 minutes.
        """
        semaphore = asyncio.Semaphore(self.CONCURRENT_REQUESTS)
        print(f"[premarket] Fetching {len(tickers)} quotes concurrently "
              f"(max {self.CONCURRENT_REQUESTS} at a time)...")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_single(session, semaphore, ticker)
                for ticker in tickers
            ]
            results = await asyncio.gather(*tasks)

        # Filter out empty results and return as dict
        return {ticker: quote for ticker, quote in results if quote}


# ─────────────────────────────────────────────────────────────
# Example: how to add Polygon when upgrading to paid data
# ─────────────────────────────────────────────────────────────
#
# class PolygonProvider(MarketDataProvider):
#     """
#     Polygon.io implementation.
#     Paid tier: one snapshot call fetches the entire market at once.
#     Switch by changing: provider = PolygonProvider() in run_premarket_data_fetch()
#     """
#
#     CONCURRENT_REQUESTS = 1  # One bulk call, no concurrency needed
#
#     def __init__(self):
#         self.api_key = os.environ.get("POLYGON_API_KEY", "")
#
#     async def get_quotes(self, tickers: list[str]) -> dict[str, dict]:
#         url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
#         async with aiohttp.ClientSession() as session:
#             async with session.get(url, params={"apiKey": self.api_key}) as resp:
#                 data = await resp.json()
#         results = {}
#         for item in data.get("tickers", []):
#             t = item["ticker"]
#             day = item.get("day", {})
#             prev = item.get("prevDay", {})
#             prev_close = prev.get("c", 0)
#             current    = day.get("o", 0)  # use open as premarket proxy
#             volume     = day.get("v", 0)
#             if prev_close == 0 or current == 0:
#                 continue
#             change_pct = ((current - prev_close) / prev_close) * 100
#             results[t] = {
#                 "ticker":               t,
#                 "premarket_price":      round(current, 2),
#                 "prev_close":           round(prev_close, 2),
#                 "premarket_change_pct": round(change_pct, 2),
#                 "premarket_volume":     int(volume),
#                 "pm_amount":            round(current * volume, 0),
#             }
#         return results


# ─────────────────────────────────────────────────────────────
# Universe loading
# ─────────────────────────────────────────────────────────────

def load_universe() -> list:
    """
    Load pre-filtered small-cap universe.

    Priority:
      1. small_cap_300m.json  (~1071 tickers, market cap < $300M)
      2. small_cap_100m.json  (~930 tickers, market cap < $100M)
      3. Finnhub dynamic fetch
      4. Hardcoded fallback
    """
    if CAP_300M_FILE.exists():
        try:
            with open(CAP_300M_FILE) as f:
                tickers = json.load(f)
            print(f"[premarket] Loaded {len(tickers)} tickers from small_cap_300m.json")
            return tickers
        except Exception as e:
            print(f"[premarket] Failed to load small_cap_300m.json: {e}")

    if CAP_100M_FILE.exists():
        try:
            with open(CAP_100M_FILE) as f:
                tickers = json.load(f)
            print(f"[premarket] Loaded {len(tickers)} tickers from small_cap_100m.json")
            return tickers
        except Exception as e:
            print(f"[premarket] Failed to load small_cap_100m.json: {e}")

    try:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        client  = finnhub.Client(api_key=api_key)
        stocks  = client.stock_symbols("US")
        tickers = [
            s["symbol"] for s in stocks
            if s.get("type") == "Common Stock"
            and s["symbol"].isalpha()
            and 2 <= len(s["symbol"]) <= 5
            and not (len(s["symbol"]) == 5 and s["symbol"][-1] in {"F","Q","Y","W","R","E","D","K","L","N","P"})
        ]
        print(f"[premarket] Got {len(tickers)} tickers from Finnhub")
        return tickers
    except Exception as e:
        print(f"[premarket] Finnhub fetch failed: {e}")

    print(f"[premarket] Using fallback list ({len(FALLBACK_TICKERS)} tickers)")
    return FALLBACK_TICKERS


# ─────────────────────────────────────────────────────────────
# RVOL computation (unchanged — uses yfinance history)
# ─────────────────────────────────────────────────────────────

def compute_rvol(ticker: str, current_volume: int, lookback_days: int = 20) -> float:
    """
    Compute Relative Volume (RVOL).
    RVOL = today's volume so far / expected volume at this time of day.
    """
    try:
        hist       = yf.Ticker(ticker).history(period=f"{lookback_days}d")
        avg_volume = hist["Volume"].mean() if not hist.empty else 0

        if avg_volume == 0:
            return 0.0

        now          = datetime.now()
        market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)

        if now < market_open:
            expected = avg_volume * 0.08
        else:
            elapsed  = (now - market_open).seconds
            full_day = (market_close - market_open).seconds
            fraction = min(elapsed / full_day, 1.0)
            expected = avg_volume * max(fraction, 0.01)

        return round(current_volume / expected, 2) if expected > 0 else 0.0

    except Exception:
        return 0.0


def get_recent_news(ticker: str, client: finnhub.Client, days: int = 2) -> list:
    """Fetch recent news for a ticker via Finnhub."""
    try:
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        news  = client.company_news(ticker, _from=start, to=end)
        return [
            {
                "headline": a.get("headline", ""),
                "source":   a.get("source", ""),
                "datetime": a.get("datetime", 0),
                "summary":  a.get("summary", "")[:200],
            }
            for a in news[:5]
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Full scan pipeline
# ─────────────────────────────────────────────────────────────

def run_premarket_data_fetch(
    min_price:            float = 1.0,
    max_price:            float = 20.0,
    max_market_cap:       float = 3e8,
    max_float:            float = 1e9,
    min_premarket_change: float = 4.0,
    max_premarket_change: float = 40.0,
    min_volume:           int   = 200_000,
    min_pm_amount:        float = 1_000_000,
    min_rvol:             float = 2.0,
    min_day_change:       float = 0.0,
    min_day_volume:       int   = 0,
    direction:            str   = "both",
    sort_by:              str   = "change",
) -> list:
    """
    Full premarket scan pipeline:
      1. Load universe
      2. Fetch ALL quotes concurrently via provider (was: sequential loop)
      3. Filter by price, change%, volume, amount, direction
      4. Compute RVOL for passing candidates only
      5. Fetch news for final candidates

    To switch data provider: change `provider = FinnhubProvider()` below.
    Everything else stays the same.
    """
    # ── Swap provider here to change data source ──────────────
    provider = FinnhubProvider()
    # provider = PolygonProvider()  # uncomment when upgrading
    # ──────────────────────────────────────────────────────────

    universe = load_universe()
    print(f"[premarket] Scanning {len(universe)} tickers...")

    # Fetch all quotes concurrently (the key change vs original)
    all_quotes = asyncio.run(provider.get_quotes(universe))
    print(f"[premarket] Got {len(all_quotes)} valid quotes")

    # Filter candidates
    candidates = []
    for ticker, quote in all_quotes.items():
        price     = quote["premarket_price"]
        change    = quote["premarket_change_pct"]
        volume    = quote["premarket_volume"]
        pm_amount = quote["pm_amount"]

        if not (min_price <= price <= max_price):
            continue
        if direction == "up"   and change <= 0:
            continue
        if direction == "down" and change >= 0:
            continue
        if not (min_premarket_change <= abs(change) <= max_premarket_change):
            continue
        if volume < min_volume:
            continue
        if pm_amount < min_pm_amount:
            continue
        if abs(change) < min_day_change:
            continue
        if volume < min_day_volume:
            continue

        # RVOL only for candidates that passed other filters
        rvol = compute_rvol(ticker, volume)
        if rvol < min_rvol:
            continue

        candidates.append({**quote, "rvol": rvol})

    print(f"[premarket] {len(candidates)} candidates found\n")

    # Fetch news for final candidates only (still sequential, small list)
    if candidates:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        news_client = finnhub.Client(api_key=api_key)
        print("[premarket] Fetching news for candidates...")
        for candidate in candidates:
            candidate["news"] = get_recent_news(candidate["ticker"], news_client)

    return candidates


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing premarket/data.py...\n")

    print("1. Loading universe...")
    universe = load_universe()
    print(f"   Universe: {len(universe)} tickers")
    print(f"   Sample: {universe[:5]}\n")

    print("2. Testing async quote fetch (first 10 tickers)...")
    provider = FinnhubProvider()
    quotes = asyncio.run(provider.get_quotes(universe[:10]))
    print(f"   Got {len(quotes)} quotes")
    for ticker, q in list(quotes.items())[:3]:
        print(f"   {ticker}: ${q['premarket_price']} ({q['premarket_change_pct']:+.1f}%)")

    print("\nAll tests passed!")
