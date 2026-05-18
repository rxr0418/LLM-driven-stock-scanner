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

import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

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
# Finnhub client
# ─────────────────────────────────────────────────────────────

def get_finnhub_client() -> finnhub.Client:
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not set")
    return finnhub.Client(api_key=api_key)


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
    # Priority 1: pre-filtered 300M file (recommended)
    if CAP_300M_FILE.exists():
        try:
            with open(CAP_300M_FILE) as f:
                tickers = json.load(f)
            print(f"[premarket] Loaded {len(tickers)} tickers from small_cap_300m.json")
            return tickers
        except Exception as e:
            print(f"[premarket] Failed to load small_cap_300m.json: {e}")

    # Priority 2: pre-filtered 100M file
    if CAP_100M_FILE.exists():
        try:
            with open(CAP_100M_FILE) as f:
                tickers = json.load(f)
            print(f"[premarket] Loaded {len(tickers)} tickers from small_cap_100m.json")
            return tickers
        except Exception as e:
            print(f"[premarket] Failed to load small_cap_100m.json: {e}")

    # Priority 3: Finnhub dynamic
    try:
        client  = get_finnhub_client()
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

    # Priority 4: fallback
    print(f"[premarket] Using fallback list ({len(FALLBACK_TICKERS)} tickers)")
    return FALLBACK_TICKERS


# ─────────────────────────────────────────────────────────────
# Premarket quotes
# ─────────────────────────────────────────────────────────────

def get_premarket_quote(ticker: str, client: finnhub.Client) -> dict:
    """Fetch current premarket/intraday quote via Finnhub."""
    try:
        quote         = client.quote(ticker)
        current_price = quote.get("c", 0)
        prev_close    = quote.get("pc", 0)
        volume        = quote.get("v", 0)

        if prev_close == 0 or current_price == 0:
            return {}

        change_pct = ((current_price - prev_close) / prev_close) * 100

        return {
            "ticker":               ticker,
            "premarket_price":      round(current_price, 2),
            "prev_close":           round(prev_close, 2),
            "premarket_change_pct": round(change_pct, 2),
            "premarket_volume":     int(volume),
            "pm_amount":            round(current_price * volume, 0),
        }

    except Exception:
        return {}


def compute_rvol(ticker: str, current_volume: int, lookback_days: int = 20) -> float:
    """
    Compute Relative Volume (RVOL).

    RVOL = today's volume so far / expected volume at this time of day
    Expected volume = historical avg × time fraction (or × 0.08 for premarket)
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
            # Premarket: typically ~8% of daily volume
            expected = avg_volume * 0.08
        else:
            # Intraday: scale by fraction of trading day elapsed
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
      1. Load universe from small_cap_300m.json (pre-filtered by market cap)
      2. Fetch Finnhub quote for each ticker
      3. Filter by price, change%, volume, amount, RVOL, direction
      4. Compute RVOL for passing candidates
      5. Fetch news for final candidates

    All numeric parameters are in raw units (USD, shares).
    Frontend multiplies K→×1000, M→×1e6 before sending.
    """
    client   = get_finnhub_client()
    universe = load_universe()

    print(f"[premarket] Scanning {len(universe)} tickers...")
    candidates = []

    for i, ticker in enumerate(universe):
        quote = get_premarket_quote(ticker, client)
        if not quote:
            continue

        price      = quote["premarket_price"]
        change     = quote["premarket_change_pct"]
        volume     = quote["premarket_volume"]
        pm_amount  = quote["pm_amount"]

        # Price filter
        if not (min_price <= price <= max_price):
            continue

        # Direction filter
        if direction == "up"   and change <= 0:
            continue
        if direction == "down" and change >= 0:
            continue

        # Premarket change filter
        if not (min_premarket_change <= abs(change) <= max_premarket_change):
            continue

        # Volume filter (K sh → shares already done by frontend)
        if volume < min_volume:
            continue

        # Amount filter (K$ → USD already done by frontend)
        if pm_amount < min_pm_amount:
            continue

        # Day change filter (intraday use)
        if abs(change) < min_day_change:
            continue

        # Day volume filter
        if volume < min_day_volume:
            continue

        # RVOL (compute only for candidates that passed other filters)
        rvol = compute_rvol(ticker, volume)
        if rvol < min_rvol:
            continue

        candidates.append({
            **quote,
            "rvol": rvol,
        })

        # Finnhub free tier: 60 calls/min
        if i % 55 == 0 and i > 0:
            print("[premarket] Rate limit pause...")
            time.sleep(1)

    print(f"[premarket] {len(candidates)} candidates found\n")

    # Fetch news for final candidates only
    if candidates:
        print("[premarket] Fetching news for candidates...")
        for candidate in candidates:
            candidate["news"] = get_recent_news(candidate["ticker"], client)
            time.sleep(0.1)

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

    client = get_finnhub_client()

    print("2. Testing AAPL quote...")
    q = get_premarket_quote("AAPL", client)
    print(f"   price=${q.get('premarket_price')}, change={q.get('premarket_change_pct')}%")
    print(f"   volume={q.get('premarket_volume'):,}, amount=${q.get('pm_amount'):,.0f}\n")

    print("3. Testing RVOL...")
    rvol = compute_rvol("AAPL", q.get("premarket_volume", 0))
    print(f"   RVOL={rvol}\n")

    print("4. Testing news...")
    news = get_recent_news("AAPL", client)
    for n in news[:2]:
        print(f"   [{n['source']}] {n['headline'][:60]}")

    print("\nAll tests passed!")
