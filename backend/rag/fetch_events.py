"""
rag/fetch_events.py - Fetch upcoming earnings dates and store in events table.

Data source: yfinance (free)
Run: python fetch_events.py [--tickers AAPL MSFT ...]
     Defaults to swing UNIVERSE if no tickers specified.

Events stored:
  - EARNINGS  : next earnings date from yfinance
  - Stale entries (event_date < today) auto-deleted on each run
"""

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from swing.data import UNIVERSE


def fetch_earnings_date(ticker: str) -> dict | None:
    """Return next earnings date for ticker, or None if unavailable."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).calendar
        if not info:
            return None

        # New yfinance returns a dict; old versions returned a DataFrame
        if isinstance(info, dict):
            dates = info.get("Earnings Date", [])
            if not dates:
                return None
            ev_date = dates[0]
        else:
            # DataFrame fallback
            if "Earnings Date" not in info.index:
                return None
            vals = info.loc["Earnings Date"].dropna().tolist()
            if not vals:
                return None
            ev_date = vals[0]

        if hasattr(ev_date, "date"):
            ev_date = ev_date.date()
        days_away = (ev_date - date.today()).days
        return {
            "ticker":      ticker,
            "event_type":  "EARNINGS",
            "event_date":  str(ev_date),
            "days_away":   days_away,
            "description": f"{ticker} earnings expected {ev_date} ({days_away}d away)",
        }
    except Exception as e:
        print(f"  [events] {ticker}: yfinance error — {e}")
    return None


def run(tickers: list[str]) -> None:
    from database import upsert_events, delete_stale_events

    print(f"[fetch_events] Fetching earnings dates for {len(tickers)} tickers...")
    delete_stale_events()

    saved = 0
    for ticker in tickers:
        ev = fetch_earnings_date(ticker)
        if ev and ev["days_away"] >= 0:
            upsert_events([ev])
            print(f"  {ticker}: earnings {ev['event_date']} ({ev['days_away']}d)")
            saved += 1
        else:
            print(f"  {ticker}: no upcoming earnings found")

    print(f"[fetch_events] Done — {saved}/{len(tickers)} events saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else UNIVERSE
    run(tickers)
