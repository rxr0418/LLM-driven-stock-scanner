"""
rag/fetch_ratings.py - Fetch recent analyst upgrades/downgrades via yfinance.

Stores the last 30 days of rating changes with embeddings for semantic search.
Run: python fetch_ratings.py [--tickers AAPL MSFT ...] [--days 30]
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


def fetch_ratings(ticker: str, lookback_days: int = 30) -> list[dict]:
    """Return recent analyst rating changes for ticker."""
    try:
        import yfinance as yf
        rec = yf.Ticker(ticker).upgrades_downgrades
        if rec is None or rec.empty:
            return []

        cutoff = date.today() - timedelta(days=lookback_days)
        rec = rec[rec.index.date >= cutoff]
        if rec.empty:
            return []

        results = []
        for ts, row in rec.iterrows():
            from_grade = str(row.get("FromGrade", "") or "")
            to_grade   = str(row.get("ToGrade",   "") or "")
            action     = str(row.get("Action",    "") or "")
            firm       = str(row.get("Firm", "Unknown") or "Unknown")

            if not to_grade:
                continue

            if from_grade:
                summary = f"{firm} {action} {ticker} from {from_grade} to {to_grade}"
            else:
                summary = f"{firm} initiates {ticker} at {to_grade}"

            results.append({
                "ticker":      ticker,
                "firm":        firm,
                "old_rating":  from_grade,
                "new_rating":  to_grade,
                "action":      action,
                "rating_date": str(ts.date()),
                "summary":     summary,
            })
        return results

    except Exception as e:
        print(f"  [ratings] {ticker}: yfinance error — {e}")
        return []


def run(tickers: list[str], lookback_days: int = 30) -> None:
    from database import upsert_analyst_ratings

    print(f"[fetch_ratings] Fetching ratings for {len(tickers)} tickers (last {lookback_days}d)...")

    total = 0
    for ticker in tickers:
        ratings = fetch_ratings(ticker, lookback_days)
        if ratings:
            upsert_analyst_ratings(ratings)
            for r in ratings:
                print(f"  {ticker}: {r['summary']} ({r['rating_date']})")
            total += len(ratings)
        else:
            print(f"  {ticker}: no recent ratings")

    print(f"[fetch_ratings] Done — {total} rating changes saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else UNIVERSE
    run(tickers, args.days)
