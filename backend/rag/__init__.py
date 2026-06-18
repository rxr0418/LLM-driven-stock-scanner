"""
rag/ — Data fetchers for the RAG knowledge base.

Scripts (run manually or via cron):
  fetch_events.py        : Upcoming earnings dates (yfinance)
  fetch_ratings.py       : Analyst upgrades/downgrades (yfinance)
  fetch_sec.py           : 8-K / 10-Q summaries (SEC EDGAR + Haiku)
  seed_sector_knowledge  : Sector-level trading rules (one-time seed)
"""
