"""
rag/fetch_sec.py - Fetch recent SEC 8-K / 10-Q filings via EDGAR, summarize with Haiku.

Flow:
  1. Resolve ticker → CIK via EDGAR company search
  2. Fetch recent filings list (8-K, 10-Q)
  3. Download filing document text
  4. Use Claude Haiku to extract a structured summary (key metrics, guidance, risks)
  5. Store summary + embedding in sec_filings table

Run: python fetch_sec.py [--tickers AAPL MSFT ...] [--types 8-K 10-Q] [--limit 3]

Note: EDGAR rate limit is 10 req/s. Script sleeps between requests.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from swing.data import UNIVERSE

EDGAR_BASE   = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
HEADERS      = {"User-Agent": "LLM-Stock-Scanner xingru.ren0418@gmail.com"}

HAIKU_PROMPT = """You are a financial analyst. Extract key information from this SEC filing.

Output JSON only (no markdown):
{
  "filing_type": "8-K or 10-Q",
  "key_points": ["bullet 1", "bullet 2", "bullet 3"],
  "revenue_guidance": "if mentioned, else null",
  "risk_flags": ["any new risks mentioned"],
  "sentiment": "POSITIVE | NEGATIVE | NEUTRAL",
  "one_line_summary": "one sentence max 20 words"
}

Focus on: earnings surprises, guidance changes, material events, management commentary.
Ignore: boilerplate legal language, exhibit lists, signature blocks."""


def get_cik(ticker: str) -> str | None:
    """Resolve ticker to CIK (10-digit zero-padded)."""
    try:
        url = f"{EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
        # Try direct ticker lookup via company tickers JSON
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                return cik
    except Exception as e:
        print(f"  [sec] CIK lookup failed for {ticker}: {e}")
    return None


def get_recent_filings(cik: str, form_types: list[str], limit: int = 3) -> list[dict]:
    """Return recent filings metadata for given form types."""
    try:
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()

        recent = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])
        primary_doc = recent.get("primaryDocument", [])

        results = []
        for form, filed_date, acc, doc in zip(forms, dates, accessions, primary_doc):
            if form in form_types:
                acc_clean = acc.replace("-", "")
                results.append({
                    "form":         form,
                    "filed_date":   filed_date,
                    "accession":    acc,
                    "document_url": f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}",
                })
                if len(results) >= limit:
                    break
        return results
    except Exception as e:
        print(f"  [sec] filing list failed for CIK {cik}: {e}")
        return []


def fetch_filing_text(url: str, max_chars: int = 8000) -> str | None:
    """Download filing document and return truncated text."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        text = r.text

        # Strip HTML tags naively
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        print(f"  [sec] document fetch failed: {e}")
        return None


def summarize_with_haiku(ticker: str, form: str, text: str) -> dict | None:
    """Use Claude Haiku to extract a structured summary."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=HAIKU_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Ticker: {ticker}\nForm: {form}\n\nFiling text:\n{text}",
            }],
        )
        raw = response.content[0].text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except Exception as e:
        print(f"  [sec] Haiku summarization failed: {e}")
    return None


def run(tickers: list[str], form_types: list[str] = None, limit: int = 2) -> None:
    if form_types is None:
        form_types = ["8-K", "10-Q"]

    from database import upsert_sec_filing

    print(f"[fetch_sec] Processing {len(tickers)} tickers, forms={form_types}...")

    total = 0
    for ticker in tickers:
        print(f"\n  {ticker}:")
        cik = get_cik(ticker)
        if not cik:
            print(f"    CIK not found, skipping")
            continue

        time.sleep(0.15)  # EDGAR rate limit
        filings = get_recent_filings(cik, form_types, limit=limit)

        for filing in filings:
            time.sleep(0.15)
            text = fetch_filing_text(filing["document_url"])
            if not text:
                continue

            summary = summarize_with_haiku(ticker, filing["form"], text)
            if not summary:
                continue

            one_line = summary.get("one_line_summary", "")
            embedding_text = (
                f"{ticker} {filing['form']} {filing['filed_date']} | "
                f"{one_line} | "
                f"sentiment={summary.get('sentiment', 'NEUTRAL')} | "
                f"key_points={'; '.join(summary.get('key_points', []))}"
            )

            upsert_sec_filing({
                "ticker":       ticker,
                "filing_type":  filing["form"],
                "filed_date":   filing["filed_date"],
                "summary":      one_line,
                "key_metrics":  summary,
                "embedding_text": embedding_text,
            })

            print(f"    {filing['form']} {filing['filed_date']}: {one_line}")
            total += 1

    print(f"\n[fetch_sec] Done — {total} filings stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--types",   nargs="*", default=["8-K", "10-Q"])
    parser.add_argument("--limit",   type=int,  default=2)
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else UNIVERSE[:5]  # default small batch
    run(tickers, args.types, args.limit)
