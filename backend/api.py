"""
api.py - FastAPI backend for the LLM-driven stock scanner.

Endpoints:
  GET  /api/health          → health check
  GET  /api/regime          → current market regime
  GET  /api/scan            → run full scan (factor only, no LLM)
  POST /api/scan/full       → run full scan with LLM analysis
  GET  /api/history         → list saved watchlists
  GET  /api/history/{id}    → load a specific saved watchlist

Run with:
  uvicorn api:app --reload --port 8000
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Local modules ─────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent))

from swing.data        import fetch_price_data, fetch_news_batch, UNIVERSE
from swing.regime      import detect_regime
from swing.scanner     import run_scan
from swing.llm_analyst import analyze_watchlist
from swing.main        import stable_regime, save_results

from premarket.premarket_data     import run_premarket_data_fetch
from premarket.premarket_catalyst import analyze_candidates_batch
from premarket.premarket_scanner  import run_premarket_scan, rank_candidates

from database import log_scan_results

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM-Driven Stock Scanner",
    description="Regime-adaptive factor scanner with LLM news analysis",
    version="1.0.0",
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://llm-driven-stock-scanner.vercel.app",],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache to avoid re-running expensive scans
_cache: dict = {}
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────

class RegimeResponse(BaseModel):
    regime: str
    vix: float
    realized_vol: float
    trend_strength: float
    volatile_votes: int
    trending_votes: int
    recommended_factors: list
    description: str
    timestamp: str


class StockSignal(BaseModel):
    ticker: str
    score: float
    signal: Optional[str] = None
    confidence: Optional[int] = None
    news_alignment: Optional[str] = None
    reason: Optional[str] = None
    risk_flag: Optional[str] = None
    news_titles: Optional[list] = None


class ScanResponse(BaseModel):
    regime: str
    description: str
    factors_used: list
    long_candidates: list
    short_candidates: list
    timestamp: str
    has_llm_analysis: bool = False
    vix: float = 0.0
    realized_vol: float = 0.0
    trend_strength: float = 0.0


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "universe_size": len(UNIVERSE),
    }


@app.get("/api/regime", response_model=RegimeResponse)
def get_regime():
    """
    Detect and return the current market regime.
    Uses stability filter to prevent rapid switching.
    """
    raw    = detect_regime()
    stable = stable_regime(raw, min_streak=2)
    return stable


@app.get("/api/scan", response_model=ScanResponse)
def get_scan(top_n: int = 10):
    """
    Run factor scan without LLM analysis.
    Fast — no API calls, returns in ~10 seconds.

    Query params:
      top_n : number of candidates per side (default 10)
    """
    print(f"[api] Running factor scan, top_n={top_n}...")

    price_data    = fetch_price_data(UNIVERSE, lookback_days=90)
    raw_regime    = detect_regime()
    regime_result = stable_regime(raw_regime, min_streak=2)
    scan_results  = run_scan(price_data, regime_result, top_n=top_n)

    if "error" in scan_results:
        raise HTTPException(status_code=500, detail=scan_results["error"])
    print(f"[api] regime_result keys: {list(regime_result.keys())}")
    print(f"[api] vix={regime_result.get('vix')}, rvol={regime_result.get('realized_vol')}")

    return {
    **scan_results,
    "vix":            regime_result["vix"],
    "realized_vol":   regime_result["realized_vol"],
    "trend_strength": regime_result["trend_strength"],
    "has_llm_analysis": False,
    }


@app.post("/api/scan/full", response_model=ScanResponse)
def get_full_scan(top_n: int = 10, save: bool = True, lang: str = "en"):
    """
    Run full scan with LLM news analysis.
    Slower — makes Claude API calls for each candidate.

    Query params:
      top_n : number of candidates per side (default 10)
      save  : save results to disk (default true)
    """
    print(f"[api] Running full scan with LLM, top_n={top_n}...")

    price_data    = fetch_price_data(UNIVERSE, lookback_days=90)
    raw_regime    = detect_regime()
    regime_result = stable_regime(raw_regime, min_streak=2)
    scan_results  = run_scan(price_data, regime_result, top_n=top_n)

    if "error" in scan_results:
        raise HTTPException(status_code=500, detail=scan_results["error"])

    # Fetch news for all candidates
    all_tickers = (
        [x["ticker"] for x in scan_results["long_candidates"]] +
        [x["ticker"] for x in scan_results["short_candidates"]]
    )
    news_data = fetch_news_batch(all_tickers, max_articles=5)

    # LLM analysis
    watchlist = analyze_watchlist(scan_results, news_data, top_n=top_n, lang=lang)

    # Save to disk
    if save:
        filepath = save_results(watchlist)
        print(f"[api] Saved to {filepath}")

    # Cache latest result
    _cache["latest"] = watchlist

    return {
    **watchlist,
    "long_candidates":  watchlist.get("long_watchlist", []),
    "short_candidates": watchlist.get("short_watchlist", []),
    "has_llm_analysis": True,
}

@app.get("/api/latest")
def get_latest():
    """
    Return the most recently cached scan result.
    If no scan has been run yet, load the most recent saved file.
    """
    # Try in-memory cache first
    if "latest" in _cache:
        return _cache["latest"]

    # Fall back to most recent saved file
    saved_files = sorted(RESULTS_DIR.glob("watchlist_*.json"), reverse=True)
    if saved_files:
        with open(saved_files[0]) as f:
            data = json.load(f)
        _cache["latest"] = data
        return data

    raise HTTPException(
        status_code=404,
        detail="No scan results available. Run /api/scan/full first."
    )


@app.get("/api/history")
def get_history():
    """
    List all saved watchlist files with metadata.
    """
    saved_files = sorted(RESULTS_DIR.glob("watchlist_*.json"), reverse=True)
    history = []
    for f in saved_files[:20]:  # return last 20
        try:
            with open(f) as fp:
                data = json.load(fp)
            history.append({
                "id":        f.stem,
                "timestamp": data.get("timestamp", ""),
                "regime":    data.get("regime", ""),
                "factors":   data.get("factors_used", []),
                "filename":  f.name,
            })
        except Exception:
            continue
    return {"history": history}


@app.get("/api/history/{scan_id}")
def get_historical_scan(scan_id: str):
    """
    Load a specific saved watchlist by ID (filename stem).
    """
    filepath = RESULTS_DIR / f"{scan_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

    with open(filepath) as f:
        return json.load(f)


@app.get("/api/universe")
def get_universe():
    """Return the current stock universe."""
    return {"tickers": UNIVERSE, "count": len(UNIVERSE)}

@app.get("/api/premarket/scan")
def get_premarket_scan(
    # 股票基本条件
    min_price:    float = 1.0,
    max_price:    float = 20.0,
    max_market_cap: float = 3e8,
    max_float:    float = 1e9,    # 默认不设限用很大的数
    # 盘前条件
    min_change:   float = 4.0,
    max_change:   float = 40.0,
    min_pm_volume: int  = 200_000,
    min_pm_amount: float = 1e6,   # 盘前成交额，默认100万
    min_rvol:     float = 2.0,
    # 盘中条件
    min_day_change: float = 0.0,  # 单日涨幅，默认不限
    min_day_volume: int   = 0,    # 当日成交量，默认不限
    # 排序
    sort_by:      str   = "change",  # change/rvol/volume/amount
    # 方向
    direction:    str   = "both",    # both/up/down
):
    candidates = run_premarket_data_fetch(
        min_price=min_price,
        max_price=max_price,
        max_market_cap=max_market_cap,
        max_float=max_float,
        min_premarket_change=min_change,
        max_premarket_change=max_change,
        min_volume=min_pm_volume,
        min_pm_amount=min_pm_amount,
        min_rvol=min_rvol,
        min_day_change=min_day_change,
        min_day_volume=min_day_volume,
        direction=direction,
        sort_by=sort_by,
    )
    ranked = rank_candidates(candidates, sort_by=sort_by)
    try:
        log_scan_results(ranked)
    except Exception as e:
        print(f"[api] DB logging failed: {e}")
    return {
        "candidates": ranked,
        "count":      len(ranked),
        "has_llm":    False,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/api/premarket/scan/full")
def get_premarket_full_scan(
    min_price:     float = 1.0,
    max_price:     float = 20.0,
    max_market_cap: float = 3e8,
    max_float:     float = 1e9,
    min_change:    float = 4.0,
    max_change:    float = 40.0,
    min_pm_volume: int   = 200_000,
    min_pm_amount: float = 1e6,
    min_rvol:      float = 2.0,
    min_day_change: float = 0.0,
    min_day_volume: int   = 0,
    sort_by:       str   = "change",
    direction:     str   = "both",
    lang:          str   = "en",
):
    candidates = run_premarket_data_fetch(
        min_price=min_price,
        max_price=max_price,
        max_market_cap=max_market_cap,
        max_float=max_float,
        min_premarket_change=min_change,
        max_premarket_change=max_change,
        min_volume=min_pm_volume,
        min_pm_amount=min_pm_amount,
        min_rvol=min_rvol,
        min_day_change=min_day_change,
        min_day_volume=min_day_volume,
        direction=direction,
        sort_by=sort_by,
    )
    if candidates:
        candidates = analyze_candidates_batch(candidates, lang=lang)
    ranked = rank_candidates(candidates, sort_by=sort_by)
    try:
        log_scan_results(ranked)
    except Exception as e:
        print(f"[api] DB logging failed: {e}")
    return {
        "candidates": ranked,
        "count":      len(ranked),
        "has_llm":    True,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)