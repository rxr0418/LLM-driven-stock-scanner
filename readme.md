# LLM-Driven Stock Scanner

A full-stack stock scanner combining **regime-adaptive factor selection**, **premarket momentum detection**, and **LLM-powered catalyst analysis** to surface actionable trading candidates.

---

## What It Does

**Swing Trade mode** — runs daily before market open:
1. Detects the current market regime (TRENDING / VOLATILE / NEUTRAL) using VIX, realized volatility, and trend consistency
2. Selects factor signals based on regime — momentum in trending markets, reversal in volatile markets
3. Scores and ranks S&P 500 stocks cross-sectionally
4. Fetches recent news for top candidates
5. Uses Claude (Anthropic) to analyze whether news supports or contradicts each factor signal
6. Outputs a ranked watchlist with signal, confidence score, and one-line reason per stock

**Day Trade mode** — runs during premarket (4:00–9:30 AM ET):
1. Loads a pre-filtered universe of 1,000+ small-cap stocks (market cap < $300M)
2. Fetches real-time premarket quotes from Finnhub for each ticker
3. Filters by price, float, premarket change %, volume, dollar amount, and RVOL
4. All filter parameters are user-configurable via a custom scan UI
5. Optional LLM deep scan analyzes catalyst quality (FDA approval, earnings beat, contract win, etc.) and outputs TRADE / WATCH / AVOID signals

---

## Live Demo

| | URL |
|--|--|
| Dashboard | https://llm-driven-stock-scanner.vercel.app |
| API Docs  | https://llm-driven-stock-scanner-production.up.railway.app/docs |

---

## Sample Output

**Swing Trade**
```
Regime  : NEUTRAL
Factors : momentum_20d + reversal_5d + volume_spike
VIX     : 18.4 | Realized Vol: 10.7% | Trend: 55%

LONG CANDIDATES
  ADBE  | BUY     | conf=72% | Agentic AI expansion supports momentum; Figma competition warrants caution.
  QCOM  | BUY     | conf=62% | Rebound thesis intact but ARM competition noted.
  HD    | BUY     | conf=62% | Strong factor signal; mixed news offset by bargain valuation framing.
  SBUX  | NEUTRAL | conf=48% | $400M restructuring charge undermines bullish momentum signal.
  BA    | NEUTRAL | conf=42% | China deal disappointed; share slide contradicts perfect factor score.

SHORT CANDIDATES
  LLY   | AVOID   | conf=62% | Q1 guidance upgrade and obesity trial results contradict short signal.
  NFLX  | AVOID   | conf=38% | Raised guidance and 283% analyst upside directly contradict short.
  COP   | AVOID   | conf=38% | Energy sector rally contradicts bearish signal.
```

**Day Trade (premarket)**
```
🌅 PREMARKET MOVERS — 08:14 ET

  BBAI  | TRADE | conf=81% | DoD contract win ($240M) directly catalyzes gap-up; volume confirms.
         +18.4%  RVOL=8.5x  Vol=420K sh  Float=12M sh

  OCGN  | WATCH | conf=55% | FDA Fast Track granted; positive but not approval-level catalyst.
         +12.1%  RVOL=5.2x  Vol=280K sh  Float=8M sh

  MARA  | WATCH | conf=48% | Bitcoin up 4% overnight; sector move, not company-specific catalyst.
         +9.8%   RVOL=3.1x  Vol=510K sh  Float=95M sh
```

---

## Key Design Decisions

### Regime-Adaptive Factor Selection

Factor effectiveness depends on market environment. Empirical research on this dataset shows:

| Period | Regime | Best Factor | Mean IC |
|--------|--------|-------------|---------|
| 2019–2023 | Trending (low VIX) | momentum_20d | +0.030 |
| 2025–2026 | Volatile (tariff shock) | reversal_20d | +0.037 |

Momentum factors that delivered IC > 0.03 during 2019–2023 turned negative after the April 2025 tariff shock. A scanner unaware of this would keep applying momentum signals in a regime where they no longer work. This regime-switching insight was discovered organically through systematic backtesting, not assumed in advance.

### Why Small-Cap for Day Trade?

Small-cap stocks (market cap < $300M, float < 20M shares) are the primary target for premarket scanning because the same dollar volume produces a proportionally larger price move. With 5M shares of float, $500K of buying pressure can move a stock 10%+. Large-cap stocks require orders of magnitude more capital to produce the same percentage move.

### LLM for Catalyst Quality, Not Factor Computation

Factor computation requires precise arithmetic on large DataFrames — LLMs are unreliable for this. LLM analysis is applied only where language understanding matters: evaluating whether a catalyst is genuine (FDA approval vs. FDA Fast Track), whether it is proportional to the price move, and what the primary risk is at open.

### RVOL as the Core Day Trade Signal

Relative Volume (RVOL = today's volume / expected volume at this time of day) surfaces genuine volume anomalies across stocks of different sizes. A stock with 200K shares traded by 8 AM when its 8 AM historical average is 20K (RVOL = 10x) is far more interesting than a large-cap trading at its normal pace (RVOL = 1.0x).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI (api.py)                        │
│           Swing Trade endpoints + Day Trade endpoints        │
└──────┬───────────────────────────────┬────────────────────────┘
       │                               │
       ▼                               ▼
  swing/                         premarket/
  ─────────────────────          ──────────────────────────
  data.py       OHLCV+news       premarket_data.py
  regime.py     Regime detect      Universe, quotes, RVOL
  scanner.py    Factor scores     premarket_catalyst.py
  llm_analyst.py LLM analysis       LLM catalyst analysis
  main.py       Orchestration     premarket_scanner.py
                                    Scoring and ranking

┌──────────────────────────────────────────────────────────────┐
│                   React Frontend (Vite)                      │
│     Swing Trade tab              Day Trade tab               │
│     Factor Scan                  Custom Scan (14 params)     │
│     Full Scan (LLM)              Deep Scan (LLM catalyst)    │
└──────────────────────────────────────────────────────────────┘
```

**Deployment:** Backend on Railway (auto-deploy on push), frontend on Vercel.

---

## Regime Detection

Three signals vote independently. Two or more votes determine the regime:

| Signal | VOLATILE | TRENDING |
|--------|----------|----------|
| VIX | ≥ 25 | ≤ 15 |
| Realized Vol (20d) | ≥ 20% annualized | ≤ 12% annualized |
| Trend Consistency (20d) | ≤ 40% days aligned | ≥ 60% days aligned |

A **stability filter** prevents rapid switching: a new regime must persist for 2+ consecutive days before being confirmed.

---

## Factor Library

| Factor | Expression | Best Regime |
|--------|------------|-------------|
| `reversal_5d` | `-close.diff(5)` | VOLATILE |
| `reversal_20d` | `-close.diff(20)` | VOLATILE |
| `momentum_20d` | `close.pct_change(20)` | TRENDING |
| `momentum_60d` | `close.pct_change(60)` | TRENDING |
| `volume_spike` | `volume / volume.rolling(20).mean()` | NEUTRAL |
| `vol_adjusted_reversal` | `-close.diff(5) / realized_vol_10d` | VOLATILE |

Factors are combined using cross-sectional percentile ranking then equal-weight averaging.

---

## Premarket Scanner

**Universe:** 1,071 pre-filtered US small-cap stocks (market cap < $300M, refreshed weekly via yfinance).

**Default filter parameters:**

| Parameter | Default | Unit |
|-----------|---------|------|
| Price range | $1 – $20 | USD |
| Market cap | ≤ 300 | M$ |
| Float | ≤ 20,000 | K sh (= 20M shares) |
| Premarket change | 4% – 40% | % |
| Premarket volume | ≥ 200 | K sh |
| Premarket amount | ≥ 1,000 | K$ (= $1M) |
| RVOL | ≥ 2x | x |

All parameters are user-configurable. Units are fixed (K/M) regardless of language selection to prevent input ambiguity.

**RVOL computation:**
- Premarket (before 9:30 AM ET): `current_volume / (avg_daily_volume × 0.08)`
- Intraday: `current_volume / (avg_daily_volume × elapsed_fraction)`

**LLM catalyst classification (Deep Scan):**
`FDA_APPROVAL` · `EARNINGS_BEAT` · `CONTRACT_WIN` · `ANALYST_UPGRADE` · `SECTOR_MOVE` · `UNKNOWN`

---

## LLM Prompt Engineering

**Few-shot examples** cover the main scenarios: catalyst confirms signal, catalyst contradicts signal, ambiguous sector move.

**Confidence calibration rubric:**
```
90–100: Hard data directly confirms (FDA approval, earnings beat + raised guidance)
75–89 : Clear company-specific catalyst present
55–74 : Positive catalyst but proportionality uncertain
35–54 : Sector move or mixed evidence
15–34 : Catalyst weak relative to price move
 0–14 : No real catalyst — likely manipulation risk
```

**Forced JSON output** ensures reliable parsing across all responses.

---

## Project Structure

```
scanner/
├── backend/
│   ├── api.py                     # FastAPI entry point
│   ├── swing/
│   │   ├── data.py                # OHLCV + news fetch (yfinance)
│   │   ├── regime.py              # Regime detection
│   │   ├── scanner.py             # Factor computation and ranking
│   │   ├── llm_analyst.py         # LLM news analysis
│   │   └── main.py                # Pipeline orchestration
│   └── premarket/
│       ├── premarket_data.py      # Universe loading, quotes, RVOL
│       ├── premarket_catalyst.py  # LLM catalyst analysis
│       ├── premarket_scanner.py   # Scoring and ranking
│       ├── small_cap_300m.json    # Pre-filtered universe (<$300M)
│       └── small_cap_100m.json    # Tighter universe (<$100M)
├── frontend/
│   └── src/
│       ├── App.jsx                # React app (Swing + Day Trade tabs)
│       └── App.css
└── README.md
```

---

## Setup

**1. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**2. Set API keys**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export FINNHUB_API_KEY="your-finnhub-key"
```

**3. Run backend**
```bash
uvicorn api:app --reload --port 8000
```

**4. Run frontend**
```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/regime` | Current market regime |
| GET | `/api/scan` | Swing factor scan (no LLM) |
| POST | `/api/scan/full` | Swing scan with LLM analysis |
| GET | `/api/premarket/scan` | Premarket scan with custom params |
| POST | `/api/premarket/scan/full` | Premarket scan with LLM catalyst analysis |

---

## Limitations

- Swing scanner universe limited to ~55 large-cap S&P 500 stocks
- Premarket scanner uses Finnhub free tier (60 calls/min) — full scan takes ~18 minutes
- No real-time WebSocket streaming (snapshot-based, manual refresh)
- News source limited to Yahoo Finance headlines

---

## Planned Improvements

- Earnings calendar integration to flag binary event risk automatically
- Auto-refresh toggle for premarket scanner (every N minutes)
- Historical mover logging to build a data-driven high-frequency watchlist
- Dual-signal overlay: flag stocks appearing in both Swing and Day Trade results simultaneously
- Upgrade to Polygon.io bulk snapshot API for sub-minute full-market scans
- ICIR-weighted factor combination to replace equal-weight averaging

---

## Motivation

Traditional scanners (Finviz, Trade Ideas) apply fixed screening rules regardless of market conditions. This project tests the hypothesis that combining regime-adaptive quantitative signals with LLM catalyst analysis produces higher-quality candidates than either approach alone.

The regime-switching insight is empirical: momentum factors delivered IC > 0.03 during the 2019–2023 trending market but turned negative after the April 2025 tariff shock, while reversal factors recovered to IC > 0.037 in the same volatile period. This was discovered through systematic backtesting across 50+ factor variants, not assumed in advance.

---

## References

- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

---

## Author

Jamie Ren · B.S. Computer Science & Statistics, University of Toronto · M.S. Information Science, Trine University

*Built as part of quantitative research and AI engineering skill development.*
