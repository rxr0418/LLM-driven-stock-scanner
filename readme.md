# LLM-Driven Stock Scanner

A full-stack stock scanner combining **regime-adaptive factor selection**, **premarket momentum detection**, and **LLM-powered catalyst analysis** to surface actionable trading candidates. Includes a **RAG pipeline** that grounds LLM analysis in self-accumulated historical trade data.

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
4. All filter parameters are user-configurable via a custom scan UI with input validation
5. LLM deep scan analyzes catalyst quality and outputs TRADE / WATCH / AVOID signals with entry timing
6. Scan results are logged to Supabase PostgreSQL for historical analysis and RAG

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
  BA    | NEUTRAL | conf=42% | China deal disappointed; share slide contradicts perfect factor score.

SHORT CANDIDATES
  LLY   | AVOID   | conf=62% | Q1 guidance upgrade and obesity trial results contradict short signal.
  NFLX  | AVOID   | conf=38% | Raised guidance and 283% analyst upside directly contradict short.
```

**Day Trade (premarket)**
```
🌅 PREMARKET MOVERS — 08:14 ET

  BBAI  | TRADE | conf=88% | DoD contract win directly catalyzes gap-up; RVOL confirms institutional buying.
         +18.4%  RVOL=8.5x  Vol=420K sh  Float=12M sh  Risk=LOW
         Entry: Wait for first 1-min candle to close green after 9:30 open.

  OCGN  | WATCH | conf=45% | FDA Fast Track is NOT approval — +22% likely overreaction, expect gap fill.
         +12.1%  RVOL=5.2x  Vol=280K sh  Float=8M sh   Risk=MEDIUM
         Entry: Only enter if holds above +15% in first 5 minutes.

  MDJH  | AVOID | conf=15% | No catalyst + 2M float + 25x RVOL = classic pump-and-dump pattern.
         +35.0%  RVOL=25x   Vol=180K sh  Float=2M sh   Risk=HIGH
```

---

## RAG Pipeline

The scanner accumulates its own historical trade data in Supabase PostgreSQL and uses it to ground LLM analysis in empirical evidence rather than general knowledge.

### How It Works

```
Every scan:
  Results → Supabase (scan_results table)

Every day at 10:00 AM ET:
  update_outcomes.py fetches 1-min bars from yfinance
  Computes open_return = (price at 10:00 AM - open price) / open price
  Sets outcome = WIN / LOSS / NEUTRAL based on signal vs actual move
  Writes back to Supabase

Before each LLM analysis:
  get_historical_context() queries catalyst_stats view
  get_relevant_knowledge() retrieves matching rules from knowledge table
  Both are injected into the Claude prompt
```

### What Gets Injected into the LLM Prompt

```
=== OUR HISTORICAL DATA (FDA_APPROVAL, 47 cases) ===
  Win rate:        73%
  Avg open return: +12.4%
  Sample size:     47 trades

=== OUR KNOWLEDGE BASE ===
  - FDA Fast Track is NOT approval. Stocks often reverse 50%+ of premarket gains at open.
  - No news + float < 5M shares + RVOL > 10x = high probability pump and dump. Avoid.
  - Bitcoin miners (MARA, RIOT, CLSK) move together. Use BTC price as leading indicator.
```

This transforms LLM output from generic financial commentary into analysis grounded in the system's own track record.

### Database Schema (Supabase PostgreSQL + pgvector)

```sql
scan_results   -- premarket scan history with outcomes
news           -- news articles with vector embeddings (future semantic search)
knowledge      -- manually curated rules and observations
catalyst_stats -- auto-computed view: win rate and avg return by catalyst type
```

---

## Key Design Decisions

### Regime-Adaptive Factor Selection

Factor effectiveness depends on market environment. Empirical research on this dataset shows:

| Period | Regime | Best Factor | Mean IC |
|--------|--------|-------------|---------|
| 2019–2023 | Trending (low VIX) | momentum_20d | +0.030 |
| 2025–2026 | Volatile (tariff shock) | reversal_20d | +0.037 |

This regime-switching insight was discovered organically through systematic backtesting across 50+ factor variants, not assumed in advance.

### Why Small-Cap for Day Trade?

Small-cap stocks (market cap < $300M, float < 20M shares) are the primary target because the same dollar volume produces a proportionally larger price move. With 5M shares of float, $500K of buying pressure can move a stock 10%+.

### LLM Role: Classifier and Verifier, Not Calculator

In Swing Trade, LLM acts as a **signal negator** — factor signals are forward-looking but news can override them. In Day Trade, LLM acts as a **catalyst verifier** — it distinguishes FDA approval from FDA Fast Track, judges whether price moves are proportional to catalysts, and flags manipulation risk.

Factor computation stays in deterministic Python code. LLM is only applied where language understanding matters.

### RVOL as the Core Day Trade Signal

RVOL (today's volume / expected volume at this time of day) surfaces genuine anomalies across stocks of different sizes, time-adjusted for premarket vs. intraday sessions using 20-day historical baselines.

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
                               ↓
                         database.py
                         Supabase PostgreSQL
                         scan_results / news / knowledge
                               ↑
                         update_outcomes.py
                         Daily outcome backfill (10 AM ET)

┌──────────────────────────────────────────────────────────────┐
│                   React Frontend (Vite)                      │
│     Swing Trade tab              Day Trade tab               │
│     Factor Scan                  Custom Scan (14 params)     │
│     Full Scan (LLM)              Deep Scan (LLM + RAG)       │
└──────────────────────────────────────────────────────────────┘
```

**Deployment:** Backend on Railway, frontend on Vercel, database on Supabase.

---

## Regime Detection

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

---

## Premarket Scanner

**Universe:** 1,071 pre-filtered US small-cap stocks (market cap < $300M, refreshed weekly).

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

**LLM catalyst output fields:**
```
catalyst_type      FDA_APPROVAL / FDA_FAST_TRACK / EARNINGS_BEAT / CONTRACT_WIN / ...
catalyst_strength  STRONG / MODERATE / WEAK / NONE
proportionality    OVER / FAIR / UNDER  (is the price move justified?)
manipulation_risk  HIGH / MEDIUM / LOW
signal             TRADE / WATCH / AVOID
confidence         0–100
reason             one sentence
risk               main risk at open
entry_timing       specific entry suggestion
```

---

## LLM Prompt Engineering

**Day Trade prompt** includes:
- Critical distinction between FDA approval and FDA Fast Track designation
- Proportionality judgment (is +30% justified by the catalyst?)
- Manipulation risk assessment (no news + tiny float + extreme RVOL)
- Specific entry timing advice
- Injected RAG context from historical database

**Confidence calibration:**
```
85–100: Binary catalyst confirmed (FDA approval text, signed acquisition)
70–84 : Strong company-specific catalyst
50–69 : Moderate catalyst, uncertain follow-through
30–49 : Weak catalyst or proportionality mismatch
10–29 : No real catalyst or disproportionate move
 0–9  : Active manipulation signals
```

---

## Project Structure

```
scanner/
├── backend/
│   ├── api.py                     # FastAPI entry point
│   ├── database.py                # Supabase access layer (RAG)
│   ├── swing/
│   │   ├── data.py                # OHLCV + news fetch
│   │   ├── regime.py              # Regime detection
│   │   ├── scanner.py             # Factor computation
│   │   ├── llm_analyst.py         # LLM news analysis
│   │   └── main.py                # Pipeline orchestration
│   └── premarket/
│       ├── premarket_data.py      # Universe, quotes, RVOL
│       ├── premarket_catalyst.py  # LLM catalyst analysis + RAG injection
│       ├── premarket_scanner.py   # Scoring, ranking, history logging
│       ├── update_outcomes.py     # Daily outcome backfill script
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

```bash
cd backend
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export FINNHUB_API_KEY="your-finnhub-key"
export SUPABASE_URL="postgresql://postgres.xxx:password@aws-1-us-west-1.pooler.supabase.com:5432/postgres"
```

```bash
uvicorn api:app --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

**Daily workflow:**
```bash
# At 10:00 AM ET — auto-fill open returns and outcomes
python premarket/update_outcomes.py

# Check cumulative stats
python premarket/update_outcomes.py summary
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/regime` | Current market regime |
| GET | `/api/scan` | Swing factor scan |
| POST | `/api/scan/full` | Swing scan with LLM |
| GET | `/api/premarket/scan` | Premarket scan (14 custom params) |
| POST | `/api/premarket/scan/full` | Premarket scan with LLM + RAG |

---

## Limitations

- Swing scanner universe limited to ~55 S&P 500 stocks
- Premarket scanner uses Finnhub free tier (60 calls/min) — full scan ~18 minutes
- RAG context requires data accumulation; meaningful stats need 30+ cases per catalyst type
- No real-time WebSocket streaming (snapshot-based)

---

## Planned Improvements

- Earnings calendar integration for binary event flagging
- Auto-refresh toggle (every N minutes during premarket)
- Dual-signal overlay: flag stocks in both Swing and Day Trade results simultaneously
- Vector embeddings for news semantic search (pgvector already enabled)
- Polygon.io bulk snapshot API for sub-minute full-market scans
- ICIR-weighted factor combination for Swing Trade

---

## Motivation

Traditional scanners apply fixed rules regardless of market conditions. This project tests the hypothesis that regime-adaptive quantitative signals combined with LLM catalyst analysis — grounded in self-accumulated historical data — produce higher-quality candidates than any single approach alone.

The regime-switching insight is empirical: momentum factors delivered IC > 0.03 during 2019–2023 but turned negative after the April 2025 tariff shock, while reversal factors recovered to IC > 0.037. The RAG layer is designed to compound this advantage over time: as the system accumulates its own trade history, LLM analysis becomes increasingly grounded in the system's specific track record rather than generic financial knowledge.

---

## References

- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

---

## Author

Jamie Ren · B.S. Computer Science & Statistics, University of Toronto · M.S. Information Science, Trine University

*Built as part of quantitative research and AI engineering skill development.*
