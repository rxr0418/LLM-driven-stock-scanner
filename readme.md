# LLM-Driven Stock Scanner

A full-stack stock scanner combining **regime-adaptive factor selection**, **premarket momentum detection**, and **LLM-powered catalyst analysis** to surface actionable trading candidates. The swing trade pipeline uses a **multi-agent architecture** (Search Agent → Memory Agent → Decision Agent) grounded in a five-layer RAG knowledge base that self-accumulates from real trading outcomes.

---

## What It Does

**Swing Trade mode** — runs daily at 4:30 PM PT via GitHub Actions:
1. Detects the current market regime (TRENDING / VOLATILE / NEUTRAL) using VIX, realized volatility, and trend consistency
2. Selects and weights factors based on regime — momentum in trending markets, reversal in volatile markets
3. Scores and ranks all S&P 500 stocks (~478) cross-sectionally, selects top N candidates per side
4. For each candidate, runs a **sequential** two-agent pipeline:
   - **Search Agent** reads Yahoo Finance headlines, then calls Tavily if needed (Claude native tool_use, max 2 searches). Outputs `catalyst_type` used by next agent.
   - **Memory Agent** uses `catalyst_type` to build a semantic query, then retrieves across five RAG layers (knowledge rules, past decisions, upcoming events, analyst ratings, SEC filings). No LLM call — pure DB retrieval.
   - **Decision Agent** synthesizes all context via explicit Thought → Action reasoning and outputs signal, confidence, reason, and holding period. Schema-validated with up to 2 self-correction retries.
5. Writes a full decision snapshot with embedding to Supabase for future RAG retrieval
6. Outputs a ranked watchlist; `NO_POSITION` for low-conviction or high-risk candidates

**Day Trade mode** — runs daily at 6:30 AM PT via GitHub Actions:
1. Loads a pre-filtered universe of 1,000+ small-cap stocks (market cap < $300M)
2. Fetches real-time premarket quotes from Finnhub
3. Filters by price, float, premarket change %, volume, and RVOL
4. MCP agentic pipeline analyzes catalyst quality, outputs TRADE / WATCH / AVOID with entry timing
5. Scan results logged to Supabase; outcomes backfilled at 8:00 AM PT

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
  GS    | STRONG_BUY | conf=88% | hold=5d  | SpaceX IPO underwriting confirms momentum; Goldman upgraded by 2 firms this week.
  ADBE  | BUY        | conf=72% | hold=5d  | Agentic AI expansion supports momentum; Figma competition warrants caution.
  BA    | NO_POSITION| conf= 0% | skip     | China deal disappointed; news contradicts factor signal.

SHORT CANDIDATES
  ULTA  | SHORT      | conf=65% | hold=3d  | 23% price decline and weakest factor score; no bullish catalyst.
  CBOE  | NO_POSITION| conf= 0% | skip     | SpaceX options listing contradicts short signal.
```

**Day Trade (premarket)**
```
PREMARKET MOVERS — 08:14 ET

  BBAI  | TRADE | conf=88% | DoD contract win directly catalyzes gap-up; RVOL confirms institutional buying.
         +18.4%  RVOL=8.5x  Vol=420K sh  Float=12M sh  Risk=LOW
         Entry: Wait for first 1-min candle to close green after 9:30 open.

  OCGN  | WATCH | conf=45% | FDA Fast Track is NOT approval — +22% likely overreaction, expect gap fill.
         +12.1%  RVOL=5.2x  Vol=280K sh  Float=8M sh   Risk=MEDIUM

  MDJH  | AVOID | conf=15% | No catalyst + 2M float + 25x RVOL = classic pump-and-dump pattern.
         +35.0%  RVOL=25x   Vol=180K sh  Float=2M sh   Risk=HIGH
```

---

## Benchmark

Metrics are measured on a **25-case golden set** (premarket day trade eval). Swing trade PnL and hit rate require 4–6 weeks of production accumulation and are marked TBD.

### Configuration Comparison

| Configuration | Pass Rate | Latency / stock | Cost / stock | Notes |
|---|---|---|---|---|
| Baseline (no RAG, no MCP) | 56% | ~3s | ~$0.004 | Plain prompt, Sonnet |
| RAG only | 56% | ~4s | ~$0.005 | catalyst_stats empty at eval time; no lift |
| Full (RAG + Tavily MCP) | 40% | ~12s | ~$0.008 | Regression: richer context caused overconfidence on sector moves |
| **Current (Haiku + tool_use + semantic RAG)** | — | ~8s | **~$0.0008** | Haiku switch; eval re-run pending |

> Cost/stock computed as: Search Agent (Haiku) + Decision Agent (Sonnet) + 2× OpenAI embeddings per scan.

### Quality Metrics (25-case golden set, premarket)

| Metric | Value | Method |
|---|---|---|
| Pass rate (baseline) | 56% (14/25) | expected_facts + forbidden_facts checks |
| RAGAS Faithfulness | 0.088 | Low — `catalyst_stats` empty at eval time; expected to improve with data |
| LLM-Judge avg score | 2.87 / 3.0 | Claude Sonnet strict rubric |
| Human avg score | 2.73 / 3.0 | Practical trading utility |
| Cohen's Kappa (judge vs human) | 0.310 | Moderate agreement; LLM judge has positivity bias |

### Swing Trade Outcome Metrics (production, accumulating)

| Metric | Value | Status |
|---|---|---|
| Overall hit rate (WIN / total signals) | — | TBD — need 30+ outcomes |
| Hit rate by regime: TRENDING | — | TBD |
| Hit rate by regime: VOLATILE | — | TBD |
| Avg return per signal (5d hold) | — | TBD |
| Avg return per signal (10d hold) | — | TBD |
| Max drawdown (worst signal sequence) | — | TBD |
| NO_POSITION rate (filter effectiveness) | — | TBD |
| Decision Agent failure rate | ~0% | Schema retry loop in production |

### Cost Breakdown (per 10-stock full swing scan)

| Component | Model | Est. cost |
|---|---|---|
| Search Agent × 10 | Haiku (cached system) | ~$0.003 |
| Decision Agent × 10 | Sonnet (cached system) | ~$0.025 |
| OpenAI embeddings × 10 | text-embedding-3-small | ~$0.001 |
| Tavily searches (avg 1.2/stock) | — | ~$0.006 |
| **Total** | | **~$0.035** |

> Previous cost before Haiku switch and prompt caching: ~$1.50 per 10-stock scan (43× reduction).

---

## Multi-Agent Architecture (Swing Trade)

```
Phase 1 — Stock Selection (deterministic, pure Python)
  Regime Worker   → VIX + realized vol + trend → regime label + factor weights
  Factor Worker   → cross-sectional IC scoring → top N candidates per side

Phase 2 — Per-Ticker Analysis (parallel across candidates, sequential within)
  ┌──────────────────────────────┐
  │       Search Agent           │
  │  Yahoo headlines             │
  │  → native tool_use           │
  │  → Tavily if needed (max 2)  │
  │  → catalyst_type output      │
  └──────────────┬───────────────┘
                 │ catalyst_type
  ┌──────────────▼───────────────┐
  │       Memory Agent           │  ← Five RAG layers (no LLM call)
  │  1. knowledge rules          │
  │  2. similar past decisions   │
  │  3. upcoming events          │
  │  4. analyst ratings          │
  │  5. SEC 8-K / 10-Q summaries │
  └──────────────┬───────────────┘
                 │
            merge() — pure Python, no LLM
                 │
  ┌──────────────▼───────────────┐
  │      Decision Agent          │
  │  Thought → Action → JSON     │
  │  schema validate + retry     │
  │  (Sonnet, cached system)     │
  └──────────────┬───────────────┘
                 │
  ┌──────────────▼───────────────┐
  │  Supabase write              │
  │  decision snapshot           │
  │  + embedding (pgvector)      │
  │  + news evidence             │
  └──────────────────────────────┘
```

### Agent Design

**Search Agent** — Claude Haiku with native `tool_use`. Registered tool: `web_search` (Tavily, max 2 calls). Agent loop checks `stop_reason`: `"tool_use"` → execute and return `tool_result`; `"end_turn"` → extract JSON. Prompt-cached system prompt. Outputs `catalyst_type` used by Memory Agent for semantic query construction.

**Memory Agent** — No LLM call. Runs sequentially after Search Agent to receive `catalyst_type`. Queries five layers:
1. `knowledge` table — semantic pgvector search (trading rules + sector rules)
2. `swing_results` table — semantic pgvector search (similar past decisions with outcomes)
3. `events` table — exact ticker lookup (earnings dates, FDA windows)
4. `analyst_ratings` table — exact ticker lookup, last 30 days
5. `sec_filings` table — exact ticker lookup, last 2 filings (Haiku-summarized 8-K/10-Q)

Returns `event_risk_flag` if a binary event is within 5 days.

**merge()** — Pure Python. Assembles all context into a single dict. Applies regime-based `max_candidates` (6 / 8 / 10). No LLM, no DB calls.

**Decision Agent** — Claude Sonnet with prompt-cached system prompt. ReAct format (Thought → Action → JSON). `_validate()` checks signal enum, confidence range, alignment enum, holding period type, reason presence. Retries up to `MAX_DECISION_RETRIES=2` times on failure, feeding error list back for self-correction.

### MCP Pipeline (Day Trade)

The premarket deep scan uses Anthropic's beta MCP API, giving Claude autonomous tool access:

```python
client.beta.messages.create(
    mcp_servers=[
        {"type": "url", "url": "https://mcp.tavily.com/...", "name": "tavily"},
        {"type": "url", "url": "https://mcp.supabase.com/...", "name": "supabase"},
    ],
    messages=[...]
)
```

Claude decides at inference time whether to call `tavily_search` or query `catalyst_stats` directly. All tool call sequences traced in Langfuse.

---

## RAG Knowledge Base

Five retrieval layers, all populated automatically or via scheduled scripts:

| Layer | Table | Retrieval | Refresh |
|---|---|---|---|
| Trading rules | `knowledge` | pgvector semantic | Manual / `add_knowledge()` |
| Sector rules | `knowledge` (category=sector) | pgvector semantic | `rag/seed_sector_knowledge.py` (one-time) |
| Past decisions | `swing_results` | pgvector semantic | Every scan (auto) |
| Earnings / events | `events` | Ticker exact match | Daily pre-scan (GitHub Actions) |
| Analyst ratings | `analyst_ratings` | Ticker exact match | Daily pre-scan (GitHub Actions) |
| SEC filings | `sec_filings` | Ticker exact match | Weekly Monday (GitHub Actions) |

**Embedding:** OpenAI `text-embedding-3-small` (1536-dim). Vectors stored as `vector(1536)` in Supabase with IVFFlat index (`lists=100`, cosine distance).

**Query construction** (`embeddings.build_knowledge_query`):
```
"TRENDING market regime | BUY signal | ticker AAPL | catalyst: CONTRACT_WIN"
```

**Context limits** (all tunable in `config.py`):
```python
MAX_KNOWLEDGE_RULES  = 4
MAX_SIMILAR_CASES    = 3
MAX_ANALYST_RATINGS  = 3
MAX_SEC_FILINGS      = 2
```

---

## Memory & Learning Loop

```
Daily scan (4:30 PM PT)
  → RAG refresh: fetch_events.py + fetch_ratings.py run first
  → Decision Agent produces signal
  → write_decision_snapshot() stores decision + embedding in swing_results

Daily at close: update_swing_outcomes.py
  → fetch actual price at 5d / 10d / 20d via yfinance
  → compute return, classify WIN / LOSS / NEUTRAL
  → backfill swing_results matched by signal_id

Weekly Monday: fetch_sec.py
  → fetch recent 8-K / 10-Q from SEC EDGAR
  → summarize with Claude Haiku → store + embedding in sec_filings

Next scan
  → Memory Agent semantic search finds similar past decisions
  → win rates and avg returns reflect real accumulated outcomes
  → system improves as data accumulates (meaningful signal at ~50 cases)
```

---

## Database Schema

```sql
-- Core swing trade tables
swing_results      -- per-signal decisions: signal_id, ticker, signal, confidence,
                   -- regime, factors_used, search_summary (JSONB), memory_context (JSONB),
                   -- react_trace, price_at_scan, actual_return, outcome, embedding vector(1536)
swing_news         -- news sources used per signal (evidence tracing)
knowledge          -- trading rules + sector rules with embeddings

-- RAG data sources
events             -- upcoming earnings / FDA dates by ticker (refreshed daily)
analyst_ratings    -- upgrade/downgrade history with embeddings (refreshed daily)
sec_filings        -- 8-K / 10-Q Haiku summaries with embeddings (refreshed weekly)

-- Premarket
scan_results       -- premarket scan history with outcomes
catalyst_stats     -- view: win rate by catalyst type
news               -- news articles
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI (api.py)                          │
└──────┬──────────────────────────────────────┬────────────────────┘
       │                                      │
       ▼                                      ▼
  swing/                                premarket/
  ──────────────────────────            ──────────────────────────
  data.py          OHLCV + news         premarket_data.py
  regime.py        Regime detect        premarket_catalyst.py
  scanner.py       Factor scores        premarket_scanner.py
  agents/                               update_premarket_outcomes.py
  ├ search_agent.py  tool_use + Haiku
  ├ memory_agent.py  5-layer RAG
  ├ merge.py         Context assembly
  └ decision_agent.py  Sonnet + retry
  main.py          Orchestration
  update_swing_outcomes.py

  rag/
  ├ fetch_events.py       yfinance earnings dates
  ├ fetch_ratings.py      yfinance analyst ratings
  ├ fetch_sec.py          EDGAR 8-K/10-Q + Haiku summary
  └ seed_sector_knowledge.py  18 sector rules (one-time)

  config.py      Central constants (models, limits, thresholds)
  database.py    Supabase access layer
  embeddings.py  OpenAI text-embedding-3-small
```

```
scanner/
├── backend/
│   ├── api.py
│   ├── config.py                     # All tunable constants
│   ├── database.py                   # Read + write for all tables
│   ├── embeddings.py                 # OpenAI embedding + query builder
│   ├── rag/
│   │   ├── fetch_events.py           # Earnings / event calendar
│   │   ├── fetch_ratings.py          # Analyst upgrades / downgrades
│   │   ├── fetch_sec.py              # SEC 8-K / 10-Q via EDGAR + Haiku
│   │   └── seed_sector_knowledge.py  # 18 sector-level rules (one-time)
│   ├── swing/
│   │   ├── data.py
│   │   ├── regime.py
│   │   ├── scanner.py
│   │   ├── agents/
│   │   │   ├── search_agent.py       # Haiku + native tool_use
│   │   │   ├── memory_agent.py       # 5-layer RAG (no LLM)
│   │   │   ├── merge.py              # Pure Python context assembly
│   │   │   └── decision_agent.py     # Sonnet + schema validation + retry
│   │   ├── main.py                   # Async orchestration
│   │   └── update_swing_outcomes.py  # 5d/10d/20d outcome backfill
│   └── premarket/
│       ├── premarket_data.py
│       ├── premarket_catalyst.py     # MCP agentic pipeline
│       ├── premarket_scanner.py
│       └── update_premarket_outcomes.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── App.css
├── .github/workflows/
│   ├── daily_scans.yml               # Swing scan + RAG refresh + outcome backfill
│   └── ci.yml
└── README.md
```

**Deployment:** Backend on Railway, frontend on Vercel, database on Supabase.

---

## Regime Detection

| Signal | VOLATILE | TRENDING |
|--------|----------|----------|
| VIX | ≥ 25 | ≤ 15 |
| Realized Vol (20d) | ≥ 20% annualized | ≤ 12% annualized |
| Trend Consistency (20d) | ≤ 40% days aligned | ≥ 60% days aligned |

Stability filter: a new regime must persist 2+ consecutive days before confirmation. At regime borders, LLM judgment resolves ambiguity using recent macro context.

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

Empirical IC on this dataset:

| Period | Regime | Best Factor | Mean IC |
|--------|--------|-------------|---------|
| 2019–2023 | Trending (low VIX) | momentum_20d | +0.030 |
| 2025–2026 | Volatile (tariff shock) | reversal_20d | +0.037 |

---

## Key Design Decisions

**Sequential vs parallel agent execution.** Search Agent and Memory Agent run sequentially (not in parallel) because Memory Agent needs `catalyst_type` from Search Agent to build a semantically precise RAG query. Parallelism is preserved across tickers.

**Memory Agent has no LLM call.** All five retrieval layers are pure DB queries. This keeps Memory Agent fast (<200ms), deterministic, and free. The LLM budget is concentrated in Decision Agent where reasoning actually matters.

**Native tool_use over hand-parsed ReAct.** Search Agent uses Claude's `tools=[]` API. This eliminates fragile regex parsing, gives structured tool inputs, and lets the model decide tool call count naturally within the limit.

**Prompt caching on all agents.** System prompts for Search Agent (Haiku) and Decision Agent (Sonnet) use `cache_control: ephemeral`. Since system prompts are identical across tickers in a single scan, the cache hit rate is ~100% after the first call.

**Haiku for Search Agent.** Catalyst classification — identifying whether a headline is an earnings beat, FDA approval, or contract win — is a structured labeling task. Haiku costs ~20× less than Sonnet and is sufficient for this classification quality.

**NO_POSITION as a signal.** The Decision Agent is designed to pass when the edge is unclear. `NO_POSITION` is triggered when: catalyst is absent and news is neutral/contradictory; a binary event (earnings, FDA) is within 5 days; regime is VOLATILE and factor score is weak; or schema validation fails all retries.

---

## Premarket Scanner

**Universe:** 1,071 pre-filtered US small-cap stocks (market cap < $300M).

**Default filter parameters:**

| Parameter | Default | Unit |
|-----------|---------|------|
| Price range | $1 – $20 | USD |
| Market cap | ≤ 300 | M$ |
| Float | ≤ 20,000 | K sh |
| Premarket change | 4% – 40% | % |
| Premarket volume | ≥ 200 | K sh |
| Premarket amount | ≥ 1,000 | K$ |
| RVOL | ≥ 2x | x |

**LLM output fields:**
```
catalyst_type      FDA_APPROVAL / FDA_FAST_TRACK / EARNINGS_BEAT / CONTRACT_WIN / ...
catalyst_strength  STRONG / MODERATE / WEAK / NONE
proportionality    OVER / FAIR / UNDER
manipulation_risk  HIGH / MEDIUM / LOW
signal             TRADE / WATCH / AVOID
confidence         0–100
reason             one sentence
entry_timing       specific entry suggestion
```

---

## GitHub Actions Schedule

| Job | Schedule (PT) | Steps |
|---|---|---|
| `swing_scan` | 4:30 PM Mon–Fri | RAG refresh (events + ratings) → swing scan → outcome backfill |
| `premarket_scan` | 6:30 AM Mon–Fri | Premarket scan |
| `premarket_outcomes` | 8:00 AM Mon–Fri | Premarket outcome backfill |
| `rag_sec_weekly` | 7:00 AM Monday | SEC 8-K/10-Q fetch for full universe |

---

## Evaluation

```
eval/
├── golden.jsonl          # 25 hand-crafted test cases (premarket)
├── run_eval.py           # Eval runner: baseline / rag_only / full modes
├── judge.py              # LLM-judge + Cohen's kappa
├── human_scores.json     # Human scores for kappa computation
└── results/              # Saved eval results (JSON)

tests/
└── unit/
    └── test_catalyst.py  # 24 pytest unit tests (LLM mocked)
```

**Golden set coverage:** FDA approval vs Fast Track, earnings beat/miss, pump-and-dump, short squeeze, sector moves, M&A, dilution, analyst upgrades, ambiguous catalyst wording.

**Known evaluation gaps:**
- Swing trade golden set does not yet exist; pass rate is undefined for swing pipeline
- `catalyst_stats` was empty at eval time, so RAG-only vs baseline showed no lift
- LLM-judge has positivity bias (scored 13/15 cases at max); cross-model judging recommended

---

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
FINNHUB_API_KEY=...
TAVILY_API_KEY=...
SUPABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres
```

```bash
uvicorn api:app --reload --port 8000
cd frontend && npm install && npm run dev
```

**First-time RAG setup (run once):**
```bash
# Supabase SQL
ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE swing_results ADD COLUMN IF NOT EXISTS embedding vector(1536);
CREATE TABLE IF NOT EXISTS events (...);        -- see database.py
CREATE TABLE IF NOT EXISTS analyst_ratings (...);
CREATE TABLE IF NOT EXISTS sec_filings (...);

# Seed sector knowledge (18 rules)
python backend/rag/seed_sector_knowledge.py

# Initial data fetch
python backend/rag/fetch_events.py
python backend/rag/fetch_ratings.py
python backend/rag/fetch_sec.py --limit 2
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/regime` | Current market regime |
| GET | `/api/scan` | Swing factor scan (Phase 1 only) |
| POST | `/api/scan/full` | Swing full scan (Phase 1 + multi-agent) |
| GET | `/api/premarket/scan` | Premarket scan (14 custom params) |
| POST | `/api/premarket/scan/full` | Premarket scan with MCP pipeline |

---

## Limitations

- RAG semantic search (swing_results) requires data accumulation; meaningful signal at ~50 cases with outcomes
- Swing trade hit rate and PnL are not yet measurable — insufficient production data
- VOLATILE regime signal quality is lower; mean-reversion needs catalyst confirmation
- No real-time WebSocket streaming (snapshot-based)
- Premarket scanner uses Finnhub free tier (60 calls/min)
- SEC filing fetch (~0.15s/request) makes full-universe runs slow; batched weekly

---

## Planned Improvements

- **Factor Evo Agent** — LLM-in-the-loop factor evolution (CogAlpha-inspired): weekly IC evaluation → propose new factor expressions → backtest → selection. Fitness function: Spearman rank IC between factor scores and 5d forward returns
- **Swing trade golden set** — 25+ hand-labeled swing decisions for eval parity with premarket pipeline
- **Eval re-run** — re-benchmark all three configurations after Haiku switch and semantic RAG
- **Polygon.io bulk snapshot** — replace Finnhub free tier for faster premarket scanning

---

## References

- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

---

## Author

Jamie Ren · B.S. Computer Science & Statistics, University of Toronto · M.S. AI Engineering, UCLA Extension

*Built as a capstone project in AI engineering, combining quantitative finance and LLM systems.*
