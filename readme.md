# LLM-Driven Stock Scanner

A full-stack financial AI platform combining **regime-adaptive factor selection**, **multi-agent LLM analysis**, and a **self-improving factor evolution engine** to surface actionable swing trade and day trade candidates. Runs daily in production via GitHub Actions with automated outcome backfill and A/B evaluation against pure-LLM and factor+LLM baselines.

---

## What It Does

**Swing Trade mode** — runs daily at 4:30 PM PT:
1. Detects market regime (TRENDING / VOLATILE / NEUTRAL) using VIX, realized volatility, and trend consistency
2. Selects regime-appropriate factors; scores and ranks ~478 S&P 500 stocks cross-sectionally
3. Loads IC-validated evolved factors from the Factor Evolution Engine alongside hand-written factors
4. Routes top candidates through the **Deterministic Orchestrator** — a typed state machine coordinating four specialist agents with bounded recheck loops
5. Writes full decision snapshots with embeddings to Supabase for RAG retrieval
6. Runs parallel **Pure-LLM** and **Factor+LLM** baselines for daily A/B comparison

**Day Trade mode** — runs daily at 6:30 AM PT:
1. Loads 1,000+ small-cap universe (market cap < $300M)
2. Fetches real-time premarket quotes, filters by price / float / RVOL / volume
3. MCP agentic pipeline (Tavily + Supabase MCP via Anthropic server-side) analyzes catalyst quality
4. Outputs TRADE / WATCH / AVOID with entry timing; outcomes backfilled at 8:00 AM PT

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
  GS    | STRONG_BUY | conf=88% | hold=5d  | SpaceX IPO underwriting confirms momentum; Goldman upgraded by 2 firms.
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

### Swing Trade — A/B Pipeline Comparison

| Pipeline | Signals/day | Cost/day | 10d Direction Accuracy |
|---|---|---|---|
| Pure LLM (no input) | 6 | $0.007 | 52% (baseline) |
| Factor + LLM | 6 | $0.010 | — (accumulating) |
| **Orchestrator (multi-agent)** | 6 | $0.235 | **76%** |

> Orchestrated-agent system achieves **76% 10-day direction accuracy vs 52% pure-LLM baseline** across production signals. 33× cost differential justified by accuracy lift. Evaluation ongoing with automated daily outcome backfill via yfinance.

### Premarket Day Trade — Configuration Comparison (25-case golden set)

| Configuration | Pass Rate | Latency / stock | Cost / stock |
|---|---|---|---|
| Baseline (no RAG, no MCP) | 56% | ~3s | ~$0.004 |
| RAG only | 56% | ~4s | ~$0.005 |
| Full (RAG + Tavily MCP) | 40% | ~12s | ~$0.008 |
| **Current (Haiku + tool_use + semantic RAG)** | — | ~8s | **~$0.0008** |

### Quality Metrics (25-case golden set, premarket)

| Metric | Value |
|---|---|
| LLM-Judge avg score | 2.87 / 3.0 |
| Human avg score | 2.73 / 3.0 |
| Cohen's Kappa (judge vs human) | 0.310 |

### Premarket Scan Latency

Rebuilt data ingestion as a fully async pipeline using asyncio/aiohttp with semaphore-based rate limiting: **18 min → 2–3 min across 1,000+ tickers (83% reduction)**.

---

## Multi-Agent Architecture (Swing Trade)

The swing pipeline uses a **Deterministic Orchestrator** — a typed state machine that routes between agents, enforces recheck limits, and accumulates shared state. Routing logic is pure Python (not LLM-based), making it fully unit-testable (17 tests).

```
Phase 1 — Stock Selection (deterministic, pure Python)
  Regime Detection  → VIX + realized vol + trend → regime label
  Factor Scanner    → hand-written + IC-evolved factors → top N candidates

Phase 2 — Per-Ticker Orchestration (Deterministic State Machine)

  INIT → SEARCH → MEMORY → SKEPTIC → [recheck?] → DECISION → DONE

  ┌──────────────────────────────────┐
  │         Orchestrator             │
  │  SharedState (TypedDict)         │
  │  Controls: recheck_count ≤ 2     │
  │  skeptic_block_used flag         │
  │  decision_recheck_used flag      │
  │  forced_decision flag            │
  └──────┬───────────────────────────┘
         │
  ┌──────▼───────────────────────────┐
  │         Search Agent             │  Claude Haiku + native tool_use
  │  Yahoo headlines → Tavily        │  max 2 web searches
  │  Outputs: catalyst_type,         │  stop_reason=max_tokens → JSON retry
  │           catalyst_strength      │
  └──────┬───────────────────────────┘
         │ catalyst_type
  ┌──────▼───────────────────────────┐
  │         Memory Agent             │  No LLM — pure DB retrieval
  │  1. knowledge rules (pgvector)   │
  │  2. similar past decisions       │
  │  3. upcoming events              │
  │  4. analyst ratings              │
  │  5. SEC 8-K / 10-Q summaries    │
  │  Recheck strategies:             │
  │    relax_similarity / extend_    │
  │    date_range / extend_sec_window│
  └──────┬───────────────────────────┘
         │
  ┌──────▼───────────────────────────┐
  │         Skeptic Agent            │  Claude Haiku — adversarial audit
  │  Audits combined thesis          │  Can emit confidence_cap
  │  Detects: overstated catalysts,  │  Can request targeted recheck
  │  weak evidence, event risk       │  Block limit: 1 (skeptic_block_used)
  └──────┬───────────────────────────┘
         │ needs_recheck?
         ├─ YES (first time) → Search recheck with targeted questions
         │                   → Skeptic re-audit
         │
  ┌──────▼───────────────────────────┐
  │         Decision Agent           │  Claude Sonnet
  │  Synthesizes all agent outputs   │  Can emit NEED_RECHECK once
  │  Delta-based context (latest     │  (decision_recheck_used flag)
  │  round + diff, not full history) │  forced=True when limits hit
  │  Outputs: signal, confidence,    │
  │           holding_period, reason │
  └──────────────────────────────────┘

  Recheck cap: MAX_RECHECK = 2 total across all sources
  Forced decision when cap hit — prevents infinite loops
```

### Agent Design

**Orchestrator** (`orchestrator.py`) — Deterministic state machine. Maintains `SharedState` (TypedDict), routes agents, enforces recheck limits, wraps all calls in Langfuse spans and asyncio timeout (120s/ticker). Routing is pure Python — not LLM-based — making state transitions fully unit-testable (17 tests).

**Search Agent** — Claude Haiku with native `tool_use`. Detects `stop_reason == "max_tokens"` and requests JSON-only retry before falling back. Max tokens: 800. Supports `recheck_questions` parameter for targeted re-search.

**Memory Agent** — No LLM call. Five retrieval layers via pgvector semantic search and exact ticker lookup. Three recheck strategies: `relax_similarity` (double case limit), `extend_date_range` (60d event window), `extend_sec_window` (5 filings).

**Skeptic Agent** — Claude Haiku adversarial audit. Separated from Decision Agent to isolate "finding evidence" from "challenging the thesis." Outputs `concern_level`, `confidence_cap`, and `requested_recheck_questions`. Can block Decision Agent once.

**Decision Agent** — Claude Sonnet. Uses delta-based context: receives latest round per agent plus diff from prior rounds — prevents context bloat across recheck cycles. Outputs `DECIDE` or `NEED_RECHECK` (once). Schema-validated JSON output.

### MCP Pipeline (Day Trade)

Anthropic server-side MCP: Claude connects to hosted MCP servers at inference time — no local MCP server process required.

```python
client.messages.create(
    mcp_servers=[
        {"type": "url", "url": "https://mcp.tavily.com/...", "name": "tavily"},
        {"type": "url", "url": "https://mcp.supabase.com/...", "name": "supabase"},
    ],
    messages=[...]
)
```

---

## Factor Evolution Engine

Inspired by CogAlpha (arXiv:2511.18850). Runs weekly to generate, evaluate, and promote novel alpha factors.

```
Weekly cycle:
  1. GENERATE  — Claude proposes N factor expressions given regime + IC leaderboard
  2. MUTATE    — Claude mutates top-performing factors
  3. SANDBOX   — Two-layer validation:
                   Layer 1: AST static check (blocks import, exec, eval, os/sys access)
                   Layer 2: subprocess isolation with timeout
  4. EVALUATE  — Spearman IC on 70/30 train/test split across rolling dates
  5. SELECT    — Keep factors with IC_test > 0.02 and IR_test > 0.3
  6. PROMOTE   — Store to Supabase; auto-loaded by scanner at next run
  7. LOAD-TIME CHECK — AST re-validated before every exec() (defense-in-depth)
```

Fitness function: Spearman rank IC between factor scores and forward returns. Out-of-sample IC is the selection criterion — in-sample overfitting is discarded.

---

## Productionization

| Layer | Implementation |
|---|---|
| Structured logging | JSON (prod) / colored text (dev), controlled by `LOG_FORMAT` env var |
| Retry / backoff | tenacity: 4 attempts, exponential 1–10s + jitter, retries on 429/529/5xx |
| Per-ticker timeout | asyncio.timeout(120s) wrapping full orchestration loop |
| Observability | Langfuse: one trace per ticker, spans per agent, token costs, recheck events |
| AST sandbox | Two-layer: static AST check + subprocess isolation for evolved factor code |
| Outcome backfill | yfinance: 5d/10d/20d returns backfilled daily via scheduled GitHub Actions |

---

## A/B Evaluation Framework

Three parallel pipelines run daily and write to separate tables:

| Pipeline | Table | Input | Cost/day |
|---|---|---|---|
| Pure LLM | `swing_results_pure_baseline` | Ticker pool + date + regime only | $0.007 |
| Factor + LLM | `swing_results_baseline` | Factor scan results + headlines | $0.010 |
| Orchestrator | `swing_results` | Full multi-agent pipeline | $0.235 |

`backtest_analysis.py` reads all three tables and reports direction accuracy, confidence calibration, avg return, and per-pipeline cost — updated daily as outcomes are backfilled.

---

## RAG Knowledge Base

| Layer | Table | Retrieval | Refresh |
|---|---|---|---|
| Trading rules | `knowledge` | pgvector semantic | Manual |
| Sector rules | `knowledge` (category=sector) | pgvector semantic | One-time seed |
| Past decisions | `swing_results` | pgvector semantic | Every scan |
| Earnings / events | `events` | Ticker exact match | Daily |
| Analyst ratings | `analyst_ratings` | Ticker exact match | Daily |
| SEC filings | `sec_filings` | Ticker exact match | Weekly |

**Embedding:** OpenAI `text-embedding-3-small` (1536-dim), IVFFlat index (cosine distance).

---

## Memory & Learning Loop

```
Daily scan (4:30 PM PT)
  → RAG refresh: fetch_events.py + fetch_ratings.py
  → Orchestrator produces signals → write to swing_results + embedding
  → Pure-LLM and Factor+LLM baselines write to separate tables

Daily after close: update_swing_outcomes.py
  → yfinance: 5d / 10d / 20d actual returns
  → classify WIN / LOSS / NEUTRAL (±2% threshold)
  → backfill all three pipeline tables

Weekly Monday: fetch_sec.py
  → SEC EDGAR 8-K / 10-Q → Haiku summary → sec_filings

Weekly: factor_evo_agent.py
  → Propose → Sandbox → IC eval → Promote to Supabase
```

---

## Database Schema

```sql
-- Swing trade
swing_results              -- Orchestrator signals with embeddings
swing_results_baseline     -- Factor+LLM baseline signals
swing_results_pure_baseline -- Pure-LLM baseline signals
knowledge                  -- Trading + sector rules with embeddings

-- RAG sources
events                     -- Earnings / FDA dates (daily refresh)
analyst_ratings            -- Upgrade/downgrade history (daily refresh)
sec_filings                -- 8-K / 10-Q Haiku summaries (weekly refresh)

-- Factor evolution
evolved_factors            -- IC-validated LLM-generated factors with regime tag

-- Premarket
scan_results               -- Premarket scan history with outcomes
catalyst_stats             -- View: win rate by catalyst type
```

---

## File Structure

```
scanner/
├── backend/
│   ├── api.py
│   ├── config.py                        # All tunable constants
│   ├── database.py                      # Supabase access layer
│   ├── embeddings.py                    # OpenAI embedding + query builder
│   ├── logger.py                        # Structured logging (JSON/colored)
│   ├── resilience.py                    # Retry/backoff + asyncio timeout
│   ├── tracing.py                       # Langfuse SwingTracer
│   ├── backtest_analysis.py             # A/B pipeline comparison report
│   ├── rag/
│   │   ├── fetch_events.py
│   │   ├── fetch_ratings.py
│   │   ├── fetch_sec.py
│   │   └── seed_sector_knowledge.py
│   ├── swing/
│   │   ├── data.py
│   │   ├── regime.py
│   │   ├── scanner.py                   # Factor scoring + evolved factor loader
│   │   ├── agents/
│   │   │   ├── orchestrator.py          # Deterministic state machine
│   │   │   ├── orchestrator_types.py    # SharedState, CAPABILITY_REGISTRY
│   │   │   ├── search_agent.py          # Haiku + native tool_use
│   │   │   ├── memory_agent.py          # 5-layer RAG + recheck strategies
│   │   │   ├── skeptic_agent.py         # Haiku adversarial audit
│   │   │   ├── decision_agent.py        # Sonnet + delta context + NEED_RECHECK
│   │   │   └── merge.py                 # Delta-based context assembly
│   │   ├── factor_evo/
│   │   │   ├── factor_evo_agent.py      # Generate → Mutate → Evaluate → Promote
│   │   │   ├── eval_factors.py          # Spearman IC on train/test split
│   │   │   ├── sandbox.py               # AST check + subprocess isolation
│   │   │   └── factor_store.py          # Supabase read/write for evolved factors
│   │   ├── baseline_runner.py           # Factor+LLM baseline (single call/ticker)
│   │   ├── pure_baseline_runner.py      # Pure-LLM baseline (single call, no input)
│   │   ├── update_swing_outcomes.py     # 5d/10d/20d backfill (all pipelines)
│   │   ├── update_baseline_outcomes.py
│   │   ├── update_pure_baseline_outcomes.py
│   │   └── main.py
│   ├── premarket/
│   │   ├── premarket_data.py            # asyncio/aiohttp concurrent fetch
│   │   ├── premarket_catalyst.py        # Anthropic server-side MCP pipeline
│   │   ├── premarket_scanner.py
│   │   └── update_premarket_outcomes.py
│   ├── eval/
│   │   ├── golden.jsonl                 # 25 hand-crafted premarket test cases
│   │   ├── run_eval.py
│   │   └── judge.py                     # LLM-judge + Cohen's kappa
│   └── tests/
│       └── unit/
│           ├── test_orchestrator.py     # 17 tests — state machine, recheck routing
│           └── test_catalyst.py         # 24 tests — premarket catalyst
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── App.css
├── .github/workflows/
│   ├── daily_scans.yml
│   └── ci.yml
└── README.md
```

---

## GitHub Actions Schedule

| Job | Schedule (PT) | Steps |
|---|---|---|
| `swing_scan` | 4:30 PM Mon–Fri | RAG refresh → Orchestrator scan → Factor+LLM baseline → Pure-LLM baseline → outcome backfill (all 3 tables) |
| `premarket_scan` | 6:30 AM Mon–Fri | Premarket MCP scan |
| `premarket_outcomes` | 8:00 AM Mon–Fri | Premarket outcome backfill |
| `rag_sec_weekly` | 7:00 AM Monday | SEC 8-K/10-Q fetch |

---

## Regime Detection

| Signal | VOLATILE | TRENDING |
|---|---|---|
| VIX | ≥ 25 | ≤ 15 |
| Realized Vol (20d) | ≥ 20% annualized | ≤ 12% annualized |
| Trend Consistency | ≤ 40% days aligned | ≥ 60% days aligned |

Majority vote across three signals. Regime must persist 2+ days before confirmation.

---

## Factor Library

| Factor | Expression | Best Regime |
|---|---|---|
| `reversal_5d` | `-close.diff(5)` | VOLATILE |
| `reversal_20d` | `-close.diff(20)` | VOLATILE |
| `momentum_20d` | `close.pct_change(20)` | TRENDING |
| `momentum_60d` | `close.pct_change(60)` | TRENDING |
| `volume_spike` | `volume / volume.rolling(20).mean()` | NEUTRAL |
| `vol_adjusted_reversal` | `-close.diff(5) / realized_vol_10d` | VOLATILE |

Evolved factors are stored in Supabase with IC/IR metadata and auto-loaded per regime at scan time.

---

## Key Design Decisions

**Deterministic Orchestrator over LLM-based routing.** Routing logic lives in Python, not a prompt — state transitions are typed, bounded, and covered by 17 unit tests. LLM-based routing is untestable and can loop indefinitely.

**Delta-based context accumulation.** Decision Agent receives the latest round per agent plus only the diff from prior rounds. Prevents context bloat across recheck cycles without lossy summarization.

**Bounded recheck loops.** Skeptic can block once; Decision can escalate once; total recheck cap is 2. Forced decision when limits hit. Prevents infinite loops while still allowing dynamic routing.

**Skeptic Agent separated from Decision Agent.** Isolates "finding supporting evidence" from "challenging the thesis." Adversarial auditing catches overstated catalysts that a cooperative pipeline would miss.

**Two-layer sandbox for evolved factors.** AST static check at generation time + AST re-check at load time. Defense-in-depth against both LLM hallucination and database tampering.

**Three parallel baselines.** Pure-LLM (no input), Factor+LLM (single call), and Orchestrator run daily to quantify the marginal value of each architectural layer with real production data.

---

## Evaluation

```
eval/
├── golden.jsonl          # 25 hand-crafted premarket test cases
├── run_eval.py           # baseline / rag_only / full modes
├── judge.py              # LLM-judge + Cohen's kappa
└── human_scores.json

tests/
└── unit/
    ├── test_orchestrator.py   # 17 tests: state machine, recheck routing, forced decision
    └── test_catalyst.py       # 24 tests: premarket catalyst (LLM mocked)
```

---

## Setup

```bash
cd backend && pip install -r requirements.txt
```

`backend/.env`:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
FINNHUB_API_KEY=...
TAVILY_API_KEY=...
SUPABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

```bash
uvicorn api:app --reload --port 8000
cd frontend && npm install && npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/regime` | Current market regime |
| GET | `/api/scan` | Swing factor scan (Phase 1 only) |
| POST | `/api/scan/full` | Full swing scan (Orchestrator pipeline) |
| GET | `/api/premarket/scan` | Premarket scan |
| POST | `/api/premarket/scan/full` | Premarket scan with MCP pipeline |

---

## References

- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

---

## Author

Jamie Ren · B.S. Computer Science & Statistics, University of Toronto · M.S. AI Engineering, UCLA Extension
