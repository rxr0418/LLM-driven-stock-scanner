# LLM-Driven Stock Scanner

A full-stack stock scanner combining **regime-adaptive factor selection**, **premarket momentum detection**, and **LLM-powered catalyst analysis** to surface actionable trading candidates. The swing trade pipeline uses a **multi-agent architecture** (Search Agent + Memory Agent + Decision Agent with ReAct reasoning) grounded in self-accumulated historical trade data.

---

## What It Does

**Swing Trade mode** — runs daily before market open:
1. Detects the current market regime (TRENDING / VOLATILE / NEUTRAL) using VIX, realized volatility, and trend consistency
2. Selects and weights factors based on regime — momentum in trending markets, reversal in volatile markets
3. Scores and ranks all S&P 500 stocks (~478) cross-sectionally, selects top N candidates per side
4. For each candidate, runs a three-agent pipeline in parallel:
   - **Search Agent** fetches Yahoo Finance headlines, assesses quality, and calls Tavily if more context is needed (ReAct loop, max 3 searches)
   - **Memory Agent** queries historical win rates from Supabase and retrieves relevant trading rules (ReAct loop with fallback query strategy)
   - **Decision Agent** synthesizes all context via explicit Thought → Action reasoning and outputs signal, confidence, reason, and holding period
5. Writes a full decision snapshot (catalyst summary, memory context, ReAct trace) to Supabase for future learning
6. Outputs a ranked watchlist with `NO_POSITION` for low-conviction or high-risk candidates

**Day Trade mode** — runs during premarket (4:00–9:30 AM ET):
1. Loads a pre-filtered universe of 1,000+ small-cap stocks (market cap < $300M)
2. Fetches real-time premarket quotes from Finnhub for each ticker
3. Filters by price, float, premarket change %, volume, dollar amount, and RVOL
4. LLM deep scan (MCP agentic pipeline) analyzes catalyst quality and outputs TRADE / WATCH / AVOID signals with entry timing
5. Scan results are logged to Supabase for historical analysis and RAG

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
  GS    | STRONG_BUY | conf=88% | hold=5d | SpaceX IPO underwriting confirms momentum; 53% win rate across 38 cases.
  ADBE  | BUY        | conf=72% | hold=5d | Agentic AI expansion supports momentum; Figma competition warrants caution.
  BA    | NO_POSITION| conf= 0% | skip    | China deal disappointed; news contradicts factor signal.

SHORT CANDIDATES
  ULTA  | SHORT      | conf=65% | hold=3d | 23% price decline and weakest factor score; no bullish catalyst.
  CBOE  | NO_POSITION| conf= 0% | skip    | SpaceX options listing contradicts short signal.
```

**Day Trade (premarket)**
```
🌅 PREMARKET MOVERS — 08:14 ET

  BBAI  | TRADE | conf=88% | DoD contract win directly catalyzes gap-up; RVOL confirms institutional buying.
         +18.4%  RVOL=8.5x  Vol=420K sh  Float=12M sh  Risk=LOW
         Entry: Wait for first 1-min candle to close green after 9:30 open.

  OCGN  | WATCH | conf=45% | FDA Fast Track is NOT approval — +22% likely overreaction, expect gap fill.
         +12.1%  RVOL=5.2x  Vol=280K sh  Float=8M sh   Risk=MEDIUM

  MDJH  | AVOID | conf=15% | No catalyst + 2M float + 25x RVOL = classic pump-and-dump pattern.
         +35.0%  RVOL=25x   Vol=180K sh  Float=2M sh   Risk=HIGH
```

---

## Multi-Agent Architecture (Swing Trade)

The swing trade pipeline uses a two-phase design:

```
Phase 1 — Stock Selection (deterministic)
  Regime Worker   → VIX + realized vol + LLM border judgment → regime label + factor weights
  Factor Worker   → cross-sectional IC scoring → top N candidates per side

Phase 2 — Per-Ticker Analysis (parallel across candidates)
  ┌─────────────────────────┐    ┌─────────────────────────────────────┐
  │     Search Agent        │    │          Memory Agent               │
  │  Finnhub headline       │    │  catalyst_stats SQL (with fallback) │
  │  → assess quality       │    │  knowledge rules                    │
  │  → Tavily if needed     │    │  cross-session persistent memory    │
  │  (ReAct, max 3 calls)   │    │  (ReAct, tiered query strategy)     │
  └──────────┬──────────────┘    └──────────────┬──────────────────────┘
             │                                  │
             └──────────── merge() ─────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │    Decision Agent        │
                    │  Thought → Action →      │
                    │  Final signal            │
                    │  (ReAct, single pass)    │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  Supabase + Langfuse     │
                    │  decision snapshot       │
                    │  + news evidence         │
                    └─────────────────────────┘
```

### Agent Design Principles

**Search Agent** owns all information gathering for a ticker. It reads Yahoo Finance headlines first, then decides whether to call Tavily for additional context. Before classifying `catalyst_type`, it explicitly lists all headlines seen and identifies the highest-impact event — preventing the first headline from dominating the classification.

**Memory Agent** uses a tiered query strategy to avoid cross-market error transfer:
```
1. (regime, signal)        → most specific, use if sample_size ≥ 10
2. (signal only)           → broader, use if step 1 fails
3. knowledge rules only    → fallback when no reliable stats exist
```

**Decision Agent** uses ReAct for explicit reasoning chains, visible in Langfuse. Outputs `NO_POSITION` when information is insufficient, regime is volatile with weak signal, or a binary risk event is detected. Also outputs `holding_period_days` calibrated to regime (1–3d volatile, 3–5d neutral, 5–10d trending).

**merge()** is pure Python — no LLM call. It gates historical stats injection by `confidence_in_prior` (only injects when `sample_size ≥ 10`) and adjusts candidate count by regime (6 in volatile, 8 in neutral, 10 in trending).

### MCP Pipeline (Day Trade)

The premarket deep scan uses Anthropic's beta MCP API, giving Claude autonomous tool access rather than pre-injecting fixed context:

```python
client.beta.messages.create(
    mcp_servers=[
        {"type": "url", "url": "https://mcp.tavily.com/...", "name": "tavily"},
        {"type": "url", "url": "https://mcp.supabase.com/...", "name": "supabase"},
    ],
    messages=[...]
)
```

Claude decides at inference time whether to call `tavily_search` or query `catalyst_stats` directly. A strong, unambiguous catalyst requires no extra tool calls; an ambiguous headline triggers both. All tool call sequences are traced in Langfuse.

---

## Memory & Learning Loop

```
Daily scan
  → Decision Agent produces signal
  → write_decision_snapshot() writes to swing_results:
      signal_id, catalyst_summary, memory_context, react_trace, price_at_scan

Daily at close (update_swing_outcomes.py)
  → fetch actual price at 5d / 10d / 20d
  → compute return, classify WIN / LOSS / NEUTRAL
  → backfill swing_results matched by signal_id

Next scan
  → Memory Agent queries swing_stats view (auto-updated)
  → win rates and avg returns reflect real accumulated outcomes
  → system improves as data accumulates
```

### Database Schema

```sql
swing_results      -- per-signal decisions with full decision snapshot (JSONB)
swing_news         -- news sources used per signal (for evidence tracing)
swing_stats        -- view: win rate + avg return by regime × signal
knowledge          -- curated trading rules injected into Memory Agent
scan_results       -- premarket scan history with outcomes
catalyst_stats     -- view: win rate by catalyst type (premarket)
news               -- news articles with vector embeddings (pgvector)
```

---

## Key Design Decisions

### Regime-Adaptive Factor Selection

Factor effectiveness depends on market environment. Empirical research on this dataset shows:

| Period | Regime | Best Factor | Mean IC |
|--------|--------|-------------|---------|
| 2019–2023 | Trending (low VIX) | momentum_20d | +0.030 |
| 2025–2026 | Volatile (tariff shock) | reversal_20d | +0.037 |

Regime output includes factor weights that are passed directly to Factor Worker, so the same factor can receive different weights depending on market conditions — not just on/off switching.

### LLM Role: Classifier, Verifier, and ReAct Reasoner

In Swing Trade, LLM acts as a **signal modifier** — factor signals are forward-looking but news and historical context can confirm, weaken, or negate them. The Decision Agent's explicit Thought chain makes the reasoning auditable and eval-able beyond just checking the final signal.

In Day Trade, LLM acts as a **catalyst verifier** — distinguishing FDA approval from FDA Fast Track, judging price proportionality, and flagging manipulation risk.

Factor computation stays in deterministic Python. LLM is only applied where language understanding and judgment matter.

### NO_POSITION as a Signal

The Decision Agent is designed to pass when the edge is unclear. `NO_POSITION` is triggered when:
- `catalyst_strength = NONE` and `news_alignment = NEUTRAL/CONTRADICTS`
- A binary risk event (earnings, FDA decision) is imminent
- Regime is VOLATILE and factor signal is weak
- `confidence_in_prior = NONE` and catalyst is ambiguous

Outputting fewer, higher-conviction signals is preferable to forcing 10 candidates regardless of evidence quality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI (api.py)                          │
│             Swing Trade endpoints + Day Trade endpoints         │
└──────┬──────────────────────────────────────┬────────────────────┘
       │                                      │
       ▼                                      ▼
  swing/                                premarket/
  ────────────────────────────          ────────────────────────────
  data.py          OHLCV + news         premarket_data.py
  regime.py        Regime detect          Universe, quotes, RVOL
  scanner.py       Factor scores         premarket_catalyst.py
  agents/          Phase 2 agents          MCP agentic pipeline
  ├ search_agent.py  ReAct + Tavily      premarket_scanner.py
  ├ memory_agent.py  ReAct + SQL           Scoring and ranking
  ├ merge.py         Context trim         update_outcomes.py
  └ decision_agent.py  Signal + hold        Daily outcome backfill
  main.py          Orchestration
  update_swing_outcomes.py
                                         ↓
                                   database.py
                                   Supabase PostgreSQL
                                   swing_results / scan_results
                                   knowledge / news / swing_news

┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                        │
│    Swing Trade tab                   Day Trade tab              │
│    Factor Scan (Phase 1)             Custom Scan (14 params)    │
│    Full Scan (multi-agent)           Deep Scan (MCP + RAG)      │
└─────────────────────────────────────────────────────────────────┘
```

```
scanner/
├── backend/
│   ├── api.py                        # FastAPI entry point
│   ├── database.py                   # Supabase access layer (read + write)
│   ├── swing/
│   │   ├── data.py                   # OHLCV + Yahoo news (yfinance)
│   │   ├── regime.py                 # Regime detection + LLM border judgment
│   │   ├── scanner.py                # Factor computation + cross-sectional ranking
│   │   ├── agents/
│   │   │   ├── search_agent.py       # ReAct: Yahoo → assess → Tavily (max 3)
│   │   │   ├── memory_agent.py       # ReAct: tiered SQL + knowledge rules
│   │   │   ├── merge.py              # Context assembly + regime gating (pure Python)
│   │   │   └── decision_agent.py     # ReAct: final signal + holding period
│   │   ├── main.py                   # Phase 1 + Phase 2 orchestration (asyncio)
│   │   └── update_swing_outcomes.py  # Daily outcome backfill (5d/10d/20d)
│   └── premarket/
│       ├── premarket_data.py         # Universe, Finnhub quotes, RVOL
│       ├── premarket_catalyst.py     # MCP agentic pipeline (Tavily + Supabase)
│       ├── premarket_scanner.py      # Scoring, ranking, Supabase logging
│       ├── update_outcomes.py        # Daily outcome backfill (30-min open return)
│       ├── small_cap_300m.json       # Pre-filtered universe (<$300M)
│       └── small_cap_100m.json       # Tighter universe (<$100M)
├── frontend/
│   └── src/
│       ├── App.jsx                   # React app (Swing + Day Trade tabs)
│       └── App.css
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

A **stability filter** prevents rapid switching: a new regime must persist for 2+ consecutive days before being confirmed. At the border between regimes, LLM judgment is used to resolve ambiguity using recent macro context.

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
catalyst_type      FDA_APPROVAL / FDA_FAST_TRACK / EARNINGS_BEAT / CONTRACT_WIN / DEAL_WIN / ...
catalyst_strength  STRONG / MODERATE / WEAK / NONE
proportionality    OVER / FAIR / UNDER
manipulation_risk  HIGH / MEDIUM / LOW
signal             TRADE / WATCH / AVOID
confidence         0–100
reason             one sentence
entry_timing       specific entry suggestion
```

---

## LLM Prompt Engineering

**Swing Trade Decision Agent prompt** includes:
- Factor score and regime context
- Search Agent catalyst summary (type, strength, alignment, risk flags)
- Memory Agent historical stats (gated by sample_size ≥ 10) and knowledge rules
- Regime-specific holding period hint
- Few-shot examples covering STRONG_BUY, NO_POSITION (binary risk), NO_POSITION (weak signal)
- Explicit ReAct format requirement with Thought chain before final JSON

**Day Trade prompt** includes:
- Critical distinction between FDA approval and FDA Fast Track
- Proportionality judgment
- Manipulation risk assessment
- Entry timing advice
- RAG context from historical database

**Confidence calibration (Swing Trade):**
```
85–100: Factor + news both strongly confirm + reliable historical stats
70–84 : Factor confirmed by news, moderate historical support
50–69 : Factor signal only, neutral news, limited history
30–49 : Mixed signals or contradicting news
10–29 : News contradicts signal or major risk present
 0    : NO_POSITION — active risk event or insufficient information
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
export TAVILY_API_KEY="your-tavily-key"
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
# Swing trade outcome backfill (after close)
python swing/update_swing_outcomes.py

# Day trade outcome backfill (10:00 AM ET)
python premarket/update_outcomes.py summary
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
| POST | `/api/premarket/scan/full` | Premarket scan with MCP agentic pipeline |

---

## Evaluation & Experiments

### Agent Components

| Component | Swing Trade | Day Trade |
|---|---|---|
| LLM reasoning | ReAct (Search + Memory + Decision agents) | MCP agentic pipeline |
| RAG | Memory Agent: swing_stats + knowledge table | catalyst_stats + knowledge injected into prompt |
| Web search | Tavily (Search Agent, max 3 calls) | Tavily MCP server |
| DB access | psycopg2 direct (Memory Agent) | Supabase MCP server |
| Observability | Langfuse traces per agent | Langfuse traces per scan |

### Eval Directory

```
eval/
├── golden.jsonl          # 25 hand-crafted test cases
├── run_eval.py           # Eval runner (baseline / rag_only / full modes)
├── judge.py              # LLM-judge with Cohen's kappa
├── human_scores.json     # Manual human scores for kappa computation
└── results/              # Saved eval results (JSON)

tests/
└── unit/
    └── test_catalyst.py  # 24 pytest unit tests (LLM mocked)
```

### Golden Set

25 cases covering: FDA approval vs Fast Track, earnings beat/miss, pump-and-dump, short squeeze, sector moves, M&A, dilution, analyst upgrades, and ambiguous catalyst wording.

### Experiment Log

| Round | Change | Pass Rate | Conclusion |
|-------|--------|-----------|------------|
| 0 | Baseline — no RAG, no MCP | 56% | Baseline established |
| 1 | Added RAG (catalyst_stats + knowledge) | 56% | No change — catalyst_stats empty at eval time |
| 2 | Added Tavily MCP search | 40% | Regression — richer context caused overconfidence on sector-move stocks |

### Metrics

**RAGAS Faithfulness: 0.088** — Low score expected; `catalyst_stats` still accumulating. Faithfulness projected to improve as `swing_results` and `scan_results` accumulate over 4–6 weeks.

**LLM-Judge: Cohen's Kappa = 0.310**

| Scorer | Avg Score | Method |
|--------|-----------|--------|
| LLM judge (Claude Sonnet) | 2.87 / 3.0 | Strict rubric: catalyst specificity, entry/exit timing |
| Human (author) | 2.73 / 3.0 | Practical trading utility |

Root cause: LLM judge positivity bias — scored 13/15 cases at 3/3. Disagreements concentrated on ACMR, BBAI, PRAX. Recommendation: cross-model judging or larger annotation set.

**Observability:** All production Claude calls traced via Langfuse, recording input, output, confidence, and tool usage per agent.

### Trade-off

Adding Tavily MCP search improved catalyst identification on ambiguous news but reduced golden set pass rate by 16pp (56% → 40%), increased latency from ~8s to ~15s per stock, and added ~$0.003 per analysis.

---

## Limitations

- RAG context requires data accumulation; meaningful stats need 30+ cases per signal type
- Swing Trade signal quality in VOLATILE regime is lower — mean-reversion signals have narrow edge without strong catalyst confirmation
- No real-time WebSocket streaming (snapshot-based)
- Premarket scanner uses Finnhub free tier (60 calls/min)

---

## Planned Improvements

- Factor Evo Agent: LLM-in-the-loop factor evolution inspired by CogAlpha, running weekly to propose and backtest new factor expressions
- Earnings calendar integration for binary event detection in Decision Agent
- pgvector semantic search in Memory Agent for similar historical case retrieval
- Polygon.io bulk snapshot API for faster premarket scanning
- Replay Agent for offline strategy backtesting using accumulated decision snapshots

---

## References

- Kakushadze (2016). *101 Formulaic Alphas*. arXiv:1601.00991
- Liu et al. (2025). *CogAlpha: Cognitive Alpha Mining with LLM-based Multi-Agent Framework*. arXiv:2511.18850
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

---

## Author

Jamie Ren · B.S. Computer Science & Statistics, University of Toronto · M.S. AI Engineering, UCLA Extension

*Built as a capstone project in AI engineering, combining quantitative finance and LLM systems.*
