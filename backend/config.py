"""
config.py - Central configuration for the LLM-Driven Stock Scanner.

All model names, API limits, and tunable constants live here.
Import from this file instead of hardcoding values in agent modules.
"""

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

# Primary model for reasoning-heavy tasks (decision agent, llm_analyst)
ANALYST_MODEL = "claude-sonnet-4-6"

# Fast/cheap model for structured classification tasks (search agent)
SEARCH_AGENT_MODEL = "claude-haiku-4-5-20251001"

# Fast/cheap model for eval judge (low token output, high volume)
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Embedding model for pgvector RAG (OpenAI-compatible via Anthropic SDK not needed;
# use text-embedding-3-small via openai client or voyage-3-lite via anthropic)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536


# ─────────────────────────────────────────────────────────────
# Agent limits
# ─────────────────────────────────────────────────────────────

# Search Agent: max Tavily web searches per ticker
MAX_TAVILY_SEARCHES = 2

# Decision Agent: max retries on schema validation failure
MAX_DECISION_RETRIES = 2

# Search Agent: max tokens per LLM call
SEARCH_AGENT_MAX_TOKENS    = 500

# Decision Agent: max tokens per LLM call
DECISION_AGENT_MAX_TOKENS  = 900

# Skeptic Agent: max tokens per critique call
SKEPTIC_AGENT_MAX_TOKENS   = 700

# LLM Analyst (single-agent path): max tokens
ANALYST_MAX_TOKENS         = 400

# Judge: max tokens (only needs to output a single integer)
JUDGE_MAX_TOKENS           = 10


# ─────────────────────────────────────────────────────────────
# Scanner / data
# ─────────────────────────────────────────────────────────────

# Days of price history to fetch for factor computation
LOOKBACK_DAYS = 252   # 1 trading year — needed for reliable IC calculation

# Minimum historical samples before Memory Agent trusts swing_stats
MIN_MEMORY_SAMPLE = 10

# RAG retrieval limits — tune these if Decision Agent prompt gets too long
MAX_KNOWLEDGE_RULES  = 4   # knowledge + sector rules returned to Decision Agent
MAX_SIMILAR_CASES    = 3   # past swing_results cases
MAX_ANALYST_RATINGS  = 3   # analyst rating changes shown in prompt
MAX_SEC_FILINGS      = 2   # SEC 8-K/10-Q summaries shown in prompt


# ─────────────────────────────────────────────────────────────
# Regime thresholds (mirrors regime.py — single source of truth)
# ─────────────────────────────────────────────────────────────

VIX_HIGH     = 25.0
VIX_LOW      = 15.0
RVOL_HIGH    = 0.20
RVOL_LOW     = 0.12
TREND_STRONG = 0.60
