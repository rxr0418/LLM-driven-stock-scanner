"""
swing/factor_evo/ - Factor Evolution Agent (Phase 1 extension).

Offline research pipeline — not in the per-ticker scan path.

Modules:
  eval_factors.py    : Spearman IC computation with train/test split
  sandbox.py         : AST blacklist + subprocess isolation for generated code
  factor_store.py    : Persist winning factors + IC history to Supabase
  factor_evo_agent.py: LLM generation + mutation loop

Run:
  python -m swing.factor_evo.factor_evo_agent --regime TRENDING --generations 3
"""
