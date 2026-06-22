"""
scanner.py - Core factor computation and stock screening module.

Responsibilities:
  - Compute factor scores for all stocks in the universe
  - Select which factors to use based on detected market regime
  - Rank stocks and return top/bottom candidates for the watchlist

Factor library:
  reversal_5d          : 5-day price reversal
  reversal_20d         : 20-day price reversal
  momentum_20d         : 20-day price momentum
  momentum_60d         : 60-day price momentum
  volume_spike         : abnormal volume relative to 20-day average
  vol_adjusted_reversal: reversal normalized by realized volatility
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Weight multiplier for IC-validated evolved factors vs hand-written factors
EVOLVED_FACTOR_WEIGHT_MULTIPLIER = 1.5


# ─────────────────────────────────────────────────────────────
# 1. Factor definitions
#    Each factor: (close, volume) -> pd.Series (one score per ticker)
#    Higher score = bullish signal (predict outperformance)
# ─────────────────────────────────────────────────────────────

def factor_reversal_5d(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    5-day price reversal.
    Stocks that fell the most over 5 days get the highest score.
    Logic: short-term oversold stocks tend to mean-revert.
    Best in: VOLATILE / choppy markets.
    """
    signal = -close.diff(5).iloc[-1]
    return signal.dropna()


def factor_reversal_20d(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    20-day price reversal.
    Same logic as reversal_5d but over a longer window.
    Captures medium-term oversold conditions.
    Best in: VOLATILE markets with sustained dislocations.
    """
    signal = -close.diff(20).iloc[-1]
    return signal.dropna()


def factor_momentum_20d(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    20-day price momentum.
    Stocks that rose the most over 20 days get the highest score.
    Logic: winners keep winning in trending markets.
    Best in: TRENDING markets.
    """
    signal = close.pct_change(20).iloc[-1]
    return signal.dropna()


def factor_momentum_60d(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    60-day price momentum.
    Longer-window momentum, more stable signal.
    Best in: TRENDING markets with sustained moves.
    """
    signal = close.pct_change(60).iloc[-1]
    return signal.dropna()


def factor_volume_spike(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    Abnormal volume spike.
    Today's volume relative to 20-day average volume.
    High ratio = unusual activity = potential catalyst.
    Used in: NEUTRAL regime as a supplementary signal.
    """
    avg_volume = volume.rolling(20).mean()
    signal     = (volume / avg_volume).iloc[-1]
    return signal.dropna()


def factor_vol_adjusted_reversal(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    Volatility-adjusted reversal.
    Reversal signal normalized by recent realized volatility.
    Stocks that dropped a lot RELATIVE TO their own typical volatility
    get a higher score — more likely to be genuine dislocations.
    Best in: VOLATILE markets.
    """
    reversal   = -close.diff(5)
    volatility = close.pct_change().rolling(10).std()

    # Normalize: how many "standard moves" did this stock drop?
    normalized = reversal / (volatility * close + 1e-8)
    signal     = normalized.iloc[-1]
    return signal.dropna()


# Factor registry: name → function (hand-written baseline factors)
FACTOR_REGISTRY = {
    "reversal_5d":           factor_reversal_5d,
    "reversal_20d":          factor_reversal_20d,
    "momentum_20d":          factor_momentum_20d,
    "momentum_60d":          factor_momentum_60d,
    "volume_spike":          factor_volume_spike,
    "vol_adjusted_reversal": factor_vol_adjusted_reversal,
}


def load_evolved_factors(regime: str, forward_days: int = 5) -> dict:
    """
    Load IC-validated evolved factors from Supabase for the given regime.
    Compiles each factor's code string into a callable and returns a dict:
      { factor_name: {"fn": callable, "ic_mean_test": float, "ir_test": float} }

    Called once per scan run — no LLM involved, just DB read + exec().
    Falls back to empty dict silently if DB is unavailable.
    """
    try:
        from swing.factor_evo.factor_store import load_top_factors
        rows = load_top_factors(regime, forward_days=forward_days, limit=5)
    except Exception as e:
        print(f"[scanner] evolved factor load skipped: {e}")
        return {}

    from swing.factor_evo.sandbox import ast_check

    loaded = {}
    for row in rows:
        name = row["name"]
        code = row["code"]
        try:
            # Re-verify AST safety on every load — defence-in-depth against
            # DB tampering or a sandboxed payload that slipped past generation checks.
            violations = ast_check(code)
            if violations:
                print(f"[scanner] evolved:{name} failed AST re-check on load, skipping: {violations[0]}")
                continue

            namespace = {"pd": pd, "np": np}
            exec(compile(code, f"<evolved:{name}>", "exec"), namespace)  # noqa: S102
            fn = namespace.get("factor_generated")
            if fn is None:
                print(f"[scanner] evolved:{name} has no factor_generated, skipping")
                continue
            loaded[name] = {
                "fn":           fn,
                "ic_mean_test": row.get("ic_mean_test", 0.0),
                "ir_test":      row.get("ir_test", 0.0),
            }
            print(f"[scanner] loaded evolved factor: {name} "
                  f"(IC_test={row.get('ic_mean_test', 0):+.4f}, IR={row.get('ir_test', 0):.3f})")
        except Exception as e:
            print(f"[scanner] evolved:{name} compile failed: {e}")

    return loaded


# ─────────────────────────────────────────────────────────────
# 2. Cross-sectional ranking
# ─────────────────────────────────────────────────────────────

def rank_stocks(factor_scores: pd.Series) -> pd.Series:
    """
    Convert raw factor scores to cross-sectional percentile ranks.
    Rank of 1.0 = highest score (strongest signal).
    Rank of 0.0 = lowest score (weakest signal).

    Using percentile ranks makes scores comparable across factors
    with different scales (e.g. price-based vs volume-based).
    """
    return factor_scores.rank(pct=True, ascending=True)


def combine_factors(
    factor_scores_dict: dict,
    weights: dict = None,
) -> pd.Series:
    """
    Combine multiple factor scores into a single composite score.

    Args:
        factor_scores_dict : {factor_name: pd.Series of raw scores}
        weights            : {factor_name: weight} — defaults to equal weight

    Returns:
        pd.Series of composite scores, one per ticker
    """
    if not factor_scores_dict:
        return pd.Series(dtype=float)

    # Default to equal weights
    if weights is None:
        w = 1.0 / len(factor_scores_dict)
        weights = {name: w for name in factor_scores_dict}

    # Rank each factor cross-sectionally, then weighted average
    ranked_scores = []
    for name, scores in factor_scores_dict.items():
        ranked = rank_stocks(scores)
        weight = weights.get(name, 1.0 / len(factor_scores_dict))
        ranked_scores.append(ranked * weight)

    # Align on common tickers and sum
    composite = pd.concat(ranked_scores, axis=1).sum(axis=1)

    # Normalize to [0, 1]
    min_val = composite.min()
    max_val = composite.max()
    if max_val > min_val:
        composite = (composite - min_val) / (max_val - min_val)

    return composite.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────
# 3. Main scan function
# ─────────────────────────────────────────────────────────────

def run_scan(
    price_data: dict,
    regime_result: dict,
    top_n: int = 20,
) -> dict:
    """
    Run the full scan: compute factors, combine, rank, return watchlist.

    Args:
        price_data    : output of data.fetch_price_data()
                        keys: close, open, high, low, volume
        regime_result : output of regime.detect_regime()
        top_n         : number of stocks to return in each list

    Returns:
        dict with keys:
          regime          : detected regime label
          long_candidates : top N stocks (strongest bullish signal)
          short_candidates: bottom N stocks (strongest bearish signal)
          factor_scores   : full composite scores for all stocks
          factors_used    : list of factor names used
          description     : regime description
    """
    close  = price_data.get("close")
    volume = price_data.get("volume")

    if close is None or close.empty:
        return {"error": "No price data available"}

    regime_label = regime_result["regime"]

    # ── Select hand-written factors based on regime ───────────
    factors_to_use = regime_result.get(
        "recommended_factors",
        ["momentum_20d", "reversal_5d", "volume_spike"]
    )

    print(f"[scanner] Regime: {regime_label}")
    print(f"[scanner] Hand-written factors: {factors_to_use}")

    # ── Load evolved factors for this regime ──────────────────
    evolved = load_evolved_factors(regime_label)
    if evolved:
        print(f"[scanner] Evolved factors loaded: {list(evolved.keys())}")

    # ── Compute each hand-written factor ─────────────────────
    factor_scores_dict = {}
    weights: dict[str, float] = {}

    for factor_name in factors_to_use:
        fn = FACTOR_REGISTRY.get(factor_name)
        if fn is None:
            print(f"[scanner] Unknown factor: {factor_name}, skipping")
            continue
        try:
            scores = fn(close, volume)
            if scores.empty:
                print(f"[scanner] {factor_name}: no valid scores, skipping")
                continue
            factor_scores_dict[factor_name] = scores
            weights[factor_name] = 1.0
            print(f"[scanner] {factor_name}: scored {len(scores)} stocks")
        except Exception as e:
            print(f"[scanner] {factor_name} failed: {e}")

    # ── Compute each evolved factor ───────────────────────────
    for name, meta in evolved.items():
        try:
            scores = meta["fn"](close, volume)
            if scores.empty:
                continue
            factor_scores_dict[name] = scores
            weights[name] = EVOLVED_FACTOR_WEIGHT_MULTIPLIER
            print(f"[scanner] evolved:{name}: scored {len(scores)} stocks (weight={EVOLVED_FACTOR_WEIGHT_MULTIPLIER}×)")
        except Exception as e:
            print(f"[scanner] evolved:{name} runtime failed: {e}")

    if not factor_scores_dict:
        return {"error": "All factors failed"}

    # Normalize weights so they sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # ── Combine into composite score ──────────────────────────
    composite = combine_factors(factor_scores_dict, weights=weights)

    # ── Build watchlist ───────────────────────────────────────
    long_candidates  = composite.head(top_n)
    short_candidates = composite.tail(top_n).sort_values(ascending=True)

    # Format output as list of dicts for easy JSON serialization
    def to_list(series: pd.Series) -> list:
        return [
            {"ticker": ticker, "score": round(float(score), 4)}
            for ticker, score in series.items()
        ]

    return {
        "regime":           regime_result["regime"],
        "long_candidates":  to_list(long_candidates),
        "short_candidates": to_list(short_candidates),
        "factor_scores":    composite.round(4).to_dict(),
        "factors_used":     list(factor_scores_dict.keys()),
        "evolved_factors":  list(evolved.keys()),
        "description":      regime_result["description"],
        "timestamp":        regime_result["timestamp"],
    }


# ─────────────────────────────────────────────────────────────
# 4. Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data   import fetch_price_data, UNIVERSE
    from regime import detect_regime

    print("Testing scanner.py...\n")

    # Use a small subset for quick testing
    test_tickers = UNIVERSE

    print("1. Fetching price data...")
    price_data = fetch_price_data(test_tickers, lookback_days=90)
    print(f"   Got {price_data['close'].shape[0]} days x {price_data['close'].shape[1]} stocks\n")

    print("2. Detecting regime...")
    regime_result = detect_regime()
    print(f"   Regime: {regime_result['regime']}\n")

    print("3. Running scan...")
    results = run_scan(price_data, regime_result, top_n=5)

    if "error" in results:
        print(f"   Error: {results['error']}")
    else:
        print(f"\n   Regime     : {results['regime']}")
        print(f"   Factors    : {results['factors_used']}")

        print(f"\n   TOP 5 LONG candidates:")
        for item in results["long_candidates"]:
            print(f"     {item['ticker']:<8} score={item['score']:.4f}")

        print(f"\n   TOP 5 SHORT candidates:")
        for item in results["short_candidates"]:
            print(f"     {item['ticker']:<8} score={item['score']:.4f}")