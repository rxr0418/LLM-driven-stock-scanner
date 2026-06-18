"""
swing/factor_evo/eval_factors.py - Factor IC evaluation with train/test split.

Computes Spearman Rank IC (Information Coefficient) between factor scores
and forward returns. Used by Factor Evo Agent to evaluate generated factors.

IC = Spearman correlation(factor_rank, forward_return_rank) across stocks
  > 0.05 : useful signal
  > 0.10 : strong signal
  < 0.02 : discard

Usage:
  from swing.eval_factors import evaluate_factor, evaluate_all_factors
  result = evaluate_factor(fn, close, volume, forward_days=5)
"""

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Core IC computation
# ─────────────────────────────────────────────────────────────

def compute_ic_series(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """
    Compute single cross-sectional Spearman IC.
    Both series indexed by ticker; aligned on common tickers.
    Returns NaN if fewer than 20 common tickers.
    """
    common = factor_scores.index.intersection(forward_returns.index)
    if len(common) < 20:
        return float("nan")
    f = factor_scores.loc[common]
    r = forward_returns.loc[common]
    # Drop rows where either series is NaN
    mask = f.notna() & r.notna()
    f, r = f[mask], r[mask]
    if len(f) < 20:
        return float("nan")
    ic, _ = stats.spearmanr(f.values, r.values)
    return float(ic)


def compute_forward_returns(close: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    """
    For each date t, compute cross-sectional forward returns:
      return(t) = close(t + forward_days) / close(t) - 1

    Returns DataFrame with same index as close (NaN at tail where future unavailable).
    """
    return close.shift(-forward_days) / close - 1


# ─────────────────────────────────────────────────────────────
# Single factor evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_factor(
    factor_fn: Callable,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    forward_days: int = 5,
    train_ratio: float = 0.7,
    min_periods: int = 30,
) -> dict:
    """
    Evaluate one factor function over a rolling IC time series.

    Args:
        factor_fn    : function(close, volume) -> pd.Series (one score per ticker)
        close        : DataFrame (dates × tickers)
        volume       : DataFrame (dates × tickers)
        forward_days : holding period for forward return calculation
        train_ratio  : fraction of dates used as train set
        min_periods  : minimum history needed before computing factor

    Returns:
        {
          ic_mean_train, ic_std_train, ir_train,    # in-sample
          ic_mean_test,  ic_std_test,  ir_test,     # out-of-sample
          ic_mean_all,   ic_win_rate,               # overall
          n_train_dates, n_test_dates,
          forward_days,  status,
        }
    """
    fwd_returns = compute_forward_returns(close, forward_days)
    n_dates = len(close)
    split_idx = int(n_dates * train_ratio)

    ic_series = []
    dates = close.index

    for i in range(min_periods, n_dates - forward_days):
        # Factor uses history up to date i
        close_window  = close.iloc[:i + 1]
        volume_window = volume.iloc[:i + 1]

        try:
            scores = factor_fn(close_window, volume_window)
        except Exception:
            continue

        if scores.empty or scores.isna().all():
            continue

        fwd = fwd_returns.iloc[i]
        ic = compute_ic_series(scores, fwd)
        if not np.isnan(ic):
            ic_series.append({"date": dates[i], "ic": ic, "is_train": i < split_idx})

    if not ic_series:
        return {"status": "insufficient_data", "ic_mean_all": 0.0}

    df = pd.DataFrame(ic_series)
    train = df[df["is_train"]]["ic"]
    test  = df[~df["is_train"]]["ic"]
    all_ic = df["ic"]

    def safe_ir(s: pd.Series) -> float:
        return float(s.mean() / s.std()) if len(s) > 1 and s.std() > 0 else 0.0

    return {
        "status":          "ok",
        "forward_days":    forward_days,
        # Train
        "ic_mean_train":   round(float(train.mean()), 4) if len(train) else 0.0,
        "ic_std_train":    round(float(train.std()),  4) if len(train) else 0.0,
        "ir_train":        round(safe_ir(train), 3),
        "n_train_dates":   len(train),
        # Test
        "ic_mean_test":    round(float(test.mean()), 4) if len(test) else 0.0,
        "ic_std_test":     round(float(test.std()),  4) if len(test) else 0.0,
        "ir_test":         round(safe_ir(test), 3),
        "n_test_dates":    len(test),
        # Overall
        "ic_mean_all":     round(float(all_ic.mean()), 4),
        "ic_win_rate":     round(float((all_ic > 0).mean()), 3),
    }


# ─────────────────────────────────────────────────────────────
# Evaluate all registered factors
# ─────────────────────────────────────────────────────────────

def evaluate_all_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    forward_days: int = 5,
    train_ratio: float = 0.7,
) -> dict[str, dict]:
    """
    Evaluate every factor in FACTOR_REGISTRY and return IC results.
    Used for benchmarking baseline factors and ranking them.
    """
    from swing.scanner import FACTOR_REGISTRY

    results = {}
    for name, fn in FACTOR_REGISTRY.items():
        print(f"  [eval] {name} (forward={forward_days}d)...", end=" ", flush=True)
        result = evaluate_factor(fn, close, volume, forward_days, train_ratio)
        results[name] = result
        if result["status"] == "ok":
            print(
                f"IC_train={result['ic_mean_train']:+.4f} "
                f"IC_test={result['ic_mean_test']:+.4f} "
                f"IR_test={result['ir_test']:.3f}"
            )
        else:
            print(result["status"])

    return results


# ─────────────────────────────────────────────────────────────
# Evaluate a dynamically generated factor (from evo agent)
# ─────────────────────────────────────────────────────────────

def evaluate_generated_factor(
    code: str,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    forward_days: int = 5,
    train_ratio: float = 0.7,
) -> dict:
    """
    Compile and evaluate a factor function string produced by Factor Evo Agent.
    The code must define a function named `factor_generated(close, volume) -> pd.Series`.

    Runs inside a sandbox (see sandbox.py) before this function is called.
    Returns evaluation result dict, or error dict on compile/runtime failure.
    """
    # Compile
    try:
        namespace = {"pd": pd, "np": np}
        exec(compile(code, "<factor>", "exec"), namespace)  # noqa: S102
        fn = namespace.get("factor_generated")
        if fn is None:
            return {"status": "error", "error": "function 'factor_generated' not defined", "ic_mean_all": 0.0}
    except SyntaxError as e:
        return {"status": "error", "error": f"SyntaxError: {e}", "ic_mean_all": 0.0}
    except Exception as e:
        return {"status": "error", "error": f"CompileError: {e}", "ic_mean_all": 0.0}

    # Evaluate
    try:
        return evaluate_factor(fn, close, volume, forward_days, train_ratio)
    except Exception as e:
        return {"status": "error", "error": f"RuntimeError: {e}", "ic_mean_all": 0.0}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    from swing.data import fetch_price_data, UNIVERSE

    forward_days = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Fetching price data ({len(UNIVERSE)} tickers)...")
    price_data = fetch_price_data(UNIVERSE, lookback_days=252)
    close  = price_data["close"]
    volume = price_data["volume"]
    print(f"Data: {close.shape[0]} days × {close.shape[1]} stocks\n")

    print(f"Evaluating all factors (forward={forward_days}d)...\n")
    results = evaluate_all_factors(close, volume, forward_days=forward_days)

    print("\n── Factor IC Summary ──")
    print(f"{'Factor':<25} {'IC_train':>9} {'IC_test':>9} {'IR_test':>8} {'WinRate':>8}")
    print("─" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1].get("ic_mean_test", 0)):
        if r["status"] == "ok":
            print(
                f"{name:<25} "
                f"{r['ic_mean_train']:>+9.4f} "
                f"{r['ic_mean_test']:>+9.4f} "
                f"{r['ir_test']:>8.3f} "
                f"{r['ic_win_rate']:>8.1%}"
            )
