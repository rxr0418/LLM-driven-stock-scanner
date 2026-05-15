"""
regime.py - Market regime detection module.

Detects the current market environment to guide factor selection:
  - TRENDING   : strong directional move, momentum factors work better
  - VOLATILE   : high uncertainty, reversal factors work better
  - NEUTRAL    : mixed signals, use combined approach

Detection logic uses three signals:
  1. VIX level         → absolute fear/greed measure
  2. Realized vol      → actual recent price volatility of SPY
  3. Trend strength    → how consistently SPY is trending
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Regime thresholds (tunable)
# ─────────────────────────────────────────────────────────────

VIX_HIGH     = 25.0  # VIX above this → volatile regime
VIX_LOW      = 15.0  # VIX below this → trending regime
RVOL_HIGH    = 0.20  # annualized realized vol above this → volatile
RVOL_LOW     = 0.12  # annualized realized vol below this → trending
TREND_STRONG = 0.60  # trend consistency above this → trending


# ─────────────────────────────────────────────────────────────
# Helper: get a clean closing price Series from yfinance
# ─────────────────────────────────────────────────────────────

def _get_close_series(ticker: str, period: str) -> pd.Series:
    """
    Download closing prices and return a clean 1-D Series.
    Handles yfinance MultiIndex columns for single-ticker downloads.
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex: ("Close", "AAPL") etc.
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # take first (only) column
    else:
        close = raw["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna()


# ─────────────────────────────────────────────────────────────
# Core signals
# ─────────────────────────────────────────────────────────────

def get_vix_level() -> float:
    """
    Fetch current VIX level.
    VIX = market's expectation of 30-day S&P 500 volatility (annualized).
    Higher VIX = more fear = more volatile environment expected.
    """
    try:
        close = _get_close_series("^VIX", period="5d")
        return float(close.iloc[-1])
    except Exception as e:
        print(f"[regime] VIX fetch failed: {e}")
        return 20.0  # default to neutral


def get_realized_vol(window: int = 20) -> float:
    """
    Compute SPY's annualized realized volatility over the last `window` days.

    Formula: std(daily_returns, window) * sqrt(252)

    Realized vol measures what actually happened, unlike VIX which is
    forward-looking (implied by options prices).
    """
    try:
        close   = _get_close_series("SPY", period="60d")
        returns = close.pct_change().dropna()
        rvol    = float(returns.tail(window).std() * np.sqrt(252))
        return rvol
    except Exception as e:
        print(f"[regime] Realized vol fetch failed: {e}")
        return 0.16  # default to neutral


def get_trend_strength(window: int = 20) -> float:
    """
    Measure how consistently SPY has been moving in one direction.

    Method:
      1. Determine overall direction over the window (up or down)
      2. Count the fraction of individual days that moved in that direction

    Returns a value between 0 and 1:
      > 0.60 → strong trend
      ~ 0.50 → random / choppy
      < 0.40 → counter-trend / very choppy
    """
    try:
        close   = _get_close_series("SPY", period="60d")
        returns = close.pct_change().dropna().tail(window)

        if len(returns) < window // 2:
            raise ValueError("Not enough data")

        # Compare last price to price `window` days ago to get overall direction
        price_now  = float(close.iloc[-1])
        price_then = float(close.iloc[-window])
        overall_direction = 1 if price_now > price_then else -1

        # Fraction of days that moved in the same direction
        same_direction = float((returns * overall_direction > 0).mean())
        return same_direction

    except Exception as e:
        print(f"[regime] Trend strength fetch failed: {e}")
        return 0.5  # default to neutral


# ─────────────────────────────────────────────────────────────
# Regime classifier
# ─────────────────────────────────────────────────────────────

def detect_regime() -> dict:
    """
    Combine VIX, realized vol, and trend strength into a regime label.

    Voting system: each signal casts one vote for VOLATILE or TRENDING.
    2+ votes → that regime wins. Otherwise → NEUTRAL.

    Returns:
        dict with regime label, raw metrics, recommended factors,
        and a human-readable description.
    """
    vix   = get_vix_level()
    rvol  = get_realized_vol()
    trend = get_trend_strength()

    # ── Voting ────────────────────────────────────────────────
    volatile_votes = 0
    trending_votes = 0

    if vix >= VIX_HIGH:
        volatile_votes += 1
    elif vix <= VIX_LOW:
        trending_votes += 1

    if rvol >= RVOL_HIGH:
        volatile_votes += 1
    elif rvol <= RVOL_LOW:
        trending_votes += 1

    if trend >= TREND_STRONG:
        trending_votes += 1
    elif trend <= (1 - TREND_STRONG):
        volatile_votes += 1

    # ── Label ─────────────────────────────────────────────────
    if volatile_votes >= 2:
        regime              = "VOLATILE"
        recommended_factors = ["reversal_5d", "reversal_20d", "vol_adjusted_reversal"]
        description         = (
            f"High volatility environment (VIX={vix:.1f}, RVol={rvol:.1%}). "
            f"Mean-reversion signals historically outperform. "
            f"Reversal factors favored."
        )
    elif trending_votes >= 2:
        regime              = "TRENDING"
        recommended_factors = ["momentum_20d", "momentum_60d"]
        description         = (
            f"Low volatility trending environment "
            f"(VIX={vix:.1f}, RVol={rvol:.1%}, Trend={trend:.0%} consistent). "
            f"Momentum signals historically outperform."
        )
    else:
        regime              = "NEUTRAL"
        recommended_factors = ["momentum_20d", "reversal_5d", "volume_spike"]
        description         = (
            f"Mixed signals (VIX={vix:.1f}, RVol={rvol:.1%}, Trend={trend:.0%}). "
            f"No clear regime. Using combined factor approach."
        )

    return {
        "regime":              regime,
        "vix":                 round(vix, 2),
        "realized_vol":        round(rvol, 4),
        "trend_strength":      round(trend, 4),
        "volatile_votes":      volatile_votes,
        "trending_votes":      trending_votes,
        "recommended_factors": recommended_factors,
        "description":         description,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def select_factors(regime_result: dict) -> list:
    """
    Return the list of factors to use given the detected regime.
    Called by scanner.py to know which signals to compute.
    """
    return regime_result["recommended_factors"]


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing regime.py...\n")

    print("1. VIX level...")
    vix = get_vix_level()
    print(f"   VIX = {vix:.2f}")
    if vix >= VIX_HIGH:
        print("   Interpretation: HIGH — fearful market, expect volatility")
    elif vix <= VIX_LOW:
        print("   Interpretation: LOW  — calm market, trend-following works")
    else:
        print("   Interpretation: NEUTRAL — mixed environment")

    print("\n2. Realized volatility (SPY 20-day)...")
    rvol = get_realized_vol()
    print(f"   Realized Vol = {rvol:.2%} annualized")

    print("\n3. Trend strength (SPY 20-day)...")
    trend = get_trend_strength()
    print(f"   Trend consistency = {trend:.0%} of days moving in trend direction")

    print("\n4. Full regime detection...")
    result = detect_regime()
    print(f"   Regime            : {result['regime']}")
    print(f"   VIX               : {result['vix']}")
    print(f"   Realized Vol      : {result['realized_vol']:.2%}")
    print(f"   Trend Strength    : {result['trend_strength']:.0%}")
    print(f"   Votes (V/T)       : {result['volatile_votes']} / {result['trending_votes']}")
    print(f"   Recommended       : {result['recommended_factors']}")
    print(f"   Description       : {result['description']}")
    print(f"   Timestamp         : {result['timestamp']}")