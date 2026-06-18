"""
rag/seed_sector_knowledge.py - Seed sector/industry-level trading rules into knowledge table.

These are generalizable, regime-aware rules that apply across tickers in the same sector.
Run once: python seed_sector_knowledge.py
Add more entries to SECTOR_RULES as you discover patterns.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


SECTOR_RULES = [
    # ── Technology ────────────────────────────────────────────
    (
        "Semiconductors track SOX index closely — single-stock momentum signals need "
        "sector confirmation. If SOX is down >2% on the day, fade individual semis BUY signals.",
        "sector", "HIGH",
    ),
    (
        "Cloud/SaaS stocks often gap up on earnings beats but mean-revert within 3 days "
        "unless ARR growth acceleration is >5ppt. Don't chase post-earnings momentum without guidance raise.",
        "sector", "HIGH",
    ),
    (
        "Mega-cap tech (AAPL, MSFT, GOOGL, META, NVDA) exhibit low factor alpha — institutional "
        "ownership is so high that momentum signals are noisier. Require catalyst_strength=STRONG.",
        "sector", "MEDIUM",
    ),

    # ── Healthcare / Biotech ──────────────────────────────────
    (
        "Biotech stocks in FDA review window (PDUFA date within 30 days) have binary outcome risk. "
        "Do not trade momentum signals — the distribution is bimodal, not normal.",
        "sector", "HIGH",
    ),
    (
        "Large-cap pharma (PFE, MRK, JNJ, ABBV) move on patent cliff news and pipeline updates. "
        "A single drug approval can trigger 5-15% moves. Catalyst must be drug-specific.",
        "sector", "MEDIUM",
    ),
    (
        "Hospital/HMO stocks (UNH, HUM, CVS) are sensitive to CMS reimbursement rate announcements "
        "and MLR guidance. These macro catalysts override individual factor scores.",
        "sector", "MEDIUM",
    ),

    # ── Financials ────────────────────────────────────────────
    (
        "Bank stocks (JPM, BAC, GS, MS, WFC) lead/lag the yield curve. "
        "In rising rate environments, BUY signals on banks have higher win rates. "
        "In inverted curve (2s10s < -30bps), reduce confidence by 20 points.",
        "sector", "HIGH",
    ),
    (
        "Regional banks are highly sensitive to deposit flight news (post-SVB regime). "
        "Any news mentioning 'deposits', 'liquidity', or 'withdrawals' is a hard risk flag — "
        "override signal to NO_POSITION regardless of factor score.",
        "sector", "HIGH",
    ),
    (
        "Insurance stocks (CB, TRV, ALL) are slow-moving with mean-reverting returns. "
        "Momentum signals rarely persist beyond 3 days. Use tight holding periods.",
        "sector", "MEDIUM",
    ),

    # ── Energy ───────────────────────────────────────────────
    (
        "Oil majors (XOM, CVX, COP) are highly correlated with WTI crude. "
        "A BUY signal with crude trending up >5% over 10 days has strong confirmation. "
        "Ignore BUY signals when crude is in backwardation and declining.",
        "sector", "HIGH",
    ),
    (
        "Clean energy/solar stocks (ENPH, SEDG, FSLR) are policy-sensitive. "
        "IRA/subsidy news creates multi-day momentum. Earnings misses in this sector "
        "are often priced in slowly — momentum can extend 5-10 days post-earnings.",
        "sector", "MEDIUM",
    ),

    # ── Consumer ─────────────────────────────────────────────
    (
        "Retail stocks are sensitive to monthly same-store sales data and foot traffic. "
        "Factor signals are strongest when aligned with consumer sentiment trend (U. Michigan).",
        "sector", "MEDIUM",
    ),
    (
        "Consumer staples (WMT, COST, PG, KO) are defensive — BUY signals in VOLATILE regime "
        "are more reliable than in TRENDING regime (flight-to-safety dynamic).",
        "sector", "MEDIUM",
    ),

    # ── Industrials / Defense ────────────────────────────────
    (
        "Defense contractors (LMT, RTX, NOC, GD) have sticky government revenue. "
        "DoD contract wins create durable 5-10 day momentum — treat CONTRACT_WIN catalyst "
        "as STRONG confirmation for defense stocks specifically.",
        "sector", "HIGH",
    ),
    (
        "Airlines (DAL, UAL, AAL, LUV) have high operating leverage to fuel costs. "
        "Momentum signals are less reliable without checking jet fuel price trend.",
        "sector", "LOW",
    ),

    # ── Regime-level cross-sector ────────────────────────────
    (
        "In VOLATILE regime, small-cap momentum signals (stocks with market cap < $10B) "
        "have 40% lower win rates due to liquidity compression. Require higher factor scores (>0.75).",
        "sector", "HIGH",
    ),
    (
        "In TRENDING regime with VIX < 15, cross-sector momentum is more reliable — "
        "factor scores above 0.7 have historically resolved in the signal direction "
        "within 5 days across most sectors.",
        "sector", "MEDIUM",
    ),
    (
        "Earnings season (Jan, Apr, Jul, Oct weeks 2-4): reduce holding_period_days by 2 "
        "across all sectors to avoid holding through binary events.",
        "sector", "HIGH",
    ),
]


def seed() -> None:
    from database import add_knowledge

    print(f"[seed_sector] Seeding {len(SECTOR_RULES)} sector knowledge entries...")
    saved = 0
    for content, category, confidence in SECTOR_RULES:
        ok = add_knowledge(content, category=category, confidence=confidence, source="sector_seed")
        if ok:
            print(f"  ✓ [{category}/{confidence}] {content[:80]}...")
            saved += 1
        else:
            print(f"  ✗ Failed to save: {content[:60]}...")

    print(f"[seed_sector] Done — {saved}/{len(SECTOR_RULES)} entries saved.")


if __name__ == "__main__":
    seed()
