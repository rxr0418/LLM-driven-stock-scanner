"""
swing/pure_baseline_runner.py - Pure LLM baseline with zero structured input.

Gives Claude only:
  - Today's date
  - Current market regime + VIX
  - A pool of ~100 tickers to pick from

No factor scores, no news, no RAG. Pure LLM knowledge.
Picks 3 BUY + 3 SHORT, saves to swing_results_pure_baseline.

Usage:
  cd backend && python3 swing/pure_baseline_runner.py
  cd backend && python3 swing/pure_baseline_runner.py --save-db
"""

import argparse
import json
import os
import random
import sys
from datetime import date, datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYST_MODEL
from logger import get_logger
from resilience import with_retry

log = get_logger(__name__)

SYSTEM_PROMPT = """You are an experienced swing trader. You will be given today's date,
current market regime, and a list of tickers. Pick exactly 3 for BUY and 3 for SHORT
based purely on your knowledge of these companies and current market conditions.

Swing trades hold 3-10 days. Focus on stocks with clear near-term catalysts or momentum.

Respond ONLY with this JSON (no markdown, no explanation outside the JSON):
{
  "picks": [
    {"ticker": "XXX", "signal": "BUY",   "confidence": 0-100, "reason": "one sentence"},
    {"ticker": "XXX", "signal": "BUY",   "confidence": 0-100, "reason": "one sentence"},
    {"ticker": "XXX", "signal": "BUY",   "confidence": 0-100, "reason": "one sentence"},
    {"ticker": "XXX", "signal": "SHORT", "confidence": 0-100, "reason": "one sentence"},
    {"ticker": "XXX", "signal": "SHORT", "confidence": 0-100, "reason": "one sentence"},
    {"ticker": "XXX", "signal": "SHORT", "confidence": 0-100, "reason": "one sentence"}
  ]
}"""


def _load_ticker_pool(n: int = 100) -> list[str]:
    """
    Load tickers from the scanner cache and sample n of them.
    Falls back to a hardcoded S&P 100 subset if cache unavailable.
    """
    try:
        from swing.data import load_sp500
        tickers = load_sp500()
        if len(tickers) > n:
            tickers = random.sample(tickers, n)
        return sorted(tickers)
    except Exception as e:
        log.warning("ticker cache unavailable, using fallback pool", extra={"error": str(e)})
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK.B","JPM","V",
            "UNH","XOM","JNJ","PG","MA","HD","CVX","MRK","ABBV","PEP","COST",
            "LLY","AVGO","TMO","MCD","ACN","DHR","ABT","WMT","BAC","CRM","NFLX",
            "AMD","INTC","QCOM","TXN","NOW","AMAT","LRCX","MRVL","ADI","KLAC",
            "GS","MS","BLK","AXP","SCHW","CB","MMC","AON","TFC","USB",
            "UNP","UPS","FDX","CSX","NSC","DAL","UAL","AAL","LUV","JBLU",
            "GE","HON","MMM","CAT","DE","EMR","ETN","PH","ROK","AME",
            "CVS","CI","HUM","ELV","CNC","MCK","CAH","ABC","MOH","HCA",
            "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","ED","XEL",
            "SPG","PLD","AMT","CCI","EQIX","PSA","EQR","AVB","WY","ARE",
        ]


@with_retry(label="pure_baseline/anthropic")
def _call_llm(client: anthropic.Anthropic, messages: list) -> anthropic.types.Message:
    return client.messages.create(
        model=ANALYST_MODEL,
        max_tokens=600,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )


def run_pure_baseline(
    save_db: bool = False,
    scan_date: date | None = None,
    ticker_pool_size: int = 100,
) -> list[dict]:
    """
    Run one pure-LLM baseline call. Returns list of 6 picks.
    """
    scan_date = scan_date or date.today()
    client    = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Get regime for context
    regime = "NEUTRAL"
    vix    = "unknown"
    try:
        from swing.regime import detect_regime
        r      = detect_regime()
        regime = r["regime"]
        vix    = f"{r['vix']:.1f}"
    except Exception:
        pass

    tickers = _load_ticker_pool(ticker_pool_size)
    ticker_str = ", ".join(tickers)

    user_msg = (
        f"Today: {scan_date}  |  Market regime: {regime}  |  VIX: {vix}\n\n"
        f"Ticker pool ({len(tickers)} stocks):\n{ticker_str}\n\n"
        f"Pick 3 BUY and 3 SHORT for swing trading."
    )

    log.info("pure baseline start", extra={"regime": regime, "vix": vix,
                                            "pool_size": len(tickers)})
    try:
        resp = _call_llm(client, [{"role": "user", "content": user_msg}])
        text = resp.content[0].text.strip()
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        parsed = json.loads(cleaned[start:end]) if start != -1 else {}
        picks  = parsed.get("picks", [])

        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost    = in_tok * 3e-6 + out_tok * 15e-6

        results = []
        for p in picks[:6]:
            results.append({
                "ticker":        p.get("ticker", ""),
                "signal":        p.get("signal", "NO_POSITION"),
                "confidence":    int(p.get("confidence", 0)),
                "reason":        p.get("reason", "")[:500],
                "regime":        regime,
                "prompt_tokens": in_tok,
                "cost_usd":      round(cost / len(picks), 5) if picks else 0,
            })

        log.info("pure baseline done", extra={
            "picks": len(results), "cost_usd": round(cost, 4),
        })

        if save_db and results:
            _save_to_db(results, scan_date)

        return results

    except Exception as e:
        log.error("pure baseline failed", extra={"error": str(e)})
        return []


def _save_to_db(picks: list[dict], scan_date: date) -> None:
    try:
        from database import get_connection
        import yfinance as yf

        conn = get_connection()
        cur  = conn.cursor()

        for p in picks:
            ticker    = p["ticker"]
            signal_id = f"PB_{scan_date.strftime('%Y%m%d')}_{ticker}"

            # Fetch current price
            price = None
            try:
                hist  = yf.Ticker(ticker).history(period="1d")
                price = round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None
            except Exception:
                pass

            cur.execute("""
                INSERT INTO swing_results_pure_baseline
                  (scan_date, ticker, signal, confidence, regime,
                   reason, prompt_tokens, cost_usd, price_at_scan,
                   created_at, signal_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (signal_id) DO NOTHING
            """, (
                scan_date, ticker, p["signal"], p["confidence"], p["regime"],
                p["reason"], p["prompt_tokens"], p["cost_usd"], price,
                datetime.now(), signal_id,
            ))

        conn.commit()
        cur.close()
        conn.close()
        log.info("saved to swing_results_pure_baseline", extra={"n": len(picks)})
    except Exception as e:
        log.error("DB save failed", extra={"error": str(e)})


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--pool-size", type=int, default=100)
    args = parser.parse_args()

    picks = run_pure_baseline(save_db=args.save_db, ticker_pool_size=args.pool_size)

    print(f"\n{'='*55}")
    print("PURE BASELINE  (zero structured input)")
    print(f"{'='*55}")
    total_cost = sum(p["cost_usd"] for p in picks)
    for p in picks:
        print(f"  {p['ticker']:<6} {p['signal']:<8} conf={p['confidence']:>3}%  {p['reason'][:60]}")
    print(f"\n  Total cost : ${total_cost:.4f}  |  {len(picks)} picks")
    if args.save_db:
        print("  Saved to swing_results_pure_baseline.")
