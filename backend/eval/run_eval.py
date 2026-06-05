"""
eval/run_eval.py - Evaluate premarket catalyst analysis across three modes.

Modes:
  baseline  : no RAG, no MCP — plain prompt only
  rag_only  : RAG injected (catalyst_stats + knowledge), no MCP
  full      : RAG + Tavily MCP + Supabase MCP (production mode)

Also runs:
  LLM-judge : scores output quality, computes Cohen's kappa
  RAGAS     : measures faithfulness of RAG-grounded outputs

Usage:
  python eval/run_eval.py                        # run all three modes
  python eval/run_eval.py --mode baseline        # run one mode
  python eval/run_eval.py --judge                # run LLM judge only
  python eval/run_eval.py --ragas                # run RAGAS only
  python eval/run_eval.py --mode full --verbose
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Load golden set
# ─────────────────────────────────────────────────────────────

def load_golden_set(path: str = "eval/golden.jsonl") -> list:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"[eval] Loaded {len(cases)} golden cases")
    return cases


# ─────────────────────────────────────────────────────────────
# Check a single case
# ─────────────────────────────────────────────────────────────

def check_case(result: dict, case: dict) -> tuple[bool, list, list]:
    result_str = json.dumps(result).lower()
    missing   = []
    forbidden = []
    for fact in case.get("expected_facts", []):
        if fact.lower() not in result_str:
            missing.append(fact)
    for fact in case.get("forbidden_facts", []):
        if fact.lower() in result_str:
            forbidden.append(fact)
    passed = len(missing) == 0 and len(forbidden) == 0
    return passed, missing, forbidden


# ─────────────────────────────────────────────────────────────
# Run one case
# ─────────────────────────────────────────────────────────────

def run_case(case: dict, mode: str, client: anthropic.Anthropic) -> dict:
    inp = case["input"]
    news_items = []
    if inp.get("news"):
        news_items = [{"headline": inp["news"], "summary": inp["news"], "source": "golden_set"}]
    from premarket.premarket_catalyst import analyze_catalyst_with_mode
    return analyze_catalyst_with_mode(
        ticker               = inp["ticker"],
        premarket_change_pct = inp["change"],
        rvol                 = inp["rvol"],
        float_shares         = inp["float_m"] * 1e6,
        market_cap           = inp["cap_m"] * 1e6,
        news_items           = news_items,
        mode                 = mode,
        client               = client,
    )


# ─────────────────────────────────────────────────────────────
# Run full eval for one mode
# ─────────────────────────────────────────────────────────────

def run_eval(mode: str, cases: list, client: anthropic.Anthropic, verbose: bool = False) -> dict:
    print(f"\n[eval] Running mode: {mode.upper()} ({len(cases)} cases)...")
    print("─" * 60)

    results  = []
    passed   = 0
    failures = []
    start    = time.time()

    for i, case in enumerate(cases):
        ticker = case["input"]["ticker"]
        tags   = case.get("tags", [])

        try:
            result = run_case(case, mode, client)
            ok, missing, forbidden_found = check_case(result, case)

            if ok:
                passed += 1
                status = "✅"
            else:
                failures.append({
                    "ticker":          ticker,
                    "tags":            tags,
                    "result":          result,
                    "missing_facts":   missing,
                    "forbidden_found": forbidden_found,
                    "expected_facts":  case["expected_facts"],
                    "forbidden_facts": case["forbidden_facts"],
                })
                status = "❌"

            if verbose or not ok:
                print(f"  {status} #{i+1:>2} {ticker:<6} "
                      f"signal={result.get('signal','?'):<5} "
                      f"catalyst={result.get('catalyst_type','?'):<20} "
                      f"conf={result.get('confidence',0):>3}%")
                if not ok:
                    if missing:
                        print(f"       missing:   {missing}")
                    if forbidden_found:
                        print(f"       forbidden: {forbidden_found}")
            else:
                print(f"  {status} #{i+1:>2} {ticker:<6}", end="\r")

            results.append({"case": case, "result": result, "passed": ok})

        except Exception as e:
            print(f"  ⚠️  #{i+1:>2} {ticker:<6} ERROR: {e}")
            failures.append({"ticker": ticker, "error": str(e)})

        time.sleep(0.5)

    elapsed   = time.time() - start
    pass_rate = passed / len(cases) * 100

    print(f"\n[eval] Mode={mode.upper()} | "
          f"Pass={passed}/{len(cases)} ({pass_rate:.0f}%) | "
          f"Time={elapsed:.0f}s")

    return {
        "mode":      mode,
        "pass_rate": round(pass_rate, 1),
        "passed":    passed,
        "total":     len(cases),
        "results":   results,
        "failures":  failures,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# LLM-judge
# ─────────────────────────────────────────────────────────────

def judge_output(result: dict, client: anthropic.Anthropic) -> int:
    """
    Score analysis quality on scale 1-3:
      3 = excellent: specific catalyst, grounded in data, clear entry/exit
      2 = acceptable: correct but generic
      1 = poor: vague, not actionable, or wrong
    """
    prompt = f"""Rate this premarket day trade analysis quality on scale 1-3.

Scoring criteria:
  3 = Excellent: correctly identifies specific catalyst type, 
      provides specific entry/exit timing, confidence matches catalyst strength
  2 = Acceptable: correct signal direction but reasoning is generic
  1 = Poor: wrong catalyst type, vague reasoning, or no actionable guidance

Analysis to score:
  Signal:            {result.get('signal')}
  Catalyst type:     {result.get('catalyst_type')}
  Catalyst strength: {result.get('catalyst_strength')}
  Confidence:        {result.get('confidence')}%
  Reason:            {result.get('reason')}
  Entry timing:      {result.get('entry_timing')}
  Exit timing:       {result.get('exit_timing')}
  Manipulation risk: {result.get('manipulation_risk')}

Return ONLY a single integer: 1, 2, or 3"""

    try:
        response = client.messages.create(
            model     = "claude-sonnet-4-6",
            max_tokens= 10,
            messages  = [{"role": "user", "content": prompt}],
        )
        score = int(response.content[0].text.strip())
        return max(1, min(3, score))
    except Exception:
        return 2


def run_llm_judge(eval_results: list, client: anthropic.Anthropic, n_cases: int = 15) -> dict:
    """
    Run LLM judge on baseline results, compute Cohen's kappa.

    Uses pass/fail from golden set as human scores proxy:
      passed case  → human score = 3
      failed case  → human score = 1
    """
    from sklearn.metrics import cohen_kappa_score

    baseline = next((r for r in eval_results if r["mode"] == "baseline"), None)
    if not baseline or not baseline.get("results"):
        print("[judge] No baseline results found. Run baseline eval first.")
        return {}

    cases_to_judge = baseline["results"][:n_cases]
    print(f"\n[judge] Scoring {len(cases_to_judge)} cases with LLM judge...")
    print("─" * 60)

    llm_scores   = []
    human_scores = []

    for item in cases_to_judge:
        result     = item.get("result", {})
        ticker     = result.get("ticker", "?")
        passed     = item.get("passed", False)
        llm_score  = judge_output(result, client)
        # Human proxy: passed golden set = 3, failed = 1
        human_score = 3 if passed else 1

        llm_scores.append(llm_score)
        human_scores.append(human_score)

        print(f"  {ticker:<6} passed={passed} | LLM={llm_score} Human={human_score} "
              f"| signal={result.get('signal')} conf={result.get('confidence')}%")

        time.sleep(0.3)

    kappa = cohen_kappa_score(human_scores, llm_scores)
    avg_llm_score = sum(llm_scores) / len(llm_scores)

    print(f"\n[judge] Cohen's Kappa: {kappa:.3f} (target ≥ 0.6)")
    print(f"[judge] Avg LLM score: {avg_llm_score:.2f} / 3.0")

    if kappa >= 0.6:
        print("[judge] ✅ Kappa meets threshold — judge is reliable")
    else:
        print("[judge] ⚠️  Kappa below threshold — judge and human disagree")

    return {
        "kappa":         round(kappa, 3),
        "avg_llm_score": round(avg_llm_score, 2),
        "n_cases":       len(cases_to_judge),
        "llm_scores":    llm_scores,
        "human_scores":  human_scores,
    }


# ─────────────────────────────────────────────────────────────
# RAGAS faithfulness
# ─────────────────────────────────────────────────────────────

def run_ragas(eval_results: list) -> dict:
    """
    Measure RAG faithfulness: does Claude's output stay grounded
    in the injected catalyst_stats context?

    Faithfulness = fraction of output claims supported by context.
    High faithfulness = Claude uses RAG data, not hallucinations.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_anthropic import ChatAnthropic
        from datasets import Dataset
    except ImportError:
        print("[ragas] Install required: pip install ragas datasets langchain-anthropic")
        return {}

    # Get rag_only results (these have RAG context injected)
    rag_result = next((r for r in eval_results if r["mode"] == "rag_only"), None)
    if not rag_result or not rag_result.get("results"):
        print("[ragas] No rag_only results found. Run rag_only eval first.")
        return {}

    # Build RAGAS dataset
    questions = []
    contexts  = []
    answers   = []

    # Get RAG context from database
    try:
        from database import get_all_catalyst_stats, get_relevant_knowledge
        rag_context = get_all_catalyst_stats() or ""
        knowledge   = get_relevant_knowledge(catalyst_type="", keywords=["fda", "earnings", "pump"]) or ""
        context_str = f"{rag_context}\n{knowledge}".strip()
    except Exception:
        context_str = "No RAG context available — catalyst_stats table is empty"

    for item in rag_result["results"][:10]:
        result = item.get("result", {})
        ticker = result.get("ticker", "?")

        question = f"Should I trade {ticker} premarket? What is the catalyst quality?"
        answer   = (
            f"Signal: {result.get('signal')}. "
            f"Catalyst: {result.get('catalyst_type')} ({result.get('catalyst_strength')}). "
            f"Reason: {result.get('reason')} "
            f"Risk: {result.get('risk')}"
        )

        questions.append(question)
        contexts.append([context_str] if context_str else ["No context available"])
        answers.append(answer)

    if not questions:
        print("[ragas] No cases to evaluate")
        return {}

    print(f"\n[ragas] Running faithfulness on {len(questions)} cases...")
    print("─" * 60)

    try:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        llm = LangchainLLMWrapper(ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=anthropic_key,
        ))
        faithfulness.llm = llm

        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer":   answers,
        })
        result  = evaluate(dataset, metrics=[faithfulness])
        score   = float(result["faithfulness"])

        print(f"[ragas] Faithfulness score: {score:.3f}")
        if score >= 0.7:
            print("[ragas] ✅ High faithfulness — Claude uses RAG context well")
        elif score >= 0.4:
            print("[ragas] ⚠️  Moderate faithfulness — some hallucination detected")
        else:
            print("[ragas] ❌ Low faithfulness — likely due to empty catalyst_stats")
            print("         Expected: RAG improves as scan_results accumulates data")

        return {"faithfulness": round(score, 3), "n_cases": len(questions)}

    except Exception as e:
        print(f"[ragas] Evaluation failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────────────────────

def print_failure_analysis(eval_results: list) -> None:
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    all_failures = {}
    for er in eval_results:
        for f in er.get("failures", []):
            ticker = f.get("ticker", "?")
            if ticker not in all_failures:
                all_failures[ticker] = []
            all_failures[ticker].append({
                "mode":           er["mode"],
                "missing":        f.get("missing_facts", []),
                "forbidden":      f.get("forbidden_found", []),
                "signal":         f.get("result", {}).get("signal", "?"),
                "catalyst":       f.get("result", {}).get("catalyst_type", "?"),
                "expected_facts": f.get("expected_facts", []),
            })

    sorted_failures = sorted(all_failures.items(), key=lambda x: -len(x[1]))
    for ticker, fails in sorted_failures[:3]:
        print(f"\n{ticker} — failed in {len(fails)} mode(s)")
        for f in fails:
            print(f"  Mode={f['mode']}: got signal={f['signal']}, "
                  f"catalyst={f['catalyst']}")
            print(f"    Expected:  {f['expected_facts']}")
            if f['missing']:
                print(f"    Missing:   {f['missing']}")
            if f['forbidden']:
                print(f"    Forbidden: {f['forbidden']}")


# ─────────────────────────────────────────────────────────────
# Experiment summary table
# ─────────────────────────────────────────────────────────────

def print_experiment_table(eval_results: list, judge_result: dict = None, ragas_result: dict = None) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT LOG (paste into README)")
    print("=" * 60)
    print(f"{'Round':<8} {'Mode':<12} {'Change':<35} {'Pass Rate':>10} {'Conclusion'}")
    print("─" * 90)

    descriptions = {
        "baseline": "No RAG, no MCP — plain prompt only",
        "rag_only": "Added RAG (catalyst_stats + knowledge)",
        "full":     "Added Tavily MCP + Supabase MCP",
    }

    for i, er in enumerate(eval_results):
        mode  = er["mode"]
        rate  = er["pass_rate"]
        desc  = descriptions.get(mode, mode)

        if i == 0:
            conclusion = "Baseline established"
        else:
            prev_rate = eval_results[i-1]["pass_rate"]
            delta     = rate - prev_rate
            if delta > 5:
                conclusion = f"↑ +{delta:.0f}pp improvement"
            elif delta < -5:
                conclusion = f"↓ {delta:.0f}pp regression"
            else:
                conclusion = f"≈ No significant change ({delta:+.0f}pp)"

        print(f"{i:<8} {mode:<12} {desc:<35} {rate:>9.0f}% {conclusion}")

    if judge_result:
        print(f"\nLLM-Judge: Cohen's Kappa = {judge_result.get('kappa', '?')} "
              f"(avg score {judge_result.get('avg_llm_score', '?')}/3.0)")

    if ragas_result:
        print(f"RAGAS Faithfulness: {ragas_result.get('faithfulness', '?'):.3f} "
              f"({'high' if ragas_result.get('faithfulness', 0) >= 0.7 else 'low — catalyst_stats empty'})")

    print("\nNote: pp = percentage points")


# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────

def save_results(eval_results: list, judge_result: dict = None, ragas_result: dict = None) -> str:
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"eval_{timestamp}.json"

    payload = {
        "eval_results":  eval_results,
        "judge_result":  judge_result,
        "ragas_result":  ragas_result,
        "timestamp":     datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[eval] Results saved to {filepath}")
    return str(filepath)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run golden set eval + LLM judge + RAGAS")
    parser.add_argument("--mode",    default="all",        help="baseline / rag_only / full / all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--golden",  default="eval/golden.jsonl")
    parser.add_argument("--judge",   action="store_true",  help="Run LLM judge (requires prior eval results)")
    parser.add_argument("--ragas",   action="store_true",  help="Run RAGAS faithfulness")
    parser.add_argument("--results", default=None,         help="Path to prior eval results JSON for judge/ragas")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load prior results if provided
    prior_results = []
    if args.results and Path(args.results).exists():
        with open(args.results) as f:
            data = json.load(f)
            # Handle both old format (list) and new format (dict with eval_results key)
            if isinstance(data, list):
                prior_results = data
            else:
                prior_results = data.get("eval_results", [])
        print(f"[eval] Loaded prior results from {args.results}")

    # Run eval modes
    eval_results = prior_results if (args.judge or args.ragas) and prior_results else []

    if not args.judge and not args.ragas:
        cases = load_golden_set(args.golden)
        modes = ["baseline", "rag_only", "full"] if args.mode == "all" else [args.mode]
        for mode in modes:
            result = run_eval(mode, cases, client, verbose=args.verbose)
            eval_results.append(result)

    # LLM judge
    judge_result = None
    if args.judge or (not args.ragas and eval_results):
        try:
            from eval.judge import run_judge
            # Save current results to temp file for judge to read
            temp_path = "eval/results/_temp_judge_input.json"
            with open(temp_path, "w") as f:
                json.dump({"eval_results": eval_results}, f)
            judge_result = run_judge(temp_path, verbose=args.verbose)
        except Exception as e:
            print(f"[judge] Failed: {e}")

    # RAGAS
    ragas_result = None
    if args.ragas or (not args.judge and eval_results):
        ragas_result = run_ragas(eval_results)

    # Print summary
    if eval_results:
        print_experiment_table(eval_results, judge_result, ragas_result)
        print_failure_analysis(eval_results)

    save_results(eval_results, judge_result, ragas_result)


if __name__ == "__main__":
    main()