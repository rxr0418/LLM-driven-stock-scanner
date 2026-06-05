"""
eval/judge.py - LLM-judge for premarket catalyst analysis quality.

Scores one quality dimension (signal correctness + reasoning specificity),
computes Cohen's kappa against human scores, target κ ≥ 0.6.

Usage:
  python eval/judge.py --results eval/results/eval_XXXXXX.json
  python eval/judge.py --results eval/results/eval_XXXXXX.json --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# LLM judge scoring
# ─────────────────────────────────────────────────────────────

def judge_output(result: dict, client: anthropic.Anthropic) -> int:
    """
    Score analysis quality on scale 1-3:

      3 = Excellent:
            - Correctly identifies specific catalyst type
            - Confidence proportional to catalyst strength
            - Entry AND exit timing are specific and actionable
            - Distinguishes nuances (FDA approval vs Fast Track)

      2 = Acceptable:
            - Correct signal direction
            - Reasoning is generic but not wrong
            - Timing advice is vague ("wait for confirmation")

      1 = Poor:
            - Wrong catalyst type or signal direction
            - Vague reasoning with no specific catalyst identified
            - No actionable timing advice
    """
    prompt = f"""You are a strict evaluator of premarket day trade analysis quality.
Score on scale 1-3. You MUST use all three scores — do not give everything a 3.

STRICT scoring rules:
  Score 3 ONLY if ALL of these are true:
    - Catalyst type is specific (e.g. FDA_APPROVAL not UNKNOWN)
    - Confidence is proportional (pump-and-dump should be <20%, strong approval >80%)
    - Entry timing names a specific condition (candle close, price level, time)
    - Exit timing has a specific target (time like 10:30 AM OR price % like +20%)

  Score 2 if:
    - Signal direction is correct but timing is vague ("wait for confirmation")
    - OR confidence seems miscalibrated for the catalyst type
    - OR only one of entry/exit timing is specific

  Score 1 if:
    - Catalyst type is UNKNOWN when news is available
    - OR signal is clearly wrong for the situation
    - OR both entry and exit timing are missing or completely vague

Analysis to evaluate:
  Ticker:            {result.get('ticker')}
  Signal:            {result.get('signal')}
  Catalyst type:     {result.get('catalyst_type')}
  Catalyst strength: {result.get('catalyst_strength')}
  Confidence:        {result.get('confidence')}%
  Manipulation risk: {result.get('manipulation_risk')}
  Reason:            {result.get('reason')}
  Risk:              {result.get('risk')}
  Entry timing:      {result.get('entry_timing')}
  Exit timing:       {result.get('exit_timing')}

Be strict. A score of 3 should be rare. Most analyses should score 1 or 2.
Return ONLY a single integer: 1, 2, or 3"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 10,
            messages   = [{"role": "user", "content": prompt}],
        )
        score = int(response.content[0].text.strip())
        return max(1, min(3, score))
    except Exception as e:
        print(f"  [judge] scoring failed: {e}")
        return 2


# ─────────────────────────────────────────────────────────────
# Human scoring
# ─────────────────────────────────────────────────────────────

def human_score(item: dict) -> int:
    """
    Compute human score based on signal direction correctness.

    Better proxy than pass/fail because it separates:
    - Output quality (what human_score measures)
    - Rubric correctness (what golden set pass/fail measures)

    3 = signal is in expected_facts (direction correct)
    2 = signal not in expected but analysis has substance (confidence > 50)
    1 = signal wrong AND confidence low or catalyst wrong
    """
    result   = item.get("result", {})
    case     = item.get("case", {})
    expected = case.get("expected_facts", [])
    signal   = result.get("signal", "")
    catalyst = result.get("catalyst_type", "")
    conf     = result.get("confidence", 0)

    # Signal direction correct
    if signal in expected:
        return 3

    # Catalyst type correct but signal wrong (partial credit)
    if catalyst in expected and conf >= 50:
        return 2

    # Both wrong
    return 1


# ─────────────────────────────────────────────────────────────
# Cohen's kappa
# ─────────────────────────────────────────────────────────────

def cohen_kappa(scores_a: list, scores_b: list) -> float:
    """
    Compute Cohen's kappa between two raters.
    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    assert len(scores_a) == len(scores_b), "Score lists must be same length"
    n = len(scores_a)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(scores_a, scores_b)) / n

    # Expected agreement by chance
    categories = set(scores_a + scores_b)
    p_e = sum(
        (scores_a.count(c) / n) * (scores_b.count(c) / n)
        for c in categories
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


# ─────────────────────────────────────────────────────────────
# Main judge run
# ─────────────────────────────────────────────────────────────

def run_judge(results_path: str, n_cases: int = 15, verbose: bool = False) -> dict:
    """
    Load eval results, score with LLM judge + human proxy,
    compute Cohen's kappa.
    """
    with open(results_path) as f:
        data = json.load(f)

    # Handle both old list format and new dict format
    if isinstance(data, list):
        eval_results = data
    else:
        eval_results = data.get("eval_results", [])

    # Find baseline results with per-case data
    baseline = next(
        (r for r in eval_results
         if r.get("mode") == "baseline" and r.get("results")),
        None
    )

    if not baseline:
        print("[judge] ERROR: No baseline results with per-case data found.")
        print("        Run: python eval/run_eval.py --mode baseline")
        return {}

    cases_to_judge = baseline["results"][:n_cases]
    print(f"\n[judge] Scoring {len(cases_to_judge)} cases...")
    print(f"[judge] Mode: baseline | Pass rate: {baseline['pass_rate']}%")
    print("─" * 65)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)

    llm_scores   = []
    human_scores = []

    for item in cases_to_judge:
        result = item.get("result", {})
        ticker = result.get("ticker", "?")
        passed = item.get("passed", False)

        llm   = judge_output(result, client)
        human = human_score(item)

        llm_scores.append(llm)
        human_scores.append(human)

        status = "✅" if passed else "❌"
        if verbose or llm != human:
            print(f"  {status} {ticker:<6} | LLM={llm} Human={human} "
                  f"| signal={result.get('signal','?'):<5} "
                  f"catalyst={result.get('catalyst_type','?'):<20} "
                  f"conf={result.get('confidence',0):>3}%")
            if llm != human:
                print(f"         reason: {result.get('reason','')[:80]}")

    kappa         = cohen_kappa(human_scores, llm_scores)
    avg_llm       = sum(llm_scores) / len(llm_scores)
    avg_human     = sum(human_scores) / len(human_scores)
    p_agree       = sum(a == b for a, b in zip(llm_scores, human_scores)) / len(llm_scores)

    print("\n" + "=" * 65)
    print("JUDGE RESULTS")
    print("=" * 65)
    print(f"  Cases scored:     {len(cases_to_judge)}")
    print(f"  Observed agree:   {p_agree:.1%}")
    print(f"  Avg LLM score:    {avg_llm:.2f} / 3.0")
    print(f"  Avg human score:  {avg_human:.2f} / 3.0")
    print(f"  Cohen's Kappa:    {kappa:.3f}  (target ≥ 0.6)")

    if kappa >= 0.6:
        print("  ✅ Kappa meets threshold — judge is reliable")
    elif kappa >= 0.4:
        print("  ⚠️  Moderate agreement — judge is usable but not reliable")
    else:
        print("  ❌ Low kappa — judge and human disagree significantly")
        print("     Common cause: human proxy (signal direction) vs LLM judge")
        print("     (output quality) measure different things")
    print("=" * 65)

    result_out = {
        "kappa":         round(kappa, 3),
        "avg_llm_score": round(avg_llm, 2),
        "avg_human_score": round(avg_human, 2),
        "observed_agreement": round(p_agree, 3),
        "n_cases":       len(cases_to_judge),
        "llm_scores":    llm_scores,
        "human_scores":  human_scores,
    }

    # Save judge results
    out_path = Path(results_path).parent / f"judge_{Path(results_path).stem}.json"
    with open(out_path, "w") as f:
        json.dump(result_out, f, indent=2)
    print(f"\n[judge] Results saved to {out_path}")

    return result_out


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge for catalyst analysis")
    parser.add_argument("--results", required=True, help="Path to eval results JSON")
    parser.add_argument("--n",       type=int, default=15, help="Number of cases to judge")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    run_judge(args.results, n_cases=args.n, verbose=args.verbose)