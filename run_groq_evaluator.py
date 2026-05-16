import csv
from pathlib import Path

from llm_tester.adapters.groq_adapter import GroqLLMAdapter
from llm_tester.evaluators.evaluation_runner import EvaluationRunner
from llm_tester.loaders.json_loader import load_grounding_cases
from llm_tester.models.test_case import LLMTestCase


def build_prompt(test_case):
    context_block = "\n".join(f"- {item}" for item in test_case.context)
    return f"""You are answering a question for evaluation.
Use only the provided context.
If the answer is not fully supported by the context, say you do not have enough information.

Question:
{test_case.input_text}

Context:
{context_block}

Answer briefly and factually."""
    

def main():
    adapter = GroqLLMAdapter(model="llama-3.3-70b-versatile")
    runner = EvaluationRunner(grounding_threshold=1.0)
    cases = load_grounding_cases("data/cases_groq.json")

    rows = []

    for case in cases:
        source_test_case = case["test_case"]
        prompt = build_prompt(source_test_case)
        live_output = adapter.generate(prompt)

        evaluated_case = LLMTestCase(
            input_text=source_test_case.input_text,
            expected_output=source_test_case.expected_output,
            actual_output=live_output,
            context=source_test_case.context,
        )

        result = runner.evaluate(evaluated_case)

        rows.append({
            "case_id": case["id"],
            "category": case.get("category", ""),
            "input_text": evaluated_case.input_text,
            "expected_output": evaluated_case.expected_output,
            "actual_output": evaluated_case.actual_output,
            "context_text": " | ".join(evaluated_case.context),
            "context_count": len(evaluated_case.context),
            "expected_grounding_score": case["expected_grounding_score"],
            "actual_grounding_score": result["grounding_score"],
            "expected_exact_match_score": case["expected_exact_match_score"],
            "actual_exact_match_score": result["exact_match_score"],
            "expected_overall_passed": case["expected_overall_passed"],
            "actual_overall_passed": result["overall_passed"],
            "notes": case.get("notes", ""),
            "why_expected": case.get("why_expected", ""),
            "details": result["details"],
        })

    output_path = Path("reports/groq_evaluation_report.csv")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(
        1 for row in rows
        if row["expected_overall_passed"] == row["actual_overall_passed"]
    )
    print(f"Matched expected overall result for {matched}/{len(rows)} cases")


if __name__ == "__main__":
    main()
