import csv
from pathlib import Path

from llm_tester.evaluators.evaluation_runner import EvaluationRunner
from llm_tester.loaders.json_loader import load_grounding_cases


def main():
    runner = EvaluationRunner(grounding_threshold=1.0)
    cases = load_grounding_cases("data/cases.json")

    print("Loaded case IDs:", [case["id"] for case in cases])

    rows = []
    for case in cases:
        test_case = case["test_case"]
        result = runner.evaluate(test_case)

        rows.append({
            "case_id": case["id"],
            "input_text": test_case.input_text,
            "expected_output": test_case.expected_output,
            "actual_output": test_case.actual_output,
            "context_count": len(test_case.context),
            "expected_grounding_score": case["expected_grounding_score"],
            "actual_grounding_score": result["grounding_score"],
            "expected_exact_match_score": case["expected_exact_match_score"],
            "actual_exact_match_score": result["exact_match_score"],
            "expected_overall_passed": case["expected_overall_passed"],
            "actual_overall_passed": result["overall_passed"],
            "category": case.get("category", ""),
            "notes": case.get("notes", ""),
            "why_expected": case.get("why_expected", ""),
            "details": result["details"],
        })

    output_path = Path("reports/evaluation_report.csv")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    passed_count = sum(
        1 for row in rows
        if row["expected_overall_passed"] == row["actual_overall_passed"]
    )
    print(f"Matched expected overall result for {passed_count}/{len(rows)} cases")


if __name__ == "__main__":
    main()
