import pytest

from llm_tester.evaluators.evaluation_runner import EvaluationRunner
from llm_tester.loaders.json_loader import load_grounding_cases


CASES = load_grounding_cases("data/cases.json")


@pytest.mark.parametrize(
    "case_data",
    CASES,
    ids=[case["id"] for case in CASES]
)
def test_evaluation_runner_matches_expected_scores(case_data):
    runner = EvaluationRunner()
    result = runner.evaluate(case_data["test_case"])

    assert result["grounding_score"] == case_data["expected_grounding_score"]
    assert result["exact_match_score"] == case_data["expected_exact_match_score"]
    assert result["overall_passed"] == case_data["expected_overall_passed"]
