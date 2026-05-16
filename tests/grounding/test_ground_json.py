import pytest

from llm_tester.loaders.json_loader import load_grounding_cases
from llm_tester.metrics.grounding import GroundingMetric


CASES = load_grounding_cases("data/cases.json")


@pytest.mark.parametrize(
    "case_data",
    CASES,
    ids=[case["id"] for case in CASES]
)
def test_grounding_cases_from_json(case_data):
    metric = GroundingMetric()
    score = metric.score(case_data["test_case"])
    assert score == case_data["expected_grounding_score"]
