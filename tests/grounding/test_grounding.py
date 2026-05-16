import pytest

from llm_tester.adapters.mock_adapter import MockLLMAdapter
from llm_tester.metrics.exact_match import ExactMatchMetric
from llm_tester.metrics.grounding import GroundingMetric
from llm_tester.models.test_case import LLMTestCase
from llm_tester.evaluators.grounding import GroundingEvaluator

# Test cases for MockLLMAdapter, ExactMatchMetric, GroundingMetric, and GroundingEvaluator
def test_mock_adapter_returns_response():
    adapter = MockLLMAdapter()
    result = adapter.generate("What is Selenium?")
    assert "MOCK_RESPONSE" in result

# ExactMatchMetric tests
def test_exact_match_metric():
    test_case = LLMTestCase(
        input_text="What is Selenium?",
        expected_output="Selenium is a chemical element.",
        actual_output="Selenium is a chemical element.",
    )
    metric = ExactMatchMetric()
    score = metric.score(test_case)
    assert score == 1.0

    test_case.actual_output = "Selenium is a software testing framework."
    score = metric.score(test_case)
    assert score == 0.0

def test_exact_match_metric_ignore_case():
    test_case = LLMTestCase(
        input_text="What is Selenium?",
        expected_output="Selenium is a chemical element.",
        actual_output="selenium is a chemical element.",
    )
    metric = ExactMatchMetric(ignore_case=True)
    score = metric.score(test_case)
    assert score == 1.0


def test_exact_match_metric_ignore_punctuation():
    test_case = LLMTestCase(
        input_text="What is Selenium?",
        expected_output="Selenium is a chemical element.",
        actual_output="Selenium is a chemical element!",
    )
    metric = ExactMatchMetric(ignore_punctuation=True)
    score = metric.score(test_case)
    assert score == 1.0

def test_exact_match_metric_ignore_whitespace():
    test_case = LLMTestCase(
        input_text="What is Selenium?",
        expected_output="Selenium is a chemical element.",
        actual_output="  Selenium   is a chemical element.  ",
    )
    metric = ExactMatchMetric()
    score = metric.score(test_case)
    assert score == 1.0

def test_exact_match_fails_with_different_outputs():
    test_case = LLMTestCase(
        input_text="What is Selenium?",
        expected_output="Selenium is a chemical element.",
        actual_output="Selenium is a software testing framework.",
    )
    metric = ExactMatchMetric()
    score = metric.score(test_case)
    assert score == 0.0


def test_exact_match_metric_fail():
    metric = ExactMatchMetric()

    test_case = LLMTestCase(
        input_text="What is the capital of France?",
        expected_output="Paris",
        actual_output="Lyon"
    )

    assert metric.score(test_case) == 0.0


def test_grounding_metric_all_claims_supported():
    metric = GroundingMetric()

    test_case = LLMTestCase(
        input_text="Where was Einstein born?",
        expected_output="Einstein was born in Germany.",
        actual_output="Einstein was born in Germany.",
        context=["Einstein was born in Germany in 1879."]
    )

    assert metric.score(test_case) == 1.0


def test_grounding_metric_partial_support():
    metric = GroundingMetric()

    test_case = LLMTestCase(
        input_text="Tell me about Einstein.",
        expected_output="Einstein was born in Germany. Einstein won a Nobel Prize. He developed the theory of relativity.",
        actual_output="Einstein was born in Germany. Einstein won a Nobel Prize.",
        context=["Einstein was born in Germany."]
    )

    assert metric.score(test_case) == 0.5


def test_grounding_metric_no_support():
    metric = GroundingMetric()

    test_case = LLMTestCase(
        input_text="Where was Einstein born?",
        expected_output="Einstein was born in Germany.",
        actual_output="Einstein was born in Switzerland.",
        context=["Einstein was born in Germany."]
    )

    assert metric.score(test_case) == 0.0


def test_grounding_metric_no_context():
    metric = GroundingMetric()

    test_case = LLMTestCase(
        input_text="Where was Einstein born?",
        expected_output="Einstein was born in Germany.",
        actual_output="Einstein was born in Germany.",
        context=[]
    )

    assert metric.score(test_case) == 0.0



def test_grounding_evaluator_pass():
    evaluator = GroundingEvaluator(threshold=1.0)

    test_case = LLMTestCase(
        input_text="Where was Einstein born?",
        expected_output="Einstein was born in Germany.",
        actual_output="Einstein was born in Germany.",
        context=["Einstein was born in Germany in 1879."]
    )

    result = evaluator.evaluate(test_case)

    assert result.metric_name == "grounding"
    assert result.score == 1.0
    assert result.passed is True


def test_grounding_evaluator_fail():
    evaluator = GroundingEvaluator(threshold=1.0)

    test_case = LLMTestCase(
        input_text="Where was Einstein born?",
        expected_output="Einstein was born in Germany.",
        actual_output="Einstein was born in Switzerland.",
        context=["Einstein was born in Germany in 1879."]
    )

    result = evaluator.evaluate(test_case)

    assert result.metric_name == "grounding"
    assert result.score == 0.0
    assert result.passed is False



@pytest.mark.parametrize(
    "test_case, expected_score",
    [
        (
            LLMTestCase(
                input_text="Where was Einstein born?",
                expected_output="Einstein was born in Germany.",
                actual_output="Einstein was born in Germany.",
                context=["Einstein was born in Germany in 1879."]
            ),
            1.0,
        ),
        (
            LLMTestCase(
                input_text="Where was Einstein born?",
                expected_output="Einstein was born in Germany.",
                actual_output="Einstein was born in Switzerland.",
                context=["Einstein was born in Germany in 1879."]
            ),
            0.0,
        ),
    ],
)
def test_grounding_metric_various_cases(test_case, expected_score):
    metric = GroundingMetric()
    score = metric.score(test_case)
    assert score == expected_score
