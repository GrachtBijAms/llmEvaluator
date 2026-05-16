# tests/test_grounding_groq.py

import pytest

from llm_tester.adapters.groq_adapter import GroqLLMAdapter
from llm_tester.metrics.grounding import GroundingMetric
from llm_tester.models.test_case import LLMTestCase


@pytest.fixture(scope="module",params=[ "llama-3.3-70b-versatile"])
def adapter(request):
    return GroqLLMAdapter(model=request.param)


@pytest.fixture(scope="module")
def metric():
    return GroundingMetric()


def test_grounding_returns_zero_when_context_is_empty(adapter, metric):
    prompt = "Where was Einstein born?"
    response = adapter.generate(prompt)

    test_case = LLMTestCase(
        input_text=prompt,
        expected_output="Einstein was born in Germany.",
        actual_output=response,
        context=[]
    )

    score = metric.score(test_case)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print(f"Grounding score: {score}")

    assert score == 0.0


def test_grounding_returns_one_when_context_supports_answer(adapter, metric):
    prompt = "Where was Einstein born?"
    response = adapter.generate(prompt)

    test_case = LLMTestCase(
        input_text=prompt,
        expected_output="Einstein was born in Germany.",
        actual_output=response,
        context=["Albert Einstein was born in German Empire on March 14, 1879."]
    )

    score = metric.score(test_case)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print(f"Grounding score: {score}")

    assert score == 1.0
