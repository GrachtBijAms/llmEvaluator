from llm_tester.adapters.groq_adapter import GroqLLMAdapter
from llm_tester.metrics.grounding import GroundingMetric
from llm_tester.models.test_case import LLMTestCase





def main():
    adapter = GroqLLMAdapter()
    prompt = "Where was Einstein born?"
    response = adapter.generate(prompt)

    metric = GroundingMetric()

    test_case = LLMTestCase(
        input_text=prompt,
        expected_output="Einstein was born in Germany.",
        actual_output=response,
        context=["Einstein was born in Germany in 1879."]
    )

    score = metric.score(test_case)

    print("Prompt:", prompt)
    print("Response:", response)
    print("Grounding score:", score)

    assert score == 0.0


if __name__ == "__main__":
    main()
