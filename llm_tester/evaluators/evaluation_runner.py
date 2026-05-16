from llm_tester.evaluators.grounding import GroundingEvaluator
from llm_tester.metrics.exact_match import ExactMatchMetric
from llm_tester.models.evaluation_result import EvaluationResult
from llm_tester.models.test_case import LLMTestCase


class EvaluationRunner:
    def __init__(self, grounding_threshold: float = 1.0):
        self.grounding_evaluator = GroundingEvaluator(threshold=grounding_threshold)
        self.exact_match_metric = ExactMatchMetric()

    def evaluate(self, test_case: LLMTestCase) -> dict:
        grounding_result = self.grounding_evaluator.evaluate(test_case)
        exact_match_score = self.exact_match_metric.score(test_case)

        overall_passed = grounding_result.passed and exact_match_score == 1.0

        return {
            "grounding_score": grounding_result.score,
            "grounding_passed": grounding_result.passed,
            "exact_match_score": exact_match_score,
            "overall_passed": overall_passed,
            "details": grounding_result.details,
        }
