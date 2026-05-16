from llm_tester.metrics.grounding import GroundingMetric
from llm_tester.models.evaluation_result import EvaluationResult
from llm_tester.models.test_case import LLMTestCase


class GroundingEvaluator:
    def __init__(self, threshold: float = 1.0):
        self.metric = GroundingMetric()
        self.threshold = threshold

    def evaluate(self, test_case: LLMTestCase) -> EvaluationResult:
        score = self.metric.score(test_case)
        passed = score >= self.threshold

        return EvaluationResult(
            metric_name="grounding",
            score=score,
            passed=passed,
            details=f"Grounding score: {score:.2f}, threshold: {self.threshold:.2f}"
        )
