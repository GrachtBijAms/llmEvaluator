import re

from llm_tester.models.test_case import LLMTestCase


class GroundingMetric:
    def __init__(self):
        pass

    def _split_claims(self, text: str) -> list[str]:
        parts = re.split(r"[.!?]+", text)
        return [part.strip() for part in parts if part.strip()]

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def score(self, test_case: LLMTestCase) -> float:
        if not test_case.context:
            return 0.0

        context_text = self._normalize(" ".join(test_case.context))
        claims = self._split_claims(test_case.actual_output)

        if not claims:
            return 0.0

        supported = 0
        for claim in claims:
            normalized_claim = self._normalize(claim)
            if normalized_claim in context_text:
                supported += 1

        return supported / len(claims)
