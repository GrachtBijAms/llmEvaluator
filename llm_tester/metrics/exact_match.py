import re
import string

from llm_tester.models.test_case import LLMTestCase


class ExactMatchMetric:
    def __init__(self, ignore_case: bool = True, ignore_punctuation: bool = True):
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation

    def _normalize(self, text: str) -> str:
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def score(self, test_case: LLMTestCase) -> float:
        expected = self._normalize(test_case.expected_output)
        actual = self._normalize(test_case.actual_output)
        return 1.0 if expected == actual else 0.0
