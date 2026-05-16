from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMTestCase:
    input_text: str
    expected_output: str
    actual_output: str
    context: List[str] = field(default_factory=list)
