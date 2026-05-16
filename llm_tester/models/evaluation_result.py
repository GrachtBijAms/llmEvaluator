from dataclasses import dataclass


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    passed: bool
    details: str = ""
