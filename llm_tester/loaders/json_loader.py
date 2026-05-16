import json
from pathlib import Path

from jsonschema import validate

from llm_tester.models.test_case import LLMTestCase


def load_grounding_cases(
    data_file_path: str,
    schema_file_path: str = "data/cases.schema.json"
):
    data_path = Path(data_file_path)
    schema_path = Path(schema_file_path)

    with data_path.open("r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validate(instance=raw_cases, schema=schema)

    loaded_cases = []
    for item in raw_cases:
        test_case = LLMTestCase(
            input_text=item["input_text"],
            expected_output=item["expected_output"],
            actual_output=item.get("actual_output"),
            context=item["context"]
        )

        loaded_cases.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "test_case": test_case,
            "expected_grounding_score": item.get("expected_grounding_score"),
            "expected_exact_match_score": item.get("expected_exact_match_score"),
            "expected_overall_passed": item.get("expected_overall_passed"),
            "min_grounding_score": item.get("min_grounding_score"),
            "notes": item.get("notes", ""),
            "why_expected": item.get("why_expected", "")
        })

    return loaded_cases
