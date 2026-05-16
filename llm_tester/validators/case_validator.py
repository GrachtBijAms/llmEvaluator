def validate_grounding_case(case: dict) -> None:
    required_keys = {
        "id": str,
        "input_text": str,
        "expected_output": str,
        "actual_output": str,
        "context": list,
        "expected_grounding_score": (int, float),
    }

    for key, expected_type in required_keys.items():
        if key not in case:
            raise ValueError(f"Missing required key: {key}")
        if not isinstance(case[key], expected_type):
            raise TypeError(f"Invalid type for {key}: {type(case[key]).__name__}")

    if not case["context"]:
        raise ValueError("context cannot be empty")
