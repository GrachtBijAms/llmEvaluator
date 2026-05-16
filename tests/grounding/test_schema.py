import json
from pathlib import Path

from jsonschema import validate


def test_grounding_cases_match_schema():
    data = json.loads(Path("data/cases.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("data/cases.schema.json").read_text(encoding="utf-8"))

    validate(instance=data, schema=schema)
