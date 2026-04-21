import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "tools" / "schemas"


@pytest.fixture(scope="module")
def run_validator():
    return Draft7Validator(json.loads((SCHEMAS / "run.schema.json").read_text()))


@pytest.fixture(scope="module")
def profile_validator():
    return Draft7Validator(json.loads((SCHEMAS / "profile.schema.json").read_text()))


def test_every_run_matches_schema(runs_raw, run_validator):
    assert runs_raw, "no run JSON files found under logs/runs/"
    errors = []
    for run in runs_raw:
        for err in run_validator.iter_errors(run):
            errors.append(f"{run.get('run_id', '?')}: {err.message}")
    assert not errors, "schema violations:\n  " + "\n  ".join(errors)


def test_every_profile_matches_schema(profiles_raw, profile_validator):
    errors = []
    for p in profiles_raw:
        for err in profile_validator.iter_errors(p):
            errors.append(f"{p.get('profile_id', '?')}: {err.message}")
    assert not errors, "schema violations:\n  " + "\n  ".join(errors)
