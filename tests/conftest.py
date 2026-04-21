import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


@pytest.fixture(scope="session")
def runs_raw() -> list[dict]:
    out = []
    for f in sorted((LOGS / "runs").glob("*.json")):
        out.append(json.loads(f.read_text()))
    return out


@pytest.fixture(scope="session")
def profiles_raw() -> list[dict]:
    out = []
    d = LOGS / "profiles"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        out.append(json.loads(f.read_text()))
    return out
