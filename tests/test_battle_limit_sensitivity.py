import runpy
from pathlib import Path

import pytest

HELPER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/audit_battle_limit_sensitivity.py")
)


def result(rows):
    return {"families": {"example": {"episode_results": rows}}}


def test_sensitivity_compares_whole_rows_not_just_success():
    a = result([{"seed": 1, "success": True, "actions": 20}])
    assert HELPER["changed_episode_seeds"](a, a) == {"example": []}
    b = result([{"seed": 1, "success": True, "actions": 21}])
    assert HELPER["changed_episode_seeds"](a, b) == {"example": [1]}


def test_sensitivity_refuses_unpaired_inputs():
    a = result([{"seed": 1}])
    with pytest.raises(ValueError, match="seeds"):
        HELPER["changed_episode_seeds"](a, result([{"seed": 2}]))
    with pytest.raises(ValueError, match="families"):
        HELPER["changed_episode_seeds"](a, {"families": {}})
