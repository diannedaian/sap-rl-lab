import json

import pytest

from sap_rl_lab.domain import Pet
from sap_rl_lab.generalization import arm_configs, profile_pool, validate_roles
from sap_rl_lab.opponents import SnapshotLeague


def test_paired_arms_only_change_opponents():
    base = {"seed": 8101, "action_cost": 0.005, "timesteps": 1048576}
    configs = arm_configs(base, ["a", "b", "c"], ["d", "e", "f"], "initial.zip", "abc")
    a, b = configs["scripted"].copy(), configs["historical_mix"].copy()
    assert len(a.pop("opponent_leagues")) == 3
    assert len(b.pop("opponent_leagues")) == 6
    assert a == b
    assert "initialize_from" not in base
    with pytest.raises(ValueError, match="exactly three"):
        arm_configs(base, ["a"], ["b", "c"], "initial.zip", "abc")


def test_generator_roles_separate_by_weights_not_filenames():
    roles = {
        "train": {"one": {"policy_sha256": "a"}},
        "validation": {"two": {"policy_sha256": "b"}},
    }
    validate_roles(roles, {"1907": {"policy_sha256": "c"}})
    with pytest.raises(ValueError, match="initializer"):
        validate_roles(roles, {"1907": {"policy_sha256": "a"}})
    roles["reserved_test"] = {"different_filename": {"policy_sha256": "a"}}
    with pytest.raises(ValueError, match="overlap"):
        validate_roles(roles, {})


def test_pool_profile_reports_repeats_and_fallback_without_changing_pool(tmp_path):
    league = SnapshotLeague()
    for label in ("one", "two"):
        league.add(2, [Pet("ant", 2, 3)], label)
    path = tmp_path / "pool.json"
    league.save(path)
    original = path.read_bytes()
    metadata = [
        {"label": label, "turn": 2, "wins_before_battle": 0, "lives_before_battle": 5}
        for label in ("one", "two")
    ]
    result = profile_pool(path, metadata)
    assert result["duplicate_fraction"] == 0.5
    assert result["fallback_bucket_by_turn"]["1"] is None
    assert result["fallback_bucket_by_turn"]["3"] == 2
    assert result["pet_counts_by_turn"]["2"] == {"ant": 2}
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="align"):
        profile_pool(path, metadata[:-1])
    data = json.loads(original)
    data["snapshots"][1]["label"] = "one"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        profile_pool(path, [metadata[0], metadata[0]])
