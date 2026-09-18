import importlib
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_midgame_confirmation")


def result():
    rows = [
        {
            "seed": 10,
            "success": True,
            "wins": 10,
            "forced_end_turns": 0,
            "truncated": False,
            "actions": 12,
            "battles": 10,
            "return": 10,
            "action_counts": {"end_turn": 10, "buy_pet": 2},
        },
        {
            "seed": 11,
            "success": False,
            "wins": 0,
            "forced_end_turns": 2,
            "truncated": False,
            "actions": 65,
            "battles": 5,
            "return": -5,
            "action_counts": {"end_turn": 3, "toggle_freeze": 62},
        },
    ]
    return {
        "episode_results": rows,
        "success_rate": 0.5,
        "forced_episode_rate": 0.5,
        "forced_end_turn_rate": 2 / 15,
        "truncation_rate": 0,
        "success_without_forcing_rate": 0.5,
        "mean_wins": 5,
        "mean_return": 2.5,
        "mean_episode_actions": 38.5,
    }


def test_integer_audit_distinguishes_episodes_from_events(audit):
    counts = audit.exact_family_counts(result(), 2, 10)
    assert counts["forced_episodes"] == 1
    assert counts["forced_events"] == 2
    assert counts["battles"] == 15


def test_integer_audit_rejects_wrong_aggregate(audit):
    data = result()
    data["forced_episode_rate"] = 2 / 15
    with pytest.raises(ValueError, match="forced_episode_rate"):
        audit.exact_family_counts(data, 2, 10)


def test_integer_audit_rejects_missing_rows(audit):
    with pytest.raises(ValueError, match="Missing"):
        audit.exact_family_counts(result(), 3, 10)


def test_replay_selection_is_bounded_and_deduplicated(audit):
    rows = [
        {"seed": i, "success": i % 2 == 0, "forced_end_turns": int(i == 18), "truncated": False}
        for i in range(30)
    ]
    selected = audit.select_rows(rows[::-1])
    assert [r["seed"] for r in selected] == [0, 1, 2, 3, 5, 7, 18]
    assert len(audit.select_rows([])) == 0
    with pytest.raises(ValueError, match="Duplicate"):
        audit.select_rows(rows + rows)


def test_contract_checks_rules_not_permission_metadata(audit):
    actual = {
        "schema_version": 1,
        "catalog_id": "v5",
        "catalog_sha256": "abc",
        "game_config": {"max_shop_tier": 3},
    }
    saved = {**actual, "allow_development": True, "masking_revision": "cached-probs-clear-v1"}
    audit.check_environment_contract(actual, saved)
    with pytest.raises(ValueError, match="different environment"):
        audit.check_environment_contract({**actual, "game_config": {"max_shop_tier": 2}}, saved)


def test_archived_reload_accepts_learned_pool_layout(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("reload_midgame_archive")
    pool = tmp_path / "data/validation_balanced/pool.json"
    pool.parent.mkdir(parents=True)
    pool.write_text("{}")
    protocol = {
        "paths": {"validation": {"validation_balanced": str(pool)}},
        "data_sha256": {"data/validation_balanced/pool.json": module.file_digest(pool)},
    }
    assert module.validation_paths(tmp_path, protocol) == {"validation_balanced": str(pool)}
    pool.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Validation pool changed"):
        module.validation_paths(tmp_path, protocol)


def test_archived_reload_rejects_pool_outside_data(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("reload_midgame_archive")
    pool = tmp_path / "not-data.json"
    pool.write_text("{}")
    with pytest.raises(ValueError, match="data directory"):
        module.validation_paths(tmp_path, {"paths": {"validation": {"test": str(pool)}}})


def test_diagnostic_control_uses_most_forcing_but_never_mixed_models(audit):
    counts = {
        "historical_mix-seed17101": {"a": {"forced_episodes": 999}},
        "scripted-seed17201": {"a": {"forced_episodes": 2}},
        "scripted-seed17101": {"a": {"forced_episodes": 2}},
        "scripted-seed17301": {"a": {"forced_episodes": 1}},
    }
    assert audit.control_to_inspect(counts) == "scripted-seed17101"
    counts["scripted-seed17301"]["b"] = {"forced_episodes": 5}
    assert audit.control_to_inspect(counts) == "scripted-seed17301"
    assert audit.control_to_inspect({}) is None
    assert audit.control_to_inspect({"scripted-seed17101": {"a": {"forced_episodes": 0}}}) is None


def test_zero_success_diagnostics_require_all_families_to_fail(audit):
    counts = {
        "scripted-seed17301": {"a": {"successes": 0}, "b": {"successes": 0}},
        "historical_mix-seed17301": {"a": {"successes": 0}},
        "scripted-seed17101": {"a": {"successes": 0}, "b": {"successes": 1}},
        "missing": {},
    }
    assert audit.zero_success_models(counts) == ["historical_mix-seed17301", "scripted-seed17301"]
    assert audit.zero_success_models({}) == []
