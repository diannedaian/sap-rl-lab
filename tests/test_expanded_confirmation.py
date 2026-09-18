import json

import pytest

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import (
    compare_reload,
    delivery_choice,
    freeze,
    latest_selected,
    save_new,
)


def test_selection_uses_unassisted_validation_not_test_or_raw_success():
    choices = {
        f"action_cost-seed{s}": {
            "arm": "action_cost",
            "seed": s,
            "validation_unassisted_success": 0.9,
            "validation_return": 7.0,
            "test_success": 1.0 if s == 37 else 0.0,
        }
        for s in (11, 23, 37)
    }
    assert delivery_choice(choices, "action_cost") == "action_cost-seed11"
    choices["action_cost-seed23"]["validation_return"] = 7.1
    assert delivery_choice(choices, "action_cost") == "action_cost-seed23"
    choices["action_cost-seed37"]["validation_unassisted_success"] = 0.91
    assert delivery_choice(choices, "action_cost") == "action_cost-seed37"


def test_latest_selected_preserves_pre_final_update_best():
    first = {"selected": True, "timesteps": 32, "score": 1}
    final = {"selected": False, "timesteps": 32, "score": 0}
    assert latest_selected([first, final]) == first
    with pytest.raises(ValueError):
        latest_selected([final])


def test_reload_compares_full_rows_not_only_headline_success():
    expected = {
        "families": {"stats": {"episode_results": [{"seed": 1, "success": True, "actions": 30}]}}
    }
    compare_reload(expected, expected)
    actual = {
        "families": {"stats": {"episode_results": [{"seed": 1, "success": True, "actions": 31}]}}
    }
    with pytest.raises(ValueError):
        compare_reload(actual, expected)
    with pytest.raises(ValueError):
        compare_reload({"families": {}}, expected)


def stage(tmp_path):
    protocol = {
        "seeds": [11, 23, 37],
        "candidate_arm": "action_cost",
        "source_files_sha256": {},
        "data_sha256": {},
        "base_config": {"timesteps": 32},
        "arms": {},
    }
    for seed in protocol["seeds"]:
        for arm in ("control", "action_cost"):
            name = f"{arm}-seed{seed}"
            folder = tmp_path / name
            folder.mkdir()
            model = folder / "best_model.zip"
            model.write_bytes(f"fake checkpoint, no inference: {name}".encode())
            protocol["arms"][name] = {"seed": seed}
            save_new(
                tmp_path / f"{name}_complete.json",
                {
                    "training_complete": True,
                    "best_model_sha256": file_digest(str(model)),
                },
            )
            save_new(
                folder / "run_manifest.json",
                {
                    "config": {"seed": seed},
                    "initial_policy_sha256": str(seed),
                    "environment_contract": {"same": True},
                },
            )
            save_new(
                folder / "validation_history.json",
                [
                    {
                        "timesteps": 32,
                        "selected": True,
                        "success_without_forcing_rate": 0.9,
                        "mean_return": 7.0,
                    }
                ],
            )
    save_new(tmp_path / "protocol.json", protocol)


def test_freeze_requires_all_runs_and_writes_once_without_opening_test(tmp_path):
    stage(tmp_path)
    freeze(tmp_path)
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert len(selection["models"]) == 6
    assert selection["delivery"] == "action_cost-seed11"
    assert not selection["held_out_evaluation_started"]
    assert not list(tmp_path.glob("evaluation-*"))
    with pytest.raises(FileExistsError):
        freeze(tmp_path)


@pytest.mark.parametrize("problem", ["model", "weights", "seed", "budget", "missing"])
def test_freeze_rejects_mismatched_or_unfinished_evidence(tmp_path, problem):
    stage(tmp_path)
    folder = tmp_path / "action_cost-seed11"
    if problem == "model":
        (folder / "best_model.zip").write_bytes(b"changed")
    elif problem in {"weights", "seed"}:
        path = folder / "run_manifest.json"
        data = json.loads(path.read_text())
        if problem == "weights":
            data["initial_policy_sha256"] = "not-paired"
        else:
            data["config"]["seed"] = 42
        path.write_text(json.dumps(data))
    elif problem == "budget":
        path = folder / "validation_history.json"
        data = json.loads(path.read_text())
        data[-1]["timesteps"] = 16
        path.write_text(json.dumps(data))
    else:
        (tmp_path / "action_cost-seed11_complete.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze(tmp_path)
    assert not (tmp_path / "selection.json").exists()
