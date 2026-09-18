import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from sap_rl_lab import tier4_experiment as parent
from sap_rl_lab.evaluation import file_digest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("train_tier4_long3", SCRIPTS / "train_tier4_long3.py")
long = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long)


def test_same_recipe_only_changes_declared_continuation_fields(tmp_path):
    config = parent.configuration(seed=54101, timesteps=2097152)
    binding = {"path": "/example/model.zip", "policy_sha256": "example"}
    result = long.make_config(asdict(config), tmp_path, 54101, 55101, binding)
    allowed = {
        "seed",
        "timesteps",
        "output_dir",
        "initialize_from",
        "expected_initial_policy_sha256",
        "evaluation_interval",
    }
    assert {k for k, v in asdict(config).items() if asdict(result)[k] != v} <= allowed
    assert result.action_cost == 0.005 and result.learning_rate == 0.0003
    assert result.environments == 8 and result.rollout_steps == 256
    assert result.timesteps == 1048576 and result.initialize_from == binding["path"]
    assert long.STEPS * 3 == 3145728
    assert long.TEST_SEED != parent.TEST_SEED and long.TEST_SEED != parent.VAL_SEED
    assert len(set(long.TRAIN_SEEDS)) == 3 and not set(long.TRAIN_SEEDS) & set(parent.SEEDS)


def test_new_testing_uses_new_seeds_not_old_evaluation_worker(tmp_path, monkeypatch):
    model = Mock()
    p = {"models": {"a": {}}, "paths": {"test": {"test_a": "pool"}}, "test_seed": long.TEST_SEED}
    monkeypatch.setattr(long.evidence, "model_for", lambda *a: (p, {}, model))
    monkeypatch.setattr(long.evidence, "require_reloads", lambda *a: None)
    evaluate = Mock(return_value={"families": {}})
    monkeypatch.setattr(long, "evaluate_suite", evaluate)
    monkeypatch.setattr(long.evidence, "count_suite", lambda *a: None)
    (tmp_path / "evaluation").mkdir()
    long.test_model(tmp_path, "a")
    assert evaluate.call_args.kwargs["seed"] == long.TEST_SEED
    assert evaluate.call_args.kwargs["episodes"] == 500


def test_expired_budget_cannot_launch(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(long.subprocess, "Popen", launch)
    with pytest.raises(TimeoutError):
        long.batch(tmp_path, "train", parent.SEEDS, 0)
    launch.assert_not_called()


def test_no_eligible_selection_is_not_relabelled(tmp_path):
    models = {}
    assert long.evidence.bind_selection("54301", ["before", "after"], None, models) is None
    assert "54301-selected" not in models


def test_real_weight_continuation_preserves_source_and_reproduces_initial(tmp_path, monkeypatch):
    from sb3_contrib import MaskablePPO

    source = tmp_path / "source"
    parent.smoke(source)
    done = parent.read(source / "ppo/complete.json")
    cp = done["evaluations"][-1]["checkpoint"]
    manifest = parent.read(source / "ppo/manifest.json")
    root = tmp_path / "long"
    root.mkdir()
    config = replace(
        long.TrainingConfig(**manifest["training"]),
        seed=55101,
        timesteps=32,
        output_dir=str(root / "candidate-54101"),
        initialize_from=cp["path"],
        expected_initial_policy_sha256=cp["policy_sha256"],
        evaluation_interval=16,
    )
    p = {
        "arms": {
            "54101": {
                "training": asdict(config),
                "guardrails": manifest["guardrails"],
                "initial": {**cp, "source_path": cp["path"]},
            }
        }
    }
    monkeypatch.setattr(long, "checked", lambda *a: p)
    monkeypatch.setattr(long, "STEPS", 32)
    long.train_arm(root, "54101")
    assert file_digest(cp["path"]) == cp["sha256"]
    result = parent.read(root / "train-54101-complete.json")
    assert result["actual_timesteps"] == 32 and result["parent_initial_reload_exact"]
    next_manifest = parent.read(root / "candidate-54101/manifest.json")
    assert next_manifest["initialization"] == "weights_only_fresh_optimizer"
    assert next_manifest["initial_policy_sha256"] == cp["policy_sha256"]
    final = MaskablePPO.load(root / "candidate-54101/last_model.zip", device="cpu")
    assert long.policy_digest(final.policy) != cp["policy_sha256"]

    bad_root = tmp_path / "bad-long"
    bad_root.mkdir()
    p["arms"]["54101"]["training"]["output_dir"] = str(bad_root / "candidate-54101")
    expected_path = Path(cp["path"]).with_suffix(".json")
    broken = json.loads(expected_path.read_text())
    next(iter(broken["evaluation"]["families"].values()))["episode_results"][0]["wins"] = -999
    bad_record = tmp_path / "bad-reference.json"
    long.write_new(bad_record, broken)
    p["arms"]["54101"]["initial"]["source_path"] = str(bad_record.with_suffix(".zip"))
    with pytest.raises(ValueError, match="before PPO"):
        long.train_arm(bad_root, "54101")
    assert parent.read(bad_root / "candidate-54101/failed.json")["actual_timesteps"] == 0


def test_third_segment_preserves_lineage_and_fresh_testing():
    assert long.SOURCE_SEGMENT_STEPS == 1048576
    assert long.SOURCE_LINEAGE_STEPS == 4194304
    assert long.SOURCE_LINEAGE_STEPS + long.STEPS == 5242880
    assert long.TEST_SEED == 70000000
    assert long.TEST_SEED != long.prior_run.TEST_SEED
    assert not set(long.TRAIN_SEEDS) & set(long.prior_run.TRAIN_SEEDS)
    assert long.PRIOR_EVAL == long.PARENT / "evaluation"
