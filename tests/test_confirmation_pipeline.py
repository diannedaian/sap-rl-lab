"""Ordering barriers protect unopened tests; the pipeline never trains."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "finish_expanded_confirmation",
    Path(__file__).resolve().parents[1] / "scripts/finish_expanded_confirmation.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def protocol():
    seeds = [1811, 1907, 2027]
    return {
        "candidate_arm": "action_cost",
        "seeds": seeds,
        "arms": {f"{arm}-seed{s}": {} for s in seeds for arm in ("control", "action_cost")},
    }


def test_freeze_then_all_six_reloads_then_nine_tests_then_summary():
    plan = module.stage_plan(protocol())
    assert plan[0] == [("freeze", None)] and plan[-1] == [("summarize", None)]
    assert len(plan[1]) == 6 and all(stage == "reload" for stage, _ in plan[1])
    assert len(plan[2]) == 9 and all(stage == "evaluate" for stage, _ in plan[2])
    assert {name for _, name in plan[1]} <= {name for _, name in plan[2]}
    assert all(stage != "run" for batch in plan for stage, _ in batch)


def test_missing_or_extra_paired_runs_are_rejected():
    p = protocol()
    p["arms"].pop("action_cost-seed2027")
    with pytest.raises(ValueError, match="six paired"):
        module.stage_plan(p)


def test_repeated_seeds_are_rejected():
    p = protocol()
    p["seeds"] = [1811, 1811, 2027]
    with pytest.raises(ValueError, match="distinct"):
        module.stage_plan(p)


def test_wait_requires_all_complete_and_rejects_aborted_batch(tmp_path):
    assert not module.training_ready(tmp_path, ["a", "b"])
    (tmp_path / "a_complete.json").write_text(json.dumps({"training_complete": True}))
    assert not module.training_ready(tmp_path, ["a", "b"])
    (tmp_path / "b_complete.json").write_text(json.dumps({"training_complete": True}))
    assert module.training_ready(tmp_path, ["a", "b"])
    (tmp_path / "aborted.json").write_text("{}")
    with pytest.raises(ValueError, match="aborted"):
        module.training_ready(tmp_path, ["a", "b"])


def test_false_completion_is_rejected_instead_of_waiting_forever(tmp_path):
    (tmp_path / "a_complete.json").write_text(json.dumps({"training_complete": False}))
    with pytest.raises(ValueError, match="Invalid"):
        module.training_ready(tmp_path, ["a"])


def prepared_root(tmp_path, monkeypatch, complete=True):
    p = protocol()
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    monkeypatch.setattr(module, "verify_inputs", lambda *args: None)
    if complete:
        for name in p["arms"]:
            (tmp_path / f"{name}_complete.json").write_text(json.dumps({"training_complete": True}))


def test_actual_pipeline_barriers_and_no_automatic_goal_claim(tmp_path, monkeypatch):
    prepared_root(tmp_path, monkeypatch)
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command[3])

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.finish(tmp_path, 0)
    assert calls == ["freeze"] + ["reload"] * 6 + ["evaluate"] * 9 + ["summarize"]
    report = json.loads((tmp_path / "post-training/complete.json").read_text())
    assert report == {"stages_complete": True, "goal_complete": False}
    with pytest.raises(FileExistsError):
        module.finish(tmp_path, 0)
    assert len(calls) == 17


def test_expired_wait_never_runs_any_stage_or_restarts_training(tmp_path, monkeypatch):
    prepared_root(tmp_path, monkeypatch, complete=False)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(TimeoutError, match="no training is restarted"):
        module.finish(tmp_path, 0)
    assert calls == []
    assert (tmp_path / "post-training/failed.json").exists()
    assert not (tmp_path / "post-training/complete.json").exists()


def test_freeze_failure_stops_before_reload_and_test(tmp_path, monkeypatch):
    prepared_root(tmp_path, monkeypatch)
    calls = []

    def fail(command, **kwargs):
        calls.append(command[3])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.finish(tmp_path, 0)
    assert calls == ["freeze"]
    assert (tmp_path / "post-training/failed.json").exists()
    assert not (tmp_path / "post-training/complete.json").exists()
