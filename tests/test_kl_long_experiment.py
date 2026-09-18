"""No-budget tests for the frozen comparison and orchestration accounting."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from sap_rl_lab import kl_long_experiment as exp
from sap_rl_lab.ppo_guardrails import GuardrailConfig


def pair():
    left = {
        "training": {"output_dir": "a", "seed": 96301, "learning_rate": 3e-4},
        "guardrails": asdict(GuardrailConfig(target_kl=None, stop_on_regression=False)),
    }
    right = deepcopy(left)
    right["training"]["output_dir"] = "b"
    right["guardrails"]["target_kl"] = 0.01
    return left, right


def test_only_kl_differs():
    exp.validate_pair(*pair())


@pytest.mark.parametrize("field,value", [("seed", 1), ("learning_rate", 1e-4)])
def test_other_training_change_rejected(field, value):
    left, right = pair()
    right["training"][field] = value
    with pytest.raises(ValueError, match="more than KL"):
        exp.validate_pair(left, right)


def test_unequal_stop_policy_rejected():
    left, right = pair()
    right["guardrails"]["stop_on_regression"] = True
    with pytest.raises(ValueError):
        exp.validate_pair(left, right)


def telemetry(kl=None, actual=80):
    return [
        {
            "timesteps": i * 2048,
            "rollout_decisions": 2048,
            "target_kl": kl,
            "maximum_optimizer_steps": 80,
            "optimizer_steps": actual,
            "kl_early_stopped": actual < 80,
            "exact_rollout_kl_mean": 0.01,
            "entropy_before_update": 0.5,
        }
        for i in (1, 2)
    ]


def test_actual_optimizer_steps_not_nominal_epochs():
    result = exp.update_metrics(telemetry(0.01, 35), 0.01, steps=4096)
    assert result["optimizer_steps"] == 70
    assert result["maximum_optimizer_steps"] == 160
    assert result["early_stopped_rollouts"] == 2


def test_missing_last_update_rejected():
    with pytest.raises(ValueError, match="telemetry"):
        exp.update_metrics(telemetry()[:1], None, steps=4096)


def test_control_cannot_silently_skip_updates():
    with pytest.raises(ValueError, match="accounting"):
        exp.update_metrics(telemetry(None, 35), None, steps=4096)


def test_expired_batch_does_not_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start"))
    with pytest.raises(TimeoutError):
        exp.batch(tmp_path, "arm", ["control"], deadline=0)


def test_failed_child_stops_only_owned_peer(monkeypatch, tmp_path):
    class Child:
        def __init__(self, pid, status):
            self.pid, self.status, self.terminated = pid, status, False

        @property
        def returncode(self):
            return self.status

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated, self.status = True, -15

        def wait(self, timeout):
            return self.status

    failed, peer = Child(111, 1), Child(112, None)
    children = iter([failed, peer])
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *a, **kw: next(children))
    with pytest.raises(RuntimeError, match="durable worker logs"):
        exp.batch(tmp_path, "arm", ["control", "kl"], float("inf"))
    assert peer.terminated and not failed.terminated


def test_plan_budgets_and_seeds_do_not_overlap():
    assert exp.STEPS * 6 == 25_165_824
    assert exp.STEPS // exp.INTERVAL + 1 == 9
    assert exp.TEST_SEED > exp.VAL_SEED + 100


def test_summary_handles_missing_eligible_without_test_reselection(monkeypatch, tmp_path):
    models = {
        f"{arm}-seed{s}-{kind}": {"sha256": "weights"}
        for s in exp.SEEDS
        for arm in ("control", "kl")
        for kind in ("final", "midpoint")
    }
    models.update({f"initial-seed{s}": {"sha256": "weights"} for s in exp.SEEDS})
    models["historical-best"] = {"sha256": "weights"}
    row = {
        "success": True,
        "wins": 10,
        "action_counts": {"buy_pet": 5},
        "forced_end_turns": 0,
        "truncated": False,
        "actions": 50,
    }
    data = {
        "episodes": 1,
        "episode_results": [row],
        "success_rate": 1.0,
        "mean_unspent_gold_per_battle": 0.5,
        "action_counts": {"buy_pet": 5},
    }
    result = {
        "families": {f: data for f in ("fresh_stats", "fresh_mix")},
        "model_sha256": "weights",
        "selection_sha256": "selection",
        "forced_episode_rate": 0.0,
    }
    monkeypatch.setattr(
        exp,
        "frozen",
        lambda output: (
            {
                "limitations": "fixture",
                "arms": {f"{a}-seed{s}": {} for s in exp.SEEDS for a in ("control", "kl")},
            },
            {"models": models, "telemetry": {}},
        ),
    )
    monkeypatch.setattr(exp, "require_reloads", lambda *args: None)
    monkeypatch.setattr(exp, "read", lambda path: result)
    monkeypatch.setattr(exp.evaluation, "file_digest", lambda path: "selection")
    saved = {}
    monkeypatch.setattr(exp, "write_new", lambda path, value: saved.update(value))
    exp.summarize(tmp_path)
    assert saved["test_episodes"] == 16_000
    assert saved["recipe_confirmed"] is False
    assert all(p["selected"] is None for p in saved["pairs"].values())
    assert all(p["final"]["kl_minus_control_learned_success"] == 0 for p in saved["pairs"].values())


def test_old_experiment_constants_stay_unchanged():
    from sap_rl_lab import kl_brake_experiment as old

    assert old.STEPS == 2_097_152
    assert old.INTERVAL == 262_144
    assert (old.VAL_SEED, old.TEST_SEED) == (33_000_000, 34_000_000)
    assert exp.STEPS == 4_194_304
    assert exp.INTERVAL == 524_288
    assert len(set(exp.TRAIN_SEEDS)) == 3
    assert not set(exp.TRAIN_SEEDS) & {96301, 96302, 96303, *old.SEEDS}
    assert exp.STEPS // 2 % exp.INTERVAL == 0
    assert exp.VAL_SEED > old.TEST_SEED + 200
    assert exp.TEST_SEED > exp.VAL_SEED + 100


def test_worker_logs_and_exit_codes_survive_failed_child(monkeypatch, tmp_path):
    class Child:
        pid = 123
        returncode = 7

        def poll(self):
            return self.returncode

    seen = []

    def spawn(command, **kwargs):
        assert command[command.index("-m") + 1] == "sap_rl_lab.kl_long_experiment"
        assert kwargs["stderr"] == exp.subprocess.STDOUT
        kwargs["stdout"].write("full traceback fixture\\n")
        seen.append(kwargs["stdout"])
        return Child()

    monkeypatch.setattr(exp.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError):
        exp.batch(tmp_path, "arm", ["fixture"], float("inf"))
    assert exp.read(tmp_path / "worker-arm-fixture-exit.json") == {"returncode": 7}
    assert (tmp_path / "worker-arm-fixture.log").read_text() == "full traceback fixture\\n"
    assert seen[0].closed


def test_pipeline_failure_has_full_traceback_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "checked", lambda _: {"maximum_pipeline_seconds": 1, "arms": {}})
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(exp, "batch", fail)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        exp.pipeline(tmp_path)
    failure = exp.read(tmp_path / "pipeline_failed.json")
    assert "Traceback" in failure["traceback"]
    assert "diagnostic failure" in failure["error"]
    assert len(calls) == 1
