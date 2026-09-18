"""No-budget tests for the frozen comparison and orchestration accounting."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from sap_rl_lab import kl_brake_experiment as exp
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

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated, self.status = True, -15

        def wait(self, timeout):
            return self.status

    failed, peer = Child(111, 1), Child(112, None)
    children = iter([failed, peer])
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *a, **kw: next(children))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        exp.batch(tmp_path, "arm", ["control", "kl"], float("inf"))
    assert peer.terminated and not failed.terminated


def test_plan_budgets_and_seeds_do_not_overlap():
    assert exp.STEPS * 6 == 12_582_912
    assert exp.STEPS // exp.INTERVAL + 1 == 9
    assert exp.TEST_SEED > exp.VAL_SEED + 100


def test_summary_handles_missing_eligible_without_test_reselection(monkeypatch, tmp_path):
    models = {
        f"{arm}-seed{s}-final": {"sha256": "weights"}
        for s in exp.SEEDS
        for arm in ("control", "kl")
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
        lambda output: ({"limitations": "fixture"}, {"models": models, "telemetry": {}}),
    )
    monkeypatch.setattr(exp, "require_reloads", lambda *args: None)
    monkeypatch.setattr(exp, "read", lambda path: result)
    monkeypatch.setattr(exp.evaluation, "file_digest", lambda path: "selection")
    saved = {}
    monkeypatch.setattr(exp, "write_new", lambda path, value: saved.update(value))
    exp.summarize(tmp_path)
    assert saved["test_episodes"] == 10_000
    assert saved["recipe_confirmed"] is False
    assert all(p["selected"] is None for p in saved["pairs"].values())
    assert all(p["final"]["kl_minus_control_learned_success"] == 0 for p in saved["pairs"].values())
