import pytest

from sap_rl_lab import exploration_pilot as ep
from sap_rl_lab.expanded_confirmation import read, save_new


def test_entropy_pilot_changes_one_setting_and_has_bounded_seeds():
    assert ep.SEEDS == (17301, 17401, 17501)
    assert ep.ENTROPIES == {"entropy0": 0.0, "entropy001": 0.01}
    assert ep.STEPS == 1_048_576
    cfg = ep.new_config(timesteps=ep.STEPS)
    assert cfg["action_cost"] == 0.005 and cfg["initialize_from"] == ""
    assert cfg["shop_action_limit_mode"] == "force_battle"


def test_pilot_metrics_separate_empty_losses_from_ordinary_losses():
    rows = [
        {
            "success": False,
            "wins": 0,
            "actions": 6,
            "forced_end_turns": 0,
            "truncated": False,
            "action_counts": {"end_turn": 6},
        },
        {
            "success": False,
            "wins": 0,
            "actions": 50,
            "forced_end_turns": 2,
            "truncated": False,
            "action_counts": {"buy_pet": 5, "merge": 2},
        },
        {
            "success": True,
            "wins": 10,
            "actions": 120,
            "forced_end_turns": 0,
            "truncated": False,
            "action_counts": {"buy_pet": 8},
        },
    ]
    result = ep.metrics({"families": {"a": {"episodes": 3, "episode_results": rows}}})
    assert result["no_purchase_zero_win_loss_rate"] == 1 / 3
    assert result["worst_family_forced_episode_rate"] == 1 / 3
    assert result["mean_wins"] == 10 / 3
    assert result["families"]["a"]["mean_pet_purchases_including_merges"] == 5


def test_pilot_metrics_reject_missing_records():
    with pytest.raises(ValueError, match="Missing"):
        ep.metrics({"families": {"a": {"episodes": 10, "episode_results": []}}})
    with pytest.raises(ValueError, match="No evaluation"):
        ep.metrics({"families": {}})


def test_pilot_batch_never_exceeds_two_workers(monkeypatch, tmp_path):
    save_new(tmp_path / "protocol.json", {"arms": dict.fromkeys(range(6))})
    children = []

    class Child:
        def __init__(self, cmd, cwd):
            assert sum(p.returncode is None for p in children) < 2
            self.returncode = None
            children.append(self)

        def poll(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(ep.subprocess, "Popen", Child)
    ep.batch(tmp_path)
    assert len(children) == 6


def test_failed_pilot_does_not_automatically_retry(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ep, "prepare", lambda p: None)

    def fail(p):
        calls.append("batch")
        raise RuntimeError("worker failed")

    monkeypatch.setattr(ep, "batch", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        ep.pipeline(tmp_path)
    assert calls == ["batch"]
    assert read(tmp_path / "pipeline_failed.json")["automatic_retry"] is False
