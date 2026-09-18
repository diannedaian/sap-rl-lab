"""No training budget: target boundaries, immutable protocol, failure behavior."""

from copy import deepcopy

import pytest

from sap_rl_lab import kl_80_experiment as exp
from sap_rl_lab import kl_long_experiment as old


def results(successes=800):
    row = {
        "learned_successes": successes,
        "learned_episodes": 1000,
        "forced_episode_rate": 0.002,
        "worst_family_forced_episode_rate": 0.01,
        "truncations": 0,
        "no_purchase_zero_win_losses": 0,
        "swap_freeze_actions_per_battle": 2.0,
    }
    return {
        f"kl-seed{s}-{k}": deepcopy(row)
        for s in exp.SEEDS
        for k in ("initial", "final", "selected")
    }


def test_exact_80_passes():
    got = exp.acceptance(results())
    assert got["score_target_met"] and got["eligible_after_evidence_audit"]
    assert got["selected_learned_successes"] == 2400


def test_one_episode_short_cannot_round_up():
    rows = results()
    rows["kl-seed17301-selected"]["learned_successes"] -= 1
    assert not exp.acceptance(rows)["score_target_met"]


def test_cannot_use_best_final_to_pass():
    rows = results(790)
    for s in exp.SEEDS:
        rows[f"kl-seed{s}-final"]["learned_successes"] = 1000
    assert not exp.acceptance(rows)["score_target_met"]


def test_missing_seed_fails_closed():
    rows = results(1000)
    del rows["kl-seed204101-selected"]
    assert not exp.acceptance(rows)["eligible_after_evidence_audit"]


def test_missing_test_rows_fails_closed():
    rows = results()
    rows["kl-seed17301-selected"]["learned_episodes"] = 999
    assert not exp.acceptance(rows)["score_target_met"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("forced_episode_rate", 0.01001),
        ("worst_family_forced_episode_rate", 0.05001),
        ("truncations", 1),
        ("no_purchase_zero_win_losses", 1),
        ("swap_freeze_actions_per_battle", 3.00001),
    ],
)
def test_reliability_regression_blocks_expansion(field, value):
    rows = results()
    rows["kl-seed17301-selected"][field] = value
    got = exp.acceptance(rows)
    assert got["score_target_met"] and not got["eligible_after_evidence_audit"]


def test_behavior_threshold_uses_same_seed_initial():
    rows = results()
    rows["kl-seed17301-initial"]["swap_freeze_actions_per_battle"] = 10
    rows["kl-seed17301-selected"]["swap_freeze_actions_per_battle"] = 12
    assert exp.acceptance(rows)["eligible_after_evidence_audit"]


def test_protocol_scope_and_old_constants_preserved():
    assert exp.STEPS * 3 == 6_291_456
    assert exp.STEPS % 2048 == exp.INTERVAL % 2048 == 0
    assert exp.STEPS // exp.INTERVAL + 1 == 9
    assert exp.VAL_SEED > old.TEST_SEED + 200
    assert exp.TEST_SEED > exp.VAL_SEED + exp.VAL_EPISODES
    assert old.STEPS == 4_194_304
    assert not set(exp.TRAIN_SEEDS) & set(old.TRAIN_SEEDS)


def test_proxy_normalizes_by_battles_and_counts_both_actions():
    row = {
        "success": True,
        "wins": 10,
        "forced_end_turns": 0,
        "truncated": False,
        "battles": 10,
        "actions": 30,
        "action_counts": {"swap_adjacent": 10, "freeze": 5},
    }
    result = exp.result_metrics(
        {
            "families": {
                n: {"episode_results": [deepcopy(row)]} for n in ("fresh_stats", "fresh_mix")
            }
        }
    )
    assert result["swap_freeze_actions_per_battle"] == 1.5
    assert result["learned_successes"] == 2


def test_expired_batch_does_not_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(TimeoutError):
        exp.batch(tmp_path, "arm", ["kl-seed17301"], 0)


def test_worker_failure_records_traceback_and_does_not_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "checked", lambda _: {"maximum_pipeline_seconds": 1, "arms": {}})
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("bounded failure")

    monkeypatch.setattr(exp, "batch", fail)
    with pytest.raises(RuntimeError):
        exp.pipeline(tmp_path)
    assert len(calls) == 1
    assert "Traceback" in exp.read(tmp_path / "pipeline_failed.json")["traceback"]
