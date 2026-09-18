import random
from copy import deepcopy

import pytest

from sap_rl_lab import reward_cost_pilot as pilot
from sap_rl_lab.expanded_confirmation import read, save_new
from sap_rl_lab.midgame_confirmation import new_config


def test_reward_treatment_and_approved_budget():
    assert pilot.STEPS * len(pilot.SEEDS) * len(pilot.ARMS) == 12_582_912
    assert pilot.SEEDS == (17301, 203101, 203201)
    assert pilot.ARMS == {
        "all_actions": {"action_cost": 0.005, "swap_cost": 0.0},
        "swap_only": {"action_cost": 0.0, "swap_cost": 0.005},
    }
    assert len({sum(costs.values()) for costs in pilot.ARMS.values()}) == 1
    assert pilot.VAL_SEED != pilot.TEST_SEED


@pytest.fixture
def saver_case(tmp_path, monkeypatch):
    save_new(tmp_path / "run_manifest.json", {"config": {"action_cost": 0, "swap_cost": 0.005}})
    monkeypatch.setattr(pilot, "policy_digest", lambda policy: policy)

    class FakeModel:
        policy = "fixed-policy"
        num_timesteps = 123

        def save(self, path):
            with open(path, "xb") as handle:
                handle.write(b"exact-weights")

    return tmp_path, FakeModel()


def test_saver_preserves_result_and_same_step_distinct_versions(saver_case):
    folder, model = saver_case
    result, calls = {"success_rate": 0.5}, []

    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    wrapped = pilot.checkpointing_evaluator(evaluate, folder)
    assert wrapped(model, "pool", episodes=100) is result
    assert wrapped(model, "pool", episodes=100) is result
    assert len(calls) == 2
    for i in range(2):
        binding = read(folder / "validation_weights" / f"eval{i:03d}.json")
        assert binding["timesteps"] == 123
        assert binding["evaluation_file"] == f"validation_123_eval{i:03d}.json"
        assert binding["policy_sha256"] == "fixed-policy"
    assert (folder / "validation_weights/run_manifest.json").read_bytes() == (
        folder / "run_manifest.json"
    ).read_bytes()


def test_saver_refuses_overwrite(saver_case):
    folder, model = saver_case
    pilot.checkpointing_evaluator(lambda *a: {}, folder)(model)
    with pytest.raises(FileExistsError):
        pilot.checkpointing_evaluator(lambda *a: {}, folder)(model)


def test_saver_detects_policy_mutation(saver_case):
    folder, model = saver_case
    original = model.save

    def bad_save(path):
        original(path)
        model.policy = "changed"

    model.save = bad_save
    with pytest.raises(ValueError, match="changed policy or random state"):
        pilot.checkpointing_evaluator(lambda *a: {}, folder)(model)


def test_saver_detects_rng_mutation(saver_case):
    folder, model = saver_case
    original = model.save

    def bad_save(path):
        original(path)
        random.random()

    model.save = bad_save
    with pytest.raises(ValueError, match="changed policy or random state"):
        pilot.checkpointing_evaluator(lambda *a: {}, folder)(model)


def test_saver_detects_manifest_change(saver_case):
    folder, model = saver_case
    wrapped = pilot.checkpointing_evaluator(lambda *a: {}, folder)
    wrapped(model)
    (folder / "run_manifest.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="manifest changed"):
        wrapped(model)


def passing_gate_inputs():
    metrics = {
        "worst_family_no_purchase_loss_rate": 0.01,
        "worst_family_forced_episode_rate": 0.02,
        "worst_family_truncation_rate": 0,
        "mean_wins": 5,
    }
    results = {f"swap_only-seed{s}": {"final": deepcopy(metrics)} for s in pilot.SEEDS}
    pairs = {
        str(s): {
            "final": {
                "swap_only": deepcopy(metrics),
                "all_actions": deepcopy(metrics),
                "learned_difference": 0,
            }
        }
        for s in pilot.SEEDS
    }
    return results, pairs


def test_candidate_gate_requires_all_three_and_both_splits():
    results, pairs = passing_gate_inputs()
    assert all(pilot.candidate_gates(results, pairs).values())
    results["swap_only-seed17301"]["final"]["worst_family_forced_episode_rate"] = 0.020001
    assert not pilot.candidate_gates(results, pairs)["final_forcing_controlled"]
    pairs["203101"]["final"]["learned_difference"] = -0.001
    assert not pilot.candidate_gates(results, pairs)["all_three_learned_differences_nonnegative"]
    del pairs["203201"]
    with pytest.raises(KeyError):
        pilot.candidate_gates(results, pairs)


def test_mean_wins_gate_does_not_reward_empty_zero_forcing():
    results, pairs = passing_gate_inputs()
    pairs["17301"]["final"]["swap_only"].update(
        worst_family_no_purchase_loss_rate=1, worst_family_forced_episode_rate=0, mean_wins=0
    )
    gates = pilot.candidate_gates(results, pairs)
    assert gates["final_forcing_controlled"]
    assert not gates["final_empty_losses_controlled"]
    assert not gates["all_three_mean_wins_not_lower"]


def test_prepare_freezes_equal_pools_and_refuses_overwrite_or_tampering(tmp_path, monkeypatch):
    paths = {}
    for split, count in (("train", 6), ("validation", 4), ("test", 5)):
        paths[split] = {}
        for i in range(count):
            path = tmp_path / f"{split}-{i}.json"
            save_new(path, {"split": split, "i": i})
            paths[split][str(i)] = str(path)
    plan = tmp_path / "plan.md"
    plan.write_text("fixed test plan")
    parent = tmp_path / "runs/stability-confirmation-v1"
    parent.mkdir(parents=True)
    save_new(parent / "protocol.json", {"test": True})
    for name in (
        "audit_exploration_pilot",
        "audit_midgame_confirmation",
        "inspect_confirmation_delivery",
        "summarize_policy_replays",
    ):
        path = tmp_path / "scripts" / f"{name}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# frozen test helper")
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "PLAN", plan)
    monkeypatch.setattr(pilot, "source_archive", lambda output: {})
    monkeypatch.setattr(
        pilot, "verify_parent", lambda output: {"paths": paths, "base_config": new_config()}
    )
    output = tmp_path / "reward-cost"
    pilot.prepare(output)
    p = pilot.checked(output)
    assert p["training_steps_total"] == 12_582_912
    assert len(p["base_config"]["opponent_leagues"]) == 6
    assert p["base_config"]["entropy_coefficient"] == 0
    assert p["base_config"]["initialize_from"] == ""
    assert len(p["arms"]) == 6
    for seed in pilot.SEEDS:
        assert p["arms"][f"swap_only-seed{seed}"] == {"seed": seed, **pilot.ARMS["swap_only"]}
    with pytest.raises(FileExistsError):
        pilot.prepare(output)
    (output / "protocol.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="protocol changed"):
        pilot.checked(output)
