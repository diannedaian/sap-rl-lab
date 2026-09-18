from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from sap_rl_lab import tier4_experiment as exp
from sap_rl_lab.ppo_guardrails import GuardrailConfig


def suite(script_success=18, learned_success=10, forced=0):
    families = {}
    for name in (*exp.FAMILIES, "validation_a", "validation_b"):
        successes = learned_success if name.startswith("validation_") else script_success
        rows = [
            {
                "wins": 8,
                "success": i < successes,
                "forced_end_turns": int(i < forced),
                "truncated": False,
                "action_counts": {"buy_pet": 10},
                "return": 1.0,
            }
            for i in range(20)
        ]
        families[name] = {"episodes": 20, "episode_results": rows}
    return {"families": families}


def test_learned_first_selection_despite_lower_macro_score():
    guard = exp.LearnedFirstGuard(GuardrailConfig(stop_on_regression=False))
    assert guard.consider(suite(), "initial")["selected"]
    assert guard.consider(suite(script_success=17, learned_success=11), "new")["selected"]
    assert guard.best["checkpoint"] == "new"


def test_learned_first_still_enforces_reliability_and_family_regression():
    guard = exp.LearnedFirstGuard(GuardrailConfig(stop_on_regression=False))
    guard.consider(suite(), "initial")
    assert not guard.consider(suite(learned_success=18, forced=2), "looping")["selected"]
    assert not guard.consider(suite(script_success=12, learned_success=18), "regression")[
        "selected"
    ]
    assert guard.best["checkpoint"] == "initial"


def test_selector_refuses_test_or_missing_learned_families():
    guard = exp.LearnedFirstGuard(GuardrailConfig(stop_on_regression=False))
    data = deepcopy(suite())
    data["families"]["test_a"] = data["families"].pop("validation_a")
    with pytest.raises(ValueError, match="exactly two"):
        guard.consider(data, "bad")


def test_frozen_recipe_budget_and_generator_separation():
    cfg = exp.configuration()
    cfg.validate()
    assert cfg.action_cost == 0.005 and cfg.swap_cost == 0
    assert cfg.shop_action_limit_mode == "force_battle" and cfg.max_actions_per_turn == 30
    assert cfg.environments * cfg.rollout_steps == 2048
    assert cfg.learning_rate == 3e-4 and cfg.gamma == 1 and cfg.entropy_coefficient == 0
    assert exp.GENERATOR_STEPS * 7 + exp.CANDIDATE_STEPS * 3 == 13_631_488
    seeds = [v[0] for v in exp.GENERATORS.values()]
    assert len(set(seeds + list(exp.SEEDS))) == 10
    assert exp.VAL_SEED != exp.TEST_SEED
    assert [v[2] for v in exp.GENERATORS.values()].count("test") == 2
    assert cfg.environment()[1].max_shop_tier == 4


def test_batch_uses_real_module_and_closes_only_owned_children(tmp_path):
    process = Mock(pid=123)
    process.poll.return_value = 0
    with patch.object(exp.subprocess, "Popen", return_value=process) as launch:
        exp.batch(tmp_path, "generator", ["train_stats"], float("inf"))
    assert launch.call_args.args[0][3] == "sap_rl_lab.tier4_experiment"
    assert launch.call_args.kwargs["stdout"].closed
    process.terminate.assert_not_called()
    with pytest.raises(FileExistsError):
        exp.batch(tmp_path, "generator", ["train_stats"], float("inf"))


def test_batch_timeout_stops_before_launch(tmp_path):
    with patch.object(exp.subprocess, "Popen") as launch:
        with pytest.raises(TimeoutError):
            exp.batch(tmp_path, "candidate", exp.SEEDS, 0)
    launch.assert_not_called()


def test_v6_bc_ppo_reload_and_learned_snapshot_smoke(tmp_path):
    destination = tmp_path / "smoke"
    exp.smoke(destination)
    summary = exp.read(destination / "summary.json")
    assert summary["ppo_decisions"] == 32 and summary["exact_reload"]
    assert summary["parameters"] == 285196
    assert summary["learned_snapshots"] > 0
    with pytest.raises(FileExistsError):
        exp.smoke(destination)
