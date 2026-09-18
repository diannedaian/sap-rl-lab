"""Training orchestration checks; real BC/PPO/reload is the separate smoke gate."""

from unittest.mock import Mock, patch

import pytest

from sap_rl_lab.fullpack import experiment as exp
from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig
from sap_rl_lab.fullpack.recipe import configuration


@pytest.fixture(autouse=True)
def restore_tier():
    exp.configure_tier(5)
    yield
    exp.configure_tier(5)


def suite(success=15, forced=0):
    return {
        "families": {
            name: {
                "episodes": 20,
                "episode_results": [
                    {
                        "wins": 8,
                        "success": i < success,
                        "forced_end_turns": int(i < forced),
                        "truncated": False,
                        "action_counts": {"buy_pet": 10},
                        "return": 1.0,
                    }
                    for i in range(20)
                ],
            }
            for name in (*exp.FAMILIES, "validation_a", "validation_b")
        }
    }


@pytest.mark.parametrize("tier", [5, 6])
def test_stage_recipe_matches_smoke_and_budget(tier):
    exp.configure_tier(tier)
    cfg = exp.configuration()
    assert cfg == configuration(tier)
    assert cfg.environment()[1].max_shop_tier == tier
    assert exp.GENERATOR_STEPS * 7 + exp.CANDIDATE_STEPS * 3 == 13_631_488
    assert len(set(v[0] for v in exp.GENERATORS.values()) | set(exp.SEEDS)) == 10
    assert len([v for v in exp.GENERATORS.values() if v[2] == "test"]) == 2
    assert exp.VAL_SEED != exp.TEST_SEED


def test_tier_episode_regions_and_candidate_seeds_are_disjoint():
    previous = exp.SEEDS, exp.VAL_SEED, exp.TEST_SEED, exp.SEED_OFFSET
    exp.configure_tier(6)
    assert not set(previous[0]) & set(exp.SEEDS)
    assert exp.VAL_SEED - previous[1] == 20_000_000
    assert exp.TEST_SEED - previous[2] == 20_000_000
    assert exp.SEED_OFFSET - previous[3] == 20_000_000


def test_guard_stops_two_regressions_but_never_selects_looping_model():
    guard = exp.LearnedFirstGuard(GuardrailConfig(regression_patience=2))
    assert guard.consider(suite(), "initial")["selected"]
    a = guard.consider(suite(success=19, forced=2), "looping-a")
    b = guard.consider(suite(success=19, forced=2), "looping-b")
    assert not a["stop"] and b["stop"] and guard.best["checkpoint"] == "initial"


def test_guard_allows_initial_learning_and_stops_on_actual_truncation():
    guard = exp.LearnedFirstGuard(GuardrailConfig())
    for _ in range(3):
        assert not guard.consider(suite(success=0, forced=5), "not-yet-competent")["stop"]
    broken = suite()
    broken["families"]["validation_a"]["episode_results"][0]["truncated"] = True
    assert guard.consider(broken, "broken")["stop_reason"] == "validation_truncation"


def test_workers_use_new_module_and_close_only_owned_processes(tmp_path):
    process = Mock(pid=123)
    process.poll.return_value = 0
    with patch.object(exp.subprocess, "Popen", return_value=process) as launch:
        exp.batch(tmp_path, "generator", ["train_stats"], float("inf"))
    assert launch.call_args.args[0][3] == "sap_rl_lab.fullpack.experiment"
    assert launch.call_args.kwargs["stdout"].closed
    process.terminate.assert_not_called()
    with pytest.raises(FileExistsError):
        exp.batch(tmp_path, "generator", ["train_stats"], float("inf"))


def test_pipeline_deadline_precedes_worker_launch(tmp_path):
    with patch.object(exp.subprocess, "Popen") as launch:
        with pytest.raises(TimeoutError):
            exp.batch(tmp_path, "candidate", exp.SEEDS, 0)
    launch.assert_not_called()


def test_launch_propagates_tier_and_is_exclusive(tmp_path):
    exp.configure_tier(6)
    with patch.object(exp.subprocess, "Popen", return_value=Mock(pid=123)) as launch:
        exp.launch(tmp_path / "run")
    assert launch.call_args.args[0][-2:] == ["--tier", "6"]
    with pytest.raises(FileExistsError):
        exp.launch(tmp_path / "run")
