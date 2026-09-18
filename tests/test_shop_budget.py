import copy
import importlib.util
from dataclasses import replace

import pytest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.domain import GameConfig, Pet
from sap_rl_lab.engine import AutoBattler, InvalidAction
from sap_rl_lab.replay import ReplayRecorder, verify_replay
from sap_rl_lab.training import TrainingConfig

FREEZE = Action(ActionKind.FREEZE, source=0)
END = Action(ActionKind.END_TURN)


def engine(**kwargs):
    result = AutoBattler(config=GameConfig(shop_action_limit_mode="force_battle", **kwargs))
    result.reset(seed=17)
    result.state.team = [Pet("pig", 50, 50), Pet("fish", 10, 10)]
    return result


def test_thirtieth_action_executes_then_normal_battle_exactly_once():
    forced = engine()
    explicit = AutoBattler(config=replace(forced.config, max_actions_per_turn=31))
    explicit.reset(seed=17)
    explicit.state = copy.deepcopy(forced.state)
    for _ in range(29):
        assert not forced.step(FREEZE).info.get("battle_outcome")
        explicit.step(FREEZE)
    actual = forced.step(FREEZE)
    explicit.step(FREEZE)
    expected = explicit.step(END)
    assert forced.state.to_dict() == explicit.state.to_dict()
    assert forced.rng.getstate() == explicit.rng.getstate()
    assert actual.reward == expected.reward
    assert actual.info["battle_trace"] == expected.info["battle_trace"]
    assert actual.info["forced_end_turn"]
    assert actual.info["shop_actions_before_battle"] == 30
    assert actual.info["gold_before_battle"] == 10
    assert forced.state.turn == 2 and forced.state.actions_this_turn == 0
    assert not actual.terminated and not actual.truncated


def test_boundary_swap_is_applied_before_battle():
    forced = engine(max_actions_per_turn=1)
    explicit = AutoBattler(config=replace(forced.config, max_actions_per_turn=2))
    explicit.reset(seed=17)
    explicit.state = copy.deepcopy(forced.state)
    swap = Action(ActionKind.SWAP, source=0, target=1)
    actual = forced.step(swap)
    explicit.step(swap)
    expected = explicit.step(END)
    assert actual.info["battle_trace"] == expected.info["battle_trace"]
    assert forced.state.to_dict() == explicit.state.to_dict()


@pytest.mark.parametrize("win", [True, False])
def test_forced_battle_preserves_true_win_loss_endings(win):
    game = engine(max_actions_per_turn=1, target_wins=1, starting_lives=1)
    game.opponent_provider = (lambda *args: []) if win else (lambda *args: [Pet("pig", 50, 50)])
    game.state.team = [Pet("pig", 50, 50)] if win else []
    transition = game.step(FREEZE)
    assert transition.terminated and not transition.truncated
    assert transition.info["success"] is win
    assert transition.reward == (1 if win else -1)
    assert "reason" not in transition.info  # A forced shop end is not an episode-ending reason.
    assert not game.legal_actions()


def test_forced_battle_keeps_global_turn_limit():
    game = engine(max_actions_per_turn=1, max_turns=1)
    result = game.step(FREEZE)
    assert result.truncated and not result.terminated
    assert result.info["reason"] == "turn_limit"
    assert result.info["forced_end_turn"]


def test_invalid_action_at_boundary_is_atomic_and_does_not_force_battle():
    game = engine(max_actions_per_turn=1)
    before = copy.deepcopy(game.state.to_dict())
    with pytest.raises(InvalidAction):
        game.step(Action(ActionKind.SELL, source=4))
    assert game.state.to_dict() == before and game.last_battle is None


def test_voluntary_end_is_never_counted_as_forced():
    game = engine(max_actions_per_turn=1)
    assert "forced_end_turn" not in game.step(END).info


def test_replay_restores_budget_contract_and_supports_legacy_defaults():
    recorder = ReplayRecorder(engine(max_actions_per_turn=1), seed=17)
    recorder.step(recorder.engine.codec.encode(FREEZE))
    replay = recorder.finish()
    verify_replay(replay)
    with pytest.raises(ValueError, match="configuration"):
        verify_replay(replay, AutoBattler())
    legacy = ReplayRecorder(AutoBattler(), seed=2)
    legacy.step(0)
    replay = legacy.finish()
    replay.config.pop("shop_action_limit_mode")
    replay.config.pop("shop_pet_stats")
    verify_replay(replay)


def test_configuration_rejects_ambiguous_or_unbounded_limits():
    for values in ({"max_actions_per_turn": 0}, {"max_turns": 0}, {"shop_action_limit_mode": "x"}):
        with pytest.raises(ValueError):
            GameConfig(**values)
    with pytest.raises(ValueError, match="forfeit"):
        TrainingConfig(shop_action_limit_mode="force_battle", forfeit_on_limit=True).validate()


def test_suite_reports_forcing_per_battle_not_unweighted_family_rate(monkeypatch):
    from sap_rl_lab.evaluation import evaluate_suite

    def fake(policy, **kwargs):
        battles = 100 if kwargs["opponent_league"] == "a" else 1
        return {
            **{
                key: 0
                for key in (
                    "success_rate",
                    "mean_return",
                    "mean_wins",
                    "truncation_rate",
                    "mean_episode_actions",
                    "success_without_forcing_rate",
                )
            },
            "battle_counts": {"win": battles},
            "forced_end_turns": 1,
            "forced_end_turn_rate": 1 / battles,
            "forced_episode_rate": 1,
            "environment_contract": {"schema_version": 1},
        }

    monkeypatch.setattr("sap_rl_lab.evaluation.evaluate_policy", fake)
    suite = evaluate_suite("greedy", {"a": "a", "b": "b"}, episodes=1, seed=1)
    assert suite["forced_end_turns"] == 2
    assert suite["forced_end_turn_rate"] == pytest.approx(2 / 101)
    assert suite["forced_episode_rate"] == 1


RL = importlib.util.find_spec("sb3_contrib") is not None


@pytest.mark.skipif(not RL, reason="RL extra is not installed")
def test_reward_and_vector_reset_semantics():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sap_rl_lab.env import SapAutoBattlerEnv

    env = SapAutoBattlerEnv(
        config=GameConfig(shop_action_limit_mode="force_battle", max_actions_per_turn=1),
        opponent_provider=lambda *args: [],
        swap_cost=0.005,
    )
    vec = DummyVecEnv([lambda: env])
    vec.seed(1)
    vec.reset()
    env.engine.state.team = [Pet("pig", 5, 5), Pet("fish", 3, 3)]
    action = env.engine.codec.encode(Action(ActionKind.SWAP, 0, 1))
    obs, rewards, dones, infos = vec.step([action])
    assert not dones[0] and "terminal_observation" not in infos[0]
    assert not infos[0]["TimeLimit.truncated"]
    assert rewards[0] == pytest.approx(0.995)
    assert infos[0]["game_reward"] == 1 and infos[0]["episode_actions"] == 1
    assert obs["global"][0, 4] == 0
    vec.close()


@pytest.mark.skipif(not RL, reason="RL extra is not installed")
def test_evaluation_exposes_system_assistance_and_correct_prebattle_stats():
    import numpy as np

    from sap_rl_lab.env import SapAutoBattlerEnv
    from sap_rl_lab.evaluation import evaluate_policy

    env = SapAutoBattlerEnv()
    action_id = env.engine.codec.encode(FREEZE)

    class FreezeOnly:
        def predict(self, observations, **kwargs):
            return np.full(len(observations["global"]), action_id), None

    result = evaluate_policy(
        FreezeOnly(),
        episodes=3,
        batch_size=2,
        env_kwargs={
            "config": GameConfig(shop_action_limit_mode="force_battle", max_actions_per_turn=2)
        },
    )
    assert result["truncation_rate"] == 0
    assert result["forced_episode_rate"] == result["forced_end_turn_rate"] == 1
    assert result["success_without_forcing_rate"] == 0
    assert result["forced_end_turns"] == 15
    assert result["mean_unspent_gold_per_battle"] == 10
    assert result["mean_empty_slots_per_battle"] == 5
    assert all(row["action_counts"] == {"toggle_freeze": 10} for row in result["episode_results"])
    assert all(row["reason"] == "lives_exhausted" for row in result["episode_results"])


@pytest.mark.skipif(not RL, reason="RL extra is not installed")
def test_new_checkpoint_and_validation_keep_environment_contract(tmp_path):
    from sb3_contrib import MaskablePPO

    from sap_rl_lab.evaluation import policy_environment_options
    from sap_rl_lab.opponents import build_spend_gold_league
    from sap_rl_lab.training import train

    pool = tmp_path / "validation.json"
    build_spend_gold_league(episodes=1, seed=77).save(pool)
    path = train(
        TrainingConfig(
            output_dir=str(tmp_path / "smoke"),
            timesteps=8,
            rollout_steps=8,
            batch_size=8,
            environments=1,
            device="cpu",
            shop_action_limit_mode="force_battle",
            max_actions_per_turn=2,
            validation_league=str(pool),
            validation_episodes=1,
        )
    )
    model = MaskablePPO.load(path, device="cpu")
    config = policy_environment_options(model)["config"]
    assert config.shop_action_limit_mode == "force_battle" and config.max_actions_per_turn == 2
    import json

    rows = json.loads((path.parent / "validation_history.json").read_text())
    assert all("forced_episode_rate" in row for row in rows)
    assert all(row["environment_contract"] == model.sap_environment_contract for row in rows)
