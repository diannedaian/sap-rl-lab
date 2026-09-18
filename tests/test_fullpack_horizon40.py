from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from stable_baselines3.common.utils import obs_as_tensor

from sap_rl_lab.fullpack.actions import Action, ActionKind
from sap_rl_lab.fullpack.domain import BattleOutcome
from sap_rl_lab.fullpack.engine import AutoBattler, BattleResult
from sap_rl_lab.fullpack.horizon40 import configuration, transfer_model
from sap_rl_lab.fullpack.imitation import fresh_model, make_env
from sap_rl_lab.fullpack.recipe import configuration as original_configuration


@pytest.mark.parametrize("tier", [5, 6])
def test_new_horizon_does_not_modify_old_contracts_or_shop_limits(tier):
    old, new = original_configuration(tier).environment(), configuration(tier).environment()
    assert old[1].max_turns == 30 and new[1].max_turns == 40
    assert new[1] == replace(old[1], max_turns=40)
    assert new[1].max_actions_per_turn == 30
    assert new[1].shop_action_limit_mode == "force_battle"
    assert old[0] == new[0]


def test_long_draw_game_continues_through_turn30_and_truncates_after40():
    catalog, config = configuration(5).environment()
    engine = AutoBattler(catalog, config)
    engine.reset(42)
    with patch(
        "sap_rl_lab.fullpack.engine.resolve_battle",
        return_value=BattleResult(BattleOutcome.DRAW, (), 0, False),
    ):
        for turn in range(1, 41):
            assert engine.state.turn == turn
            result = engine.step(Action(ActionKind.END_TURN))
            assert not result.terminated
            assert result.truncated == (turn == 40)
        assert result.info["reason"] == "turn_limit" and engine.state.turn == 41


def test_time_rescaling_preserves_policy_and_value_before_old_horizon():
    source = fresh_model(original_configuration(5, seed=9911))
    target, record = transfer_model(source, configuration(5, seed=9911))
    old_env, new_env = make_env(original_configuration(5)), make_env(configuration(5))
    try:
        source.policy.set_training_mode(False)
        target.policy.set_training_mode(False)
        old_env.reset(seed=43)
        new_env.reset(seed=43)
        assert record["turn_column_multiplier"] == 40 / 30
        for turn in range(1, 31):
            old_env.engine.state.turn = new_env.engine.state.turn = turn
            a, b = old_env._observation(), new_env._observation()
            for key in ("shop", "team"):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a["global"][1:], b["global"][1:])
            aa = obs_as_tensor({k: v[None] for k, v in a.items()}, "cpu")
            bb = obs_as_tensor({k: v[None] for k, v in b.items()}, "cpu")
            mask = old_env.action_masks()
            with torch.no_grad():
                p = source.policy.get_distribution(aa, action_masks=mask).distribution.probs
                q = target.policy.get_distribution(bb, action_masks=mask).distribution.probs
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(
                    source.policy.predict_values(aa),
                    target.policy.predict_values(bb),
                    rtol=1e-5,
                    atol=1e-7,
                )
    finally:
        for model in (source, target):
            model.get_env().close()
        old_env.close()
        new_env.close()


def test_real_40turn_ppo_update_and_reload(tmp_path):
    from sb3_contrib import MaskablePPO

    from sap_rl_lab.fullpack.evaluation import evaluate_suite
    from sap_rl_lab.fullpack.opponents import build_scripted_league
    from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig, train_guarded
    from sap_rl_lab.fullpack.recipe import LearnedFirstGuard
    from sap_rl_lab.fullpack.training import policy_digest

    config = replace(configuration(5, seed=9961), environments=2, rollout_steps=8, batch_size=8)
    catalog, game = config.environment()
    pools = {}
    for i, family in enumerate(("full_stats", "full_summon")):
        path = tmp_path / f"{family}.json"
        build_scripted_league(family, 2, 140000000 + i * 10000, catalog=catalog, config=game).save(
            path
        )
        pools[f"validation_{i}"] = str(path)
    training_path = tmp_path / "train.json"
    build_scripted_league("full_tempo", 2, 141000000, catalog=catalog, config=game).save(
        training_path
    )
    source = fresh_model(
        replace(original_configuration(5, seed=9961), environments=2, rollout_steps=8, batch_size=8)
    )
    target, binding = transfer_model(source, config)
    initial = tmp_path / "initial40.zip"
    try:
        target.save(initial)
    finally:
        source.get_env().close()
        target.get_env().close()
    config = replace(
        config,
        timesteps=32,
        evaluation_interval=16,
        output_dir=str(tmp_path / "ppo"),
        initialize_from=str(initial),
        expected_initial_policy_sha256=binding["policy_sha256"],
        opponent_leagues=(str(training_path),),
        validation_leagues=pools,
        validation_episodes=2,
        validation_seed=140050000,
    )
    done = train_guarded(
        config, GuardrailConfig(stop_on_regression=False), guard_factory=LearnedFirstGuard
    )
    assert done["actual_timesteps"] == 32 and done["stop_reason"] == "budget_complete"
    loaded = MaskablePPO.load(done["last_model"], device="cpu")
    assert loaded.sap_environment_contract["game_config"]["max_turns"] == 40
    assert loaded.target_kl == 0.01
    assert policy_digest(loaded.policy) == done["evaluations"][-1]["checkpoint"]["policy_sha256"]
    import json
    from pathlib import Path

    record = json.loads(
        Path(done["evaluations"][-1]["checkpoint"]["path"]).with_suffix(".json").read_text()
    )
    actual = evaluate_suite(loaded, pools, episodes=2, seed=140050000)
    for name in pools:
        assert (
            actual["families"][name]["episode_results"]
            == record["evaluation"]["families"][name]["episode_results"]
        )
