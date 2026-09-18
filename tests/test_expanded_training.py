"""Expanded contract, pool capture and small save/reload training checks."""

import json

import pytest

from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig
from sap_rl_lab.evaluation import evaluate_policy, policy_environment_options
from sap_rl_lab.opponents import build_scripted_league
from sap_rl_lab.training import TrainingConfig, train

CATALOG_ID = "turtle-v0.46-tier12-curriculum-v4"


def test_expanded_training_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_development"):
        TrainingConfig(catalog_id=CATALOG_ID).validate()
    c, g = TrainingConfig(catalog_id=CATALOG_ID, allow_development=True).environment()
    assert c.rules_version == 4 and g.max_shop_tier == 2 and g.max_shop_size == 9


def test_expanded_gold_observation_distinguishes_swan_rich_states():
    from sap_rl_lab.env import SapAutoBattlerEnv

    env = SapAutoBattlerEnv(load_catalog_by_id(CATALOG_ID), allow_development=True)
    env.reset(seed=1)
    values = []
    for gold in (19, 20, 21, 25, 40):
        env.engine.state.gold = gold
        observation = env._observation()
        assert env.observation_space.contains(observation)
        values.append(float(observation["global"][1]))
    assert values == sorted(set(values))


def test_saved_catalog_fingerprint_prevents_silent_rule_changes():
    class Model:
        sap_environment_contract = {
            "schema_version": 1,
            "catalog_id": CATALOG_ID,
            "game_config": {},
            "catalog_sha256": "wrong",
        }

    with pytest.raises(ValueError, match="catalog content"):
        policy_environment_options(Model())


def test_snapshot_pool_includes_forced_battles_and_correct_catalog():
    catalog = load_catalog_by_id(CATALOG_ID)
    league = build_scripted_league(
        "greedy",
        3,
        seed=55,
        catalog=catalog,
        config=GameConfig.turtle_curriculum(max_actions_per_turn=1),
    )
    # Greedy always buys while affordable; with budget one the very first
    # battle is forced, without the policy ever choosing END_TURN.
    assert league.catalog_id == CATALOG_ID and len(league._by_turn[1]) == 3
    assert all(len(s.pets) == 1 for s in league._by_turn[1])


def test_expanded_smoke_train_and_reload_raw_evaluation(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    catalog = load_catalog_by_id(CATALOG_ID)
    train_pool, val_pool = tmp_path / "train.json", tmp_path / "validation.json"
    for path, seed in ((train_pool, 41), (val_pool, 1041)):
        build_scripted_league("greedy", 3, seed=seed, catalog=catalog).save(path)
    output = tmp_path / "smoke"
    train(
        TrainingConfig(
            catalog_id=CATALOG_ID,
            allow_development=True,
            shop_action_limit_mode="force_battle",
            timesteps=32,
            environments=2,
            rollout_steps=16,
            batch_size=16,
            opponent_league=str(train_pool),
            validation_league=str(val_pool),
            validation_episodes=2,
            evaluation_interval=32,
            output_dir=str(output),
            device="cpu",
            swap_cost=0.005,
        )
    )
    model = MaskablePPO.load(output / "best_model.zip", device="cpu")
    from sap_rl_lab.stable_masking import StableMaskableMultiInputPolicy

    assert isinstance(model.policy, StableMaskableMultiInputPolicy)
    contract = model.sap_environment_contract
    assert contract["masking_revision"] == "cached-probs-clear-v1"
    assert contract["catalog_sha256"] == catalog_digest(catalog)
    assert contract["allow_development"] is True
    assert model.action_space.n == 139
    result = evaluate_policy(model, episodes=2, seed=30000, opponent_league=str(val_pool))
    history = json.loads((output / "validation_history.json").read_text())
    selected = [r for r in history if r["selected"]][-1]
    assert result["success_without_forcing_rate"] == selected["success_without_forcing_rate"]
    assert result["mean_return"] == selected["mean_return"]


@pytest.mark.parametrize("family", ["expanded_stats", "expanded_summon", "expanded_tempo"])
def test_expanded_heuristics_generate_reachable_nonempty_pools(family):
    league = build_scripted_league(family, 10, seed=34000, catalog=load_catalog_by_id(CATALOG_ID))
    assert len(league) >= 30
    assert all(0 < len(s.pets) <= 5 for bucket in league._by_turn.values() for s in bucket)
