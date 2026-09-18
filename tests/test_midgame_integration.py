"""Forty-pet shop, observations, snapshots, replay and small PPO contract tests."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest
from test_midgame_components import catalog, pet

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import catalog_digest
from sap_rl_lab.domain import GameConfig, GameState, ShopItem
from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.opponents import PetSnapshot, SnapshotLeague, build_scripted_league
from sap_rl_lab.replay import ReplayRecorder, verify_replay
from sap_rl_lab.training import TrainingConfig, train


def game(**overrides):
    g = AutoBattler(catalog(), GameConfig.turtle_midgame(**overrides))
    g.reset(13)
    return g


def feed(g, food, target=0):
    spec = g.catalog.foods[food]
    g.state.shop[0] = ShopItem("food", food, spec.cost)
    g.state.gold = max(g.state.gold, spec.cost)
    return g.step(Action(ActionKind.BUY_FOOD, 0, target))


@pytest.mark.parametrize("turn,tier", [(1, 1), (3, 2), (5, 3), (7, 3), (15, 3), (30, 3)])
def test_normal_shops_and_exact_next_tier_rewards(turn, tier):
    g = game()
    g.state.turn = turn
    seen = set()
    for _ in range(100):
        g._roll_shop(free=True)
        seen.update(s.item_id for s in g.state.shop if s and s.kind == "pet")
        assert all(
            g.catalog.pets[s.item_id].tier <= tier for s in g.state.shop if s and s.kind == "pet"
        )
    assert any(g.catalog.pets[s].tier == tier for s in seen)
    g._stock_level_up_choices()
    choices = [s for s in g.state.shop if s and s.choice_group is not None]
    assert len(choices) == 2
    assert all(g.catalog.pets[s.item_id].tier == tier + 1 for s in choices)


def test_food_pool_contains_all_three_target_tier3_foods():
    c = catalog()
    assert {f.id for f in c.foods.values() if f.tier == 3 and not f.token} == {
        "garlic",
        "salad_bowl",
        "cake",
    }
    g = game()
    g.state.turn = 5
    seen = set()
    for _ in range(200):
        g._roll_shop(True)
        seen.update(s.item_id for s in g.state.shop if s and s.kind == "food")
    assert {"garlic", "salad_bowl", "cake"} <= seen
    assert "chili" not in seen  # Bus carries the perk, not a Tier-5 shop unlock.


def test_salad_has_one_action_and_feeds_two_actual_distinct_recipients():
    g = game()
    rabbit, fish, maxed = pet("rabbit"), pet("fish"), pet("fish", attack=50, health=50)
    g.state.team = [rabbit, fish, maxed]
    g.state.shop[0] = ShopItem("food", "salad_bowl", 3)
    food_actions = [a for a in g.legal_actions() if a.kind is ActionKind.BUY_FOOD and a.source == 0]
    assert food_actions == [Action(ActionKind.BUY_FOOD, 0, 0)]
    g.step(food_actions[0])
    assert (rabbit.attack, rabbit.health) == (2, 4)
    assert rabbit.ability_uses == 2
    assert (fish.attack, fish.health) == (3, 5)
    assert (maxed.attack, maxed.health) == (50, 50)


def test_salad_one_pet_gets_one_serving_not_two():
    g = game()
    g.state.team = [pet("fish")]
    feed(g, "salad_bowl")
    assert (g.state.team[0].attack, g.state.team[0].health) == (3, 4)


def test_cake_bonus_persists_after_replacement_and_is_paid_on_sale():
    g = game()
    g.state.team = [pet("fish", 2)]
    feed(g, "cake")
    g._shop_runtime().phase("end_turn")
    g._shop_runtime().phase("end_turn")
    assert g.state.team[0].sell_bonus == 2
    feed(g, "garlic")
    g._shop_runtime().phase("end_turn")
    assert g.state.team[0].sell_bonus == 2
    gold = g.state.gold
    g.step(Action(ActionKind.SELL, 0))
    assert g.state.gold == gold + 4


def test_cake_merge_uses_larger_bonus_not_sum_and_keeps_level_sale_value():
    g = game()
    g.state.team = [pet("fish", experience=2, sell_bonus=3), pet("fish", sell_bonus=5)]
    g.step(Action(ActionKind.MERGE_TEAM, 1, 0))
    assert g.state.team[0].level == 2 and g.state.team[0].sell_bonus == 5
    gold = g.state.gold
    g.step(Action(ActionKind.SELL, 0))
    assert g.state.gold == gold + 7


def test_squirrel_discount_survives_frozen_roll_but_not_fresh_food():
    g = game()
    g.state.turn = 5
    g.state.team = [pet("squirrel")]
    g.state.shop = [None] * 9
    frozen = ShopItem("food", "cake", 3, frozen=True)
    g.state.shop[0] = frozen
    g._roll_shop(True)
    g._shop_runtime().phase("start_turn")
    assert frozen.cost == 2
    g._roll_shop(True)
    fresh = [s for s in g.state.shop if s and s.kind == "food" and not s.frozen]
    assert frozen.cost == 2
    assert all(s.cost == g.catalog.foods[s.item_id].cost for s in fresh)


def test_forced_end_turn_runs_copy_and_cake_once_before_snapshot():
    g = game(max_actions_per_turn=1)
    g.state.team = [pet("deer", perk="cake"), pet("parrot")]
    captured = []

    def opponent(turn, rng, c, config):
        captured.append([PetSnapshot.from_pet(p) for p in g.state.team])
        return [pet("fish", attack=50, health=50)]

    g.opponent_provider = opponent
    result = g.step(Action(ActionKind.SWAP, 0, 1))
    assert result.info["forced_end_turn"] and len(captured) == 1
    assert captured[0][1].sell_bonus == 1
    # Parrot is first after swapping: no friend ahead to copy.
    assert captured[0][0].copied_ability is None
    assert g.state.turn == 2 and g.state.team[1].sell_bonus == 1


def test_new_state_fields_survive_snapshot_roundtrip_and_legacy_fields_stay_absent(tmp_path):
    p = pet("parrot", copied_ability="deer", sell_bonus=4, perk="cake")
    state = GameState(team=[p], shop=[ShopItem("pet", "parrot", 3, pet=p.clone())])
    payload = state.to_dict()
    assert payload["team"][0]["copied_ability"] == "deer"
    assert payload["shop"][0]["pet"]["sell_bonus"] == 4
    league = SnapshotLeague(catalog_id=catalog().catalog_id)
    league.add(5, [p])
    path = tmp_path / "pool.json"
    league.save(path)
    restored = SnapshotLeague.load(path)(
        5, random.Random(1), catalog(), GameConfig.turtle_midgame()
    )
    assert restored[0] == p and restored[0] is not p
    legacy = GameState(team=[pet("fish")]).to_dict()["team"][0]
    assert "sell_bonus" not in legacy and "copied_ability" not in legacy


def test_observations_distinguish_new_perks_copies_sale_values_and_rich_gold():
    env = SapAutoBattlerEnv(catalog(), allow_development=True)
    env.reset(seed=1)
    states = []
    for perk, copied, bonus in [
        (None, None, 0),
        ("garlic", None, 0),
        ("chili", None, 0),
        ("cake", None, 0),
        (None, "deer", 0),
        (None, None, 50),
        (None, None, 51),
    ]:
        p = pet("parrot", perk=perk, copied_ability=copied, sell_bonus=bonus)
        env.engine.state.team = [p]
        env.engine.state.shop = [ShopItem("pet", "parrot", 3, pet=p.clone())] + [None] * 8
        obs = env._observation()
        assert env.observation_space.contains(obs)
        states.append((obs["team"].tobytes(), obs["shop"].tobytes()))
    assert len(set(x[0] for x in states)) == len(states)
    assert len(set(x[1] for x in states)) == len(states)
    values = []
    for gold in (20, 25, 205, 300, 500):
        env.engine.state.gold = gold
        values.append(float(env._observation()["global"][1]))
    assert values == sorted(set(values))


def test_midgame_reachable_episode_replay_is_exact():
    from sap_rl_lab.baselines import scripted_policy

    g = game()
    policy = scripted_policy("expanded_summon")
    recorder = ReplayRecorder(g, seed=1303)
    rng = random.Random(1303)
    while not (g.state.terminated or g.state.truncated):
        recorder.step(g.codec.encode(policy.choose(g, rng)))
    replay = recorder.finish()
    assert replay.catalog_id == catalog().catalog_id
    verify_replay(replay)
    with pytest.raises(ValueError, match="catalog content"):
        verify_replay(replace(replay, catalog_sha256="0" * 64))


def test_midgame_requires_development_opt_in_and_distinct_contract():
    with pytest.raises(ValueError, match="allow_development"):
        TrainingConfig(catalog_id=catalog().catalog_id).validate()
    with pytest.raises(ValueError, match="Tier-3"):
        GameConfig.turtle_midgame(max_shop_tier=4)


def test_midgame_smoke_train_save_reload_and_learned_pool(tmp_path):
    from sb3_contrib import MaskablePPO

    from sap_rl_lab.evaluation import evaluate_policy
    from sap_rl_lab.learned_pool import collect

    paths = []
    for seed in (17, 1017):
        path = tmp_path / f"pool-{seed}.json"
        build_scripted_league("expanded_stats", 2, seed=seed, catalog=catalog()).save(path)
        paths.append(str(path))
    output = tmp_path / "smoke"
    train(
        TrainingConfig(
            catalog_id=catalog().catalog_id,
            allow_development=True,
            shop_action_limit_mode="force_battle",
            action_cost=0.005,
            timesteps=32,
            environments=2,
            rollout_steps=16,
            batch_size=16,
            opponent_league=paths[0],
            validation_league=paths[1],
            validation_episodes=2,
            evaluation_interval=32,
            output_dir=str(output),
            device="cpu",
        )
    )
    model = MaskablePPO.load(output / "best_model.zip", device="cpu")
    assert model.sap_environment_contract["catalog_sha256"] == catalog_digest(catalog())
    assert model.action_space.n == 139
    result = evaluate_policy(model, episodes=2, seed=30000, opponent_league=paths[1])
    history = json.loads((output / "validation_history.json").read_text())
    selected = [r for r in history if r["selected"]][-1]
    assert result["success_without_forcing_rate"] == selected["success_without_forcing_rate"]
    assert result["mean_return"] == selected["mean_return"]
    league, rows = collect(
        model, episodes=2, seed=20017, opponent_provider=SnapshotLeague.load(paths[0])
    )
    assert len(rows) == 2 and len(league) > 0
    assert league.catalog_id == catalog().catalog_id
    assert np.isfinite(result["mean_return"])


def test_midgame_selection_does_not_reintroduce_one_percent_hard_gate():
    from sap_rl_lab.training import midgame_checkpoint_score

    stronger = dict(success_rate=0.8, forced_episode_rate=0.015, mean_return=5)
    weaker = dict(success_rate=0.7, forced_episode_rate=0, mean_return=4)
    assert midgame_checkpoint_score(stronger) > midgame_checkpoint_score(weaker)
    stable = dict(stronger, forced_episode_rate=0.0)
    assert midgame_checkpoint_score(stable) > midgame_checkpoint_score(stronger)
