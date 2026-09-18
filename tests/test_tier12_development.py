"""Executable spec tests, NOT recordings from the official game client."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest

from sap_rl_lab.actions import Action, ActionCodec, ActionKind
from sap_rl_lab.catalog import PetSpec, load_catalog, load_catalog_by_id
from sap_rl_lab.domain import GameConfig, Pet, ShopItem
from sap_rl_lab.engine import AutoBattler, InvalidAction, make_pet
from sap_rl_lab.replay import ReplayRecorder, verify_replay
from sap_rl_lab.shop import ContentNotReady, merged_pet, shop_slots, stock_items

DEV_ID = "turtle-v0.46-tier12-dev-v3"


def development_catalog():
    return load_catalog_by_id(DEV_ID)


def fixture_catalog():
    """Synthetic higher-tier pools let us test mechanics before implementing pets."""
    catalog = development_catalog()
    pets = dict(catalog.pets)
    for tier in range(2, 7):
        for suffix in ("a", "b"):
            pet_id = f"fixture_tier{tier}_{suffix}"
            pets[pet_id] = PetSpec(pet_id, pet_id, tier, tier, tier, False, ())
    return replace(catalog, pets=pets, implemented_shop_tiers=(1, 2, 3, 4, 5, 6))


def game_with_fixtures(**overrides):
    game = AutoBattler(fixture_catalog(), GameConfig.turtle_development(**overrides))
    game.reset(seed=17)
    return game


def rewards(game):
    return [(i, item) for i, item in enumerate(game.state.shop) if item and item.choice_group]


def cause_level_up(game, pet_id="fish"):
    game.state.team = [Pet(pet_id, 3, 4, experience=2), Pet("ant", 2, 2)]
    game.state.shop[0] = ShopItem("pet", pet_id, 3, pet=make_pet(game.catalog, pet_id))
    game.step(Action(ActionKind.MERGE, 0, 0))


def test_development_catalog_is_explicit_and_legacy_stays_eight_pets():
    old, new = load_catalog(), development_catalog()
    assert old.rules_version == 2 and len(old.rollable_pet_ids) == 8
    assert new.rules_version == 3 and len(new.rollable_pet_ids) == 10
    assert new.development_only and new.implemented_shop_tiers == (1,)
    assert (new.pets["duck"].attack, new.pets["duck"].health) == (2, 2)
    assert (new.pets["beaver"].attack, new.pets["beaver"].health) == (3, 2)
    assert AutoBattler(new).config == GameConfig.turtle_development()
    with pytest.raises(ValueError, match="selected together"):
        AutoBattler(new, GameConfig(shop_pet_stats=True))


@pytest.mark.parametrize("xp,bonus", [(1, 1), (3, 2), (6, 3)])
def test_duck_all_levels_freeze_buy_and_no_future_roll_buff(xp, bonus):
    game = AutoBattler(development_catalog())
    game.reset(seed=12)
    duck = Pet("duck", 2, 2, experience=xp)
    friend = Pet("ant", 2, 2)
    game.state.team = [duck, friend]
    game.state.shop[0] = ShopItem("pet", "fish", 3, True, Pet("fish", 2, 3))
    game.state.shop[1] = ShopItem("pet", "ant", 3, pet=Pet("ant", 49, 49))
    initial_gold = game.state.gold
    game.step(Action(ActionKind.SELL, 0))
    assert game.state.gold == initial_gold + duck.level
    assert friend.health == 2
    assert game.state.shop[0].pet.health == 3 + bonus
    assert game.state.shop[1].pet.health == 50
    assert game.state.shop[1].pet.attack == 49
    game.step(Action(ActionKind.ROLL))
    assert game.state.shop[0].pet.health == 3 + bonus
    for item in game.state.shop[1:]:
        if item and item.kind == "pet":
            assert item.pet.health == game.catalog.pets[item.item_id].health
    game.step(Action(ActionKind.BUY_PET, 0))
    assert game.state.team[-1].health == 3 + bonus
    assert game.catalog.pets["fish"].health == 3


@pytest.mark.parametrize("xp,bonus", [(1, 1), (3, 2), (6, 3)])
@pytest.mark.parametrize("friend_count", [0, 1, 2, 4])
def test_beaver_all_levels_targets_distinct_remaining_friends(xp, bonus, friend_count):
    game = AutoBattler(development_catalog())
    game.reset(seed=2)
    game.state.team = [Pet("beaver", 3, 2, xp)] + [Pet("fish", 2, 3) for _ in range(friend_count)]
    shop_before = deepcopy(game.state.to_dict()["shop"])
    game.step(Action(ActionKind.SELL, 0))
    assert sorted(p.attack for p in game.state.team) == sorted(
        [2 + bonus] * min(2, friend_count) + [2] * max(0, friend_count - 2)
    )
    assert all(p.health == 3 for p in game.state.team)
    assert game.state.to_dict()["shop"] == shop_before


def test_beaver_stat_cap():
    game = AutoBattler(development_catalog())
    game.reset(seed=2)
    game.state.team = [Pet("beaver", 3, 2, 6), Pet("fish", 49, 3)]
    game.step(Action(ActionKind.SELL, 0))
    assert game.state.team[0].attack == 50


@pytest.mark.parametrize(
    "turn,expected",
    [
        (1, (3, 1, 6)),
        (4, (3, 1, 6)),
        (5, (4, 2, 8)),
        (8, (4, 2, 8)),
        (9, (5, 2, 9)),
        (30, (5, 2, 9)),
    ],
)
def test_shop_capacity_progression_and_two_spare_spaces(turn, expected):
    game = game_with_fixtures()
    assert shop_slots(turn) == expected
    game.state.turn = turn
    game._roll_shop(free=True)
    assert len(game.state.shop) == 9  # Stable observation/action dimensions.
    assert sum(bool(x and x.kind == "pet") for x in game.state.shop) == expected[0]
    assert sum(bool(x and x.kind == "food") for x in game.state.shop) == expected[1]
    assert game.state.shop[expected[2] :] == [None] * (9 - expected[2])


@pytest.mark.parametrize(
    "source_xp,target_xp,expected", [(1, 1, 2), (1, 2, 3), (3, 3, 4), (4, 3, 5), (1, 5, 6)]
)
def test_modern_merge_uses_highest_plus_one(source_xp, target_xp, expected):
    a = Pet("fish", 7, 9, target_xp, "honey", 2, 0)
    b = Pet("fish", 10, 4, source_xp, None, 1, 3)
    before = a.clone(), b.clone()
    combined = merged_pet(a, b)
    assert (combined.attack, combined.health, combined.experience) == (11, 10, expected)
    assert (combined.temporary_attack, combined.temporary_health, combined.perk) == (2, 3, "honey")
    assert (a, b) == before


@pytest.mark.parametrize("source,target", [(0, 1), (1, 0)])
def test_team_merge_handles_equal_pets_and_both_directions(source, target):
    game = game_with_fixtures()
    marker = Pet("pig", 4, 1)
    game.state.team = [Pet("fish", 2, 3), Pet("fish", 2, 3), marker]
    gold = game.state.gold
    action = Action(ActionKind.MERGE_TEAM, source, target)
    assert game.action_mask()[game.codec.encode(action)]
    game.step(action)
    assert len(game.state.team) == 2 and game.state.team[1] is marker
    assert game.state.team[0].experience == 2 and game.state.gold == gold
    assert not rewards(game)


def test_team_merge_does_not_trigger_otter_buy_or_horse_summon():
    game = game_with_fixtures()
    friend = Pet("horse", 2, 1)
    game.state.team = [Pet("otter", 1, 3), Pet("otter", 1, 3), friend]
    game.step(Action(ActionKind.MERGE_TEAM, 0, 1))
    assert friend.health == 1
    assert game.state.team[0].temporary_attack == 0


def test_two_level_two_merges_never_duplicate_shop_rewards():
    game = game_with_fixtures()
    friend = Pet("ant", 2, 2)
    game.state.team = [Pet("fish", 5, 6, 3), Pet("fish", 7, 8, 5), friend]
    game.step(Action(ActionKind.MERGE_TEAM, 0, 1))
    assert game.state.team[0].level == 3
    assert (friend.attack, friend.health) == (4, 4)  # Fish departing-level-2 skill.
    assert not rewards(game)


@pytest.mark.parametrize("turn,tier", [(1, 2), (3, 3), (11, 6)])
def test_levelup_reward_is_exactly_next_shop_tier_not_pets_tier(turn, tier):
    game = game_with_fixtures()
    game.state.turn = turn
    game._roll_shop(free=True)
    cause_level_up(game)
    offered = rewards(game)
    assert len(offered) == 2 and offered[0][1].choice_group == offered[1][1].choice_group
    assert all(game.catalog.pets[item.item_id].tier == tier for _, item in offered)
    assert all(item.cost == 3 for _, item in offered)


@pytest.mark.parametrize("merge", [False, True])
def test_buying_or_merging_reward_removes_frozen_partner_and_charges_once(merge):
    game = game_with_fixtures()
    cause_level_up(game)
    offered = rewards(game)
    index, chosen = offered[0]
    offered[1][1].frozen = True
    chosen.pet.health = 17
    if merge:
        game.state.team = [make_pet(game.catalog, chosen.item_id)]
    game.state.gold = 3
    game.step(Action(ActionKind.MERGE, index, 0) if merge else Action(ActionKind.BUY_PET, index))
    assert game.state.gold == 0 and not rewards(game)
    assert (game.state.team[0] if merge else game.state.team[-1]).health == (18 if merge else 17)


def test_reward_links_survive_freezing_and_rolling():
    game = game_with_fixtures()
    cause_level_up(game)
    offered = rewards(game)
    for _, item in offered:
        item.frozen = True
    game.step(Action(ActionKind.ROLL))
    assert len(rewards(game)) == 2
    game.step(Action(ActionKind.BUY_PET, rewards(game)[0][0]))
    assert not rewards(game)


def test_duck_buffs_both_levelup_choices_and_merge_keeps_the_chosen_buff():
    game = game_with_fixtures()
    cause_level_up(game)
    game.state.team.append(Pet("duck", 2, 2, 3))
    game.step(Action(ActionKind.SELL, len(game.state.team) - 1))
    offered = rewards(game)
    assert all(item.pet.health == 4 for _, item in offered)  # Tier-2 fixture: base 2.
    index, choice = offered[0]
    game.state.team = [make_pet(game.catalog, choice.item_id)]
    game.step(Action(ActionKind.MERGE, index, 0))
    assert game.state.team[0].health == 5 and not rewards(game)


def test_separate_reward_pairs_do_not_clear_each_other():
    game = game_with_fixtures()
    game.state.shop = [None] * 9
    game._stock_level_up_choices()
    first_group = rewards(game)[0][1].choice_group
    game._stock_level_up_choices()
    second_group = rewards(game)[-1][1].choice_group
    assert first_group != second_group
    game.step(Action(ActionKind.BUY_PET, rewards(game)[0][0]))
    assert len(rewards(game)) == 2
    assert all(item.choice_group == second_group for _, item in rewards(game))


def test_stock_never_overwrites_frozen_items_or_its_own_new_items():
    frozen = ShopItem("pet", "ant", 3, frozen=True)
    shop = [frozen] * 5 + [None] + [None] * 3
    a, b = ShopItem("pet", "duck", 3), ShopItem("pet", "beaver", 3)
    stock_items(shop, [a, b], 6, "pet")
    assert all(item is frozen for item in shop[:5])
    assert shop[5] is a and shop[6:] == [None] * 3


def test_pigeon_stocks_distinct_crumbs_from_food_edge_and_preserves_frozen():
    game = game_with_fixtures()
    game.state.team = [Pet("pigeon", 3, 2, 6)]
    game.state.shop[5].frozen = True
    frozen = game.state.shop[5]
    game.step(Action(ActionKind.SELL, 0))
    assert game.state.shop[5] is frozen
    assert sum(bool(x and x.item_id == "bread_crumbs") for x in game.state.shop) == 3
    assert game.state.shop[4].item_id == "bread_crumbs"
    assert game.state.shop[3].item_id == "bread_crumbs"


@pytest.mark.parametrize("team_merge", [False, True])
def test_missing_reward_pool_fails_atomically_instead_of_replacing_with_tier1(team_merge):
    game = AutoBattler(development_catalog())
    game.reset(seed=3)
    game.state.team = [Pet("fish", 3, 4, 2), Pet("ant", 2, 2)]
    if team_merge:
        game.state.team.append(Pet("fish", 2, 3))
        action = Action(ActionKind.MERGE_TEAM, 2, 0)
    else:
        game.state.shop[0] = ShopItem("pet", "fish", 3, pet=Pet("fish", 2, 3))
        action = Action(ActionKind.MERGE, 0, 0)
    before = deepcopy(game.state.to_dict()), game.rng.getstate()
    with pytest.raises(ContentNotReady, match="Tier 2"):
        game.step(action)
    assert (game.state.to_dict(), game.rng.getstate()) == before


@pytest.mark.parametrize("forced", [False, True])
def test_missing_next_shop_rolls_back_whole_battle_and_budget_action(forced):
    game = AutoBattler(development_catalog(), GameConfig.turtle_development(max_actions_per_turn=1))
    game.reset(seed=3)
    game.state.turn = 2
    before = deepcopy(game.state.to_dict()), game.rng.getstate(), game.last_battle
    action = Action(ActionKind.FREEZE, 0) if forced else Action(ActionKind.END_TURN)
    with pytest.raises(ContentNotReady, match="Tier 2"):
        game.step(action)
    assert (game.state.to_dict(), game.rng.getstate(), game.last_battle) == before


def test_missing_exact_reward_tier_is_not_a_through_tier_fallback():
    game = game_with_fixtures()
    game.catalog = replace(game.catalog, pets=development_catalog().pets)
    with pytest.raises(ContentNotReady, match="Missing exact Tier 2"):
        game._stock_level_up_choices()


def test_codec_preserves_old_ids_and_illegal_team_merges_are_atomic():
    old, new = ActionCodec(), ActionCodec(team_merges=True)
    assert old.size == 71 and new.size == 91
    assert all(old.decode(i) == new.decode(i) for i in range(old.size))
    game = game_with_fixtures()
    game.state.team = [Pet("fish", 2, 3), Pet("ant", 2, 2)]
    before = game.state.to_dict()
    with pytest.raises(InvalidAction):
        game.step(Action(ActionKind.MERGE_TEAM, 0, 1))
    assert game.state.to_dict() == before


def test_development_prefix_replay_restores_catalog_config_and_action_vocabulary():
    recorder = ReplayRecorder(
        AutoBattler(development_catalog(), GameConfig.turtle_development(max_turns=2)), seed=9
    )
    recorder.step(recorder.engine.codec.encode(Action(ActionKind.BUY_PET, 0)))
    recorder.step(0)
    recorder.step(0)
    replay = recorder.finish()
    assert replay.final_state["truncated"]
    assert replay.config["shop_rules"] == "turtle_v046_development"
    verify_replay(replay)


def test_gym_rejects_unfinished_catalog_by_default_and_observes_choice_links():
    pytest.importorskip("gymnasium")
    from sap_rl_lab.env import SapAutoBattlerEnv

    with pytest.raises(ValueError, match="not training-ready"):
        SapAutoBattlerEnv(development_catalog())
    env = SapAutoBattlerEnv(fixture_catalog(), allow_development=True)
    env.reset(seed=17)
    cause_level_up(env.engine)
    obs = env._observation()
    assert env.observation_space.contains(obs)
    offered = rewards(env.engine)
    for index, _ in offered:
        links = obs["shop"][index, -9:]
        assert links.sum() == 2
        assert all(links[other_index] == 1 for other_index, _ in offered)
    env.engine.catalog = development_catalog()
    env.engine.state.turn = 2
    with pytest.raises(ContentNotReady):
        env.step(0)
    assert env.episode_actions == 0


@pytest.mark.parametrize("seed", range(5))
def test_seeded_development_rollouts_keep_state_and_mask_invariants(seed):
    game = game_with_fixtures(max_actions_per_turn=10)
    game.reset(seed=seed)
    rng = random.Random(seed)
    while not (game.state.terminated or game.state.truncated):
        mask = game.action_mask()
        assert {i for i, allowed in enumerate(mask) if allowed} == set(game.legal_action_ids())
        transition = game.step(rng.choice(game.legal_actions()))
        state = game.state
        assert 0 <= len(state.team) <= 5 and state.gold >= 0
        assert len(state.shop) == 9
        assert all(1 <= pet.experience <= 6 and pet.health > 0 for pet in state.team)
        assert transition.info.get("reason") != "shop_action_limit"
        assert all(
            item.pet is not None and item.pet.spec_id == item.item_id
            for item in state.shop
            if item is not None and item.kind == "pet"
        )
