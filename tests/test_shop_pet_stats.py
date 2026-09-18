import importlib.util
from dataclasses import replace

import pytest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import AbilitySpec, PetSpec, load_catalog
from sap_rl_lab.domain import GameConfig, Pet, ShopItem
from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.replay import ReplayRecorder, verify_replay


def shop_buffer_catalog():
    """Synthetic fixture for Duck's required primitive, not a game-parity fixture."""
    catalog = load_catalog()
    buffer = PetSpec(
        "shop_buffer",
        "Test Shop Buffer",
        1,
        1,
        1,
        False,
        (AbilitySpec("sell", "buff_shop_pets", {"health_by_level": [1, 2, 3]}),),
    )
    return replace(catalog, pets={**catalog.pets, buffer.id: buffer})


@pytest.mark.parametrize("experience,bonus", [(1, 1), (3, 2), (6, 3)])
def test_shop_buff_all_levels_including_frozen_slots(experience, bonus):
    catalog = shop_buffer_catalog()
    game = AutoBattler(catalog, GameConfig(shop_pet_stats=True))
    game.reset(seed=1)
    game.state.team = [Pet("shop_buffer", 1, 1, experience), Pet("fish", 2, 3)]
    game.state.shop = [
        ShopItem("pet", "fish", 3, frozen=True, pet=Pet("fish", 2, 3)),
        ShopItem("pet", "ant", 3, pet=Pet("ant", 49, 49)),
        None,
        None,
        ShopItem("food", "apple", 3),
    ]
    game.step(Action(ActionKind.SELL, source=0))
    assert game.state.shop[0].pet.health == 3 + bonus
    assert game.state.shop[0].pet.attack == 2
    assert game.state.shop[1].pet.health == min(49 + bonus, 50)
    assert game.state.shop[1].pet.attack == 49
    assert game.state.team[0].health == 3
    assert game.state.shop[4].pet is None


def test_shop_buff_catalog_requires_observed_shop_stats():
    with pytest.raises(ValueError, match="observable"):
        AutoBattler(shop_buffer_catalog())


@pytest.mark.parametrize("merge", [False, True])
def test_buy_and_merge_consume_stored_stats_without_mutating_source(merge):
    game = AutoBattler(config=GameConfig(shop_pet_stats=True))
    game.reset(seed=4)
    incoming = Pet("fish", 7, 9)
    game.state.shop[0] = ShopItem("pet", "fish", 3, pet=incoming)
    game.state.team = [Pet("fish", 2, 3)] if merge else []
    action = Action(ActionKind.MERGE, 0, 0) if merge else Action(ActionKind.BUY_PET, 0)
    game.step(action)
    pet = game.state.team[0]
    assert (pet.attack, pet.health) == ((8, 10) if merge else (7, 9))
    assert (incoming.attack, incoming.health) == (7, 9)
    assert game.state.shop[0] is None


def test_freezing_preserves_buff_but_rerolling_discards_unfrozen_buff():
    game = AutoBattler(config=GameConfig(shop_pet_stats=True))
    game.reset(seed=2)
    frozen, unfrozen = game.state.shop[:2]
    frozen.frozen = True
    frozen.pet.health = unfrozen.pet.health = 40
    game.step(Action(ActionKind.ROLL))
    assert game.state.shop[0] is frozen and frozen.pet.health == 40
    assert game.state.shop[1] is not unfrozen
    assert game.state.shop[1].pet.health == game.catalog.pets[game.state.shop[1].item_id].health
    game.state.team = [Pet("fish", 20, 20)]
    game.step(Action(ActionKind.END_TURN))
    assert game.state.shop[0] is frozen and frozen.pet.health == 40


def test_shop_stats_replay_and_legacy_payload_shape():
    for enabled in (False, True):
        recorder = ReplayRecorder(AutoBattler(config=GameConfig(shop_pet_stats=enabled)), seed=9)
        recorder.step(0)
        replay = recorder.finish()
        verify_replay(replay)
        item = next(x for x in replay.final_state["shop"] if x and x["kind"] == "pet")
        assert ("pet" in item) is enabled


@pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None, reason="RL extra not installed")
def test_shop_stats_are_observed_and_change_the_contract_shape():
    from gymnasium.utils.env_checker import check_env

    from sap_rl_lab.env import SapAutoBattlerEnv

    old = SapAutoBattlerEnv()
    env = SapAutoBattlerEnv(
        config=GameConfig(shop_pet_stats=True, shop_action_limit_mode="force_battle")
    )
    check_env(env, skip_render_check=True)
    before, _ = env.reset(seed=3)
    env.engine.state.shop[0].pet.health = 30
    after = env._observation()
    assert env.observation_space.contains(after)
    assert env.observation_space["shop"].shape[1] == old.observation_space["shop"].shape[1] + 6
    assert (before["shop"][0] != after["shop"][0]).sum() == 1
    assert not old.observation_space.contains(after)
