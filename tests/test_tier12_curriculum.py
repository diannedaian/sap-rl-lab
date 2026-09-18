"""Thirty-pet specification and interaction tests; not official-client fixtures."""

import random
from copy import deepcopy
from dataclasses import replace

import pytest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import load_catalog_by_id
from sap_rl_lab.domain import BattleOutcome, GameConfig, ShopItem
from sap_rl_lab.engine import AutoBattler, default_opponent, make_pet, resolve_battle
from sap_rl_lab.events import EventRuntime, attack, battle_runtime, health
from sap_rl_lab.opponents import PetSnapshot
from sap_rl_lab.replay import ReplayRecorder, verify_replay
from sap_rl_lab.shop import ContentNotReady

CATALOG_ID = "turtle-v0.46-tier12-curriculum-v4"
LEVELS = [(1, 1), (2, 3), (3, 6)]
TIER3 = "dodo badger dolphin giraffe elephant camel rabbit ox dog sheep".split()


def catalog():
    return load_catalog_by_id(CATALOG_ID)


def game(**config):
    result = AutoBattler(catalog(), GameConfig.turtle_curriculum(**config))
    result.reset(101)
    return result


def pet(name, level=1, **values):
    result = make_pet(catalog(), name)
    result.experience = (1, 3, 6)[level - 1]
    for key, value in values.items():
        setattr(result, key, value)
    return result


def runtime(team, enemies=None, *, combat=True):
    return EventRuntime(catalog(), random.Random(4), [team, enemies or []], combat=combat)


def feed(g, name, target=0):
    food = g.catalog.foods[name]
    g.state.shop[0] = ShopItem("food", name, food.cost)
    g.state.gold = max(g.state.gold, food.cost)
    return g.step(Action(ActionKind.BUY_FOOD, 0, target))


def test_catalog_contains_all_three_tiers_and_required_foods_tokens():
    c = catalog()
    assert c.rules_version == 4 and c.development_only
    assert len(c.rollable_pet_ids) == 30
    for tier in (1, 2, 3):
        assert sum(p.tier == tier and not p.token for p in c.pets.values()) == 10
    assert {p.id for p in c.pets.values() if p.tier == 3} == set(TIER3)
    assert {p.id for p in c.pets.values() if p.token} == {
        "bee",
        "zombie_cricket",
        "dirty_rat",
        "ram",
    }
    assert set(c.rollable_food_ids) == {"apple", "honey", "cupcake", "meat_bone", "sleeping_pill"}
    assert all(p.abilities for p in c.pets.values() if not p.token)
    assert AutoBattler(c).config == GameConfig.turtle_curriculum()
    assert c.pets["otter"].health == 4


@pytest.mark.parametrize("level,xp", LEVELS)
@pytest.mark.parametrize("name", ["pig", "pigeon", "duck", "beaver"])
def test_tier1_sell_abilities_in_new_event_runtime(name, level, xp):
    g = game()
    g.state.team = [pet(name, level)] + [pet("fish") for _ in range(3)]
    g.state.shop = [ShopItem("pet", "fish", 3, pet=pet("fish"))] + [None] * 8
    g.state.gold = 0
    g.step(Action(ActionKind.SELL, 0))
    assert g.state.gold == level * (2 if name == "pig" else 1)
    if name == "pigeon":
        foods = [s for s in g.state.shop if s and s.kind == "food"]
        assert len(foods) == level
        assert all(s.item_id == "bread_crumbs" and s.cost == 0 for s in foods)
    if name == "duck":
        assert g.state.shop[0].pet.health == 3 + level
    if name == "beaver":
        assert sorted(p.attack for p in g.state.team) == [2, 2 + level, 2 + level]


@pytest.mark.parametrize("level,xp", LEVELS)
def test_otter_buy_buffs_distinct_friends_and_not_self(level, xp):
    g = game()
    friends = [pet("fish") for _ in range(3)]
    g.state.team = friends[:]
    g.state.shop[0] = ShopItem("pet", "otter", 3, pet=pet("otter", level))
    g.step(Action(ActionKind.BUY_PET, 0))
    assert sorted(p.health for p in friends) == [3] * (3 - level) + [4] * level
    assert g.state.team[-1].health == 4


@pytest.mark.parametrize("xp,bonus", [(2, 1), (5, 2)])
def test_fish_merge_uses_departing_level_in_new_runtime(xp, bonus):
    g = game()
    friends = [pet("duck") for _ in range(3)]
    g.state.team = [pet("fish", experience=xp)] + friends
    g.state.shop[0] = ShopItem("pet", "fish", 3, pet=pet("fish"))
    g.step(Action(ActionKind.MERGE, 0, 0))
    assert sorted(p.attack for p in friends) == [2, 2 + bonus, 2 + bonus]
    assert sorted(p.health for p in friends) == [2, 2 + bonus, 2 + bonus]


@pytest.mark.parametrize("level,xp", LEVELS)
def test_ant_cricket_horse_and_mosquito_each_level(level, xp):
    ant, fish = pet("ant", level), pet("fish")
    r = runtime([ant, fish], combat=False)
    r.faint(0, ant)
    r.drain()
    assert (fish.attack, fish.health) == (2 + level, 3 + level)
    cricket, horse = pet("cricket", level), pet("horse", level)
    r = runtime([cricket, horse], combat=False)
    r.faint(0, cricket)
    r.drain()
    token = r.teams[0][0]
    assert token.spec_id == "zombie_cricket"
    assert token.level == level
    assert (token.attack, token.temporary_attack, health(token)) == (level, level, level)
    enemies = [pet("fish", health=10) for _ in range(5)]
    r = runtime([pet("mosquito", level)], enemies)
    r.phase("start_battle")
    assert sorted(p.health for p in enemies) == [9] * level + [10] * (5 - level)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_sheep_tokens_keep_level_and_sale_value_in_shop(level, xp):
    g = game()
    g.state.team = [pet("sheep", level)]
    feed(g, "sleeping_pill")
    assert len(g.state.team) == 2
    assert all(p.spec_id == "ram" and p.level == level for p in g.state.team)
    before = g.state.gold
    g.step(Action(ActionKind.SELL, 0))
    assert g.state.gold == before + level


def test_queued_start_ability_survives_snipe_but_dead_crab_cannot_revive():
    mosquito = pet("mosquito", health=1)
    dolphin = pet("dolphin", attack=10, health=10)
    r = runtime([mosquito], [dolphin])
    r.phase("start_battle")
    assert not r.teams[0] and dolphin.health == 9
    crab, friend = pet("crab", health=1), pet("fish", health=40)
    r = runtime([crab, friend], [pet("dolphin", attack=10)])
    r.phase("start_battle")
    assert r.teams[0] == [friend]


def test_merge_caps_total_stats_and_keeps_spent_ability_uses():
    g = game()
    g.state.team = [pet("ox", attack=49, health=48, ability_uses=1)]
    incoming = pet("ox", temporary_attack=4, temporary_health=5)
    g.state.shop[0] = ShopItem("pet", "ox", 3, pet=incoming)
    g.step(Action(ActionKind.MERGE, 0, 0))
    merged = g.state.team[0]
    assert attack(merged) == health(merged) == 50
    assert merged.ability_uses == 1


@pytest.mark.parametrize("turn", [1, 2, 3, 4, 5, 9, 11, 30])
def test_normal_shop_and_default_opponents_never_leak_tier3_or_higher(turn):
    g = game()
    g.state.turn = turn
    for _ in range(20):
        g._roll_shop(free=True)
        expected_tier = 1 if turn <= 2 else 2
        assert g.unlocked_tier == expected_tier
        assert all(
            g.catalog.pets[item.item_id].tier <= expected_tier
            for item in g.state.shop
            if item and item.kind == "pet"
        )
        assert all(
            g.catalog.pets[p.spec_id].tier <= expected_tier
            for p in default_opponent(turn, g.rng, g.catalog, g.config)
        )


@pytest.mark.parametrize("turn,reward_tier", [(1, 2), (3, 3), (9, 3), (30, 3)])
def test_reward_uses_capped_shop_tier_even_for_tier3_pet_levelup(turn, reward_tier):
    g = game()
    g.state.turn = turn
    g.state.team = [pet("dodo", experience=2)]
    g.state.shop[0] = ShopItem("pet", "dodo", 3, pet=pet("dodo"))
    g.step(Action(ActionKind.MERGE, 0, 0))
    choices = [item for item in g.state.shop if item and item.choice_group]
    assert len(choices) == 2
    assert all(g.catalog.pets[item.item_id].tier == reward_tier for item in choices)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_swan_start_turn_gold_and_worm_level_specific_discounted_food(level, xp):
    g = game()
    g.state.team = [pet("swan", level), pet("worm", level)]
    g.state.gold = 10
    g._shop_runtime().phase("start_turn")
    assert g.state.gold == 10 + level
    expected_food = ("apple", "better_apple", "best_apple")[level - 1]
    offers = [
        item for item in g.state.shop if item and item.item_id == expected_food and item.cost == 2
    ]
    assert len(offers) == 1
    before = g.state.team[0].clone()
    index = next(i for i, item in enumerate(g.state.shop) if item is offers[0])
    g.step(Action(ActionKind.BUY_FOOD, index, 0))
    assert g.state.gold == 8 + level
    assert (g.state.team[0].attack, g.state.team[0].health) == (
        before.attack + level,
        before.health + level,
    )


@pytest.mark.parametrize("level,xp", LEVELS)
def test_giraffe_start_turn_nearest_ahead_not_end_turn(level, xp):
    g = game()
    front = [pet("fish") for _ in range(4)]
    g.state.team = front + [pet("giraffe", level)]
    g._shop_runtime().phase("end_turn")
    assert [p.attack for p in front] == [2] * 4
    g._shop_runtime().phase("start_turn")
    assert [p.attack for p in front] == [2] * (4 - level) + [3] * level
    assert [p.health for p in front] == [3] * (4 - level) + [4] * level


@pytest.mark.parametrize("level,xp", LEVELS)
@pytest.mark.parametrize(
    "outcome", [None, BattleOutcome.LOSS, BattleOutcome.DRAW, BattleOutcome.WIN]
)
def test_snail_previous_loss_only_three_nearest_ahead(level, xp, outcome):
    g = game()
    g.state.team = [pet("fish") for _ in range(4)] + [pet("snail", level)]
    g.state.previous_outcome = outcome
    g._shop_runtime().phase("end_turn")
    bonus = level if outcome is BattleOutcome.LOSS else 0
    assert [p.attack for p in g.state.team[:4]] == [2] + [2 + bonus] * 3
    assert all(p.health == 3 for p in g.state.team[:4])


def test_forced_battle_runs_lifecycle_once_and_turn3_recovers_one_life():
    g = game(max_actions_per_turn=1)
    g.opponent_provider = lambda *args: []
    g.state.turn = 2
    g.state.lives = 3
    g.state.previous_outcome = BattleOutcome.LOSS
    friend, snail, swan = pet("fish"), pet("snail"), pet("swan", 2)
    g.state.team = [friend, snail, swan]
    result = g.step(Action(ActionKind.FREEZE, 0))
    assert result.info["forced_end_turn"] and g.state.turn == 3
    assert g.state.lives == 4 and g.state.gold == 12
    assert friend.attack == 3 and g.state.wins == 1
    g.step(Action(ActionKind.END_TURN))
    assert g.state.lives == 4 and friend.attack == 3


@pytest.mark.parametrize("level,xp", LEVELS)
def test_crab_adds_percentage_of_healthiest_other_pet_not_overwrite(level, xp):
    crab = pet("crab", level, health=9)
    r = runtime([crab, pet("fish", health=12), pet("ant", health=4)])
    r.phase("start_battle")
    assert crab.health == 9 + 3 * level
    alone = pet("crab", level, health=20)
    runtime([alone]).phase("start_battle")
    assert alone.health == 20


@pytest.mark.parametrize("level,bonus", [(1, 2), (2, 5), (3, 7)])
def test_dodo_rounds_down_and_ignores_meat_bone_for_attack_sharing(level, bonus):
    ahead = pet("fish")
    dodo = pet("dodo", level, attack=5, perk="meat_bone")
    behind = pet("fish")
    r = runtime([ahead, dodo, behind])
    r.phase("start_battle")
    assert ahead.attack == 2 + bonus and behind.attack == 2 and dodo.attack == 5


@pytest.mark.parametrize("level,xp", LEVELS)
def test_dolphin_retargets_after_each_lethal_shot(level, xp):
    enemies = [pet("fish", health=h) for h in (3, 4, 9)]
    r = runtime([pet("dolphin", level)], enemies)
    r.phase("start_battle")
    assert len(r.alive(1)) == 3 - min(level, 2)
    assert enemies[-1].health == (5 if level == 3 else 9)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_flamingo_faint_buffs_two_nearest_behind_in_shop_permanently(level, xp):
    g = game()
    g.state.team = [pet("fish"), pet("flamingo", level), pet("fish"), pet("fish"), pet("fish")]
    feed(g, "sleeping_pill", 1)
    assert [(p.attack, p.health) for p in g.state.team] == [
        (2, 3),
        (2 + level, 3 + level),
        (2 + level, 3 + level),
        (2, 3),
    ]
    assert all(p.temporary_attack == p.temporary_health == 0 for p in g.state.team)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_hedgehog_hits_both_teams_but_battle_never_mutates_shop_pets(level, xp):
    hedgehog, friend, enemy = pet("hedgehog", level), pet("fish", health=20), pet("fish", health=20)
    r = runtime([hedgehog, friend], [enemy])
    r.faint(0, hedgehog)
    r.drain()
    assert friend.health == enemy.health == 20 - 2 * level
    originals = [pet("hedgehog", level), pet("fish", health=20)]
    before = deepcopy(originals)
    resolve_battle(originals, [pet("fish", attack=12, health=20)], catalog(), random.Random(6))
    assert originals == before


@pytest.mark.parametrize("level,xp", LEVELS)
@pytest.mark.parametrize("lethal", [False, True])
def test_peacock_hurt_fires_even_on_lethal_damage(level, xp, lethal):
    peacock = pet("peacock", level, health=1 if lethal else 5)
    r = runtime([peacock])
    r.damage_batch([(0, peacock, 1)])
    r.drain()
    assert peacock.attack == 2 + 3 * level
    assert len(r.alive(0)) == (0 if lethal else 1)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_camel_lethal_hurt_still_buffs_nearest_surviving_friend_behind(level, xp):
    camel, friend, distant = pet("camel", level, health=1), pet("fish"), pet("fish")
    r = runtime([camel, friend, distant])
    r.damage_batch([(0, camel, 1)])
    r.drain()
    assert (friend.attack, friend.health) == (2 + level, 3 + 2 * level)
    assert (distant.attack, distant.health) == (2, 3)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_elephant_separate_hits_each_trigger_hurt(level, xp):
    elephant, peacock = pet("elephant", level), pet("peacock", health=20)
    r = runtime([elephant, peacock])
    r.emit(0, elephant, "after_attack")
    r.drain()
    assert peacock.health == 20 - level and peacock.attack == 2 + 3 * level


@pytest.mark.parametrize("level,xp", LEVELS)
def test_kangaroo_only_immediately_ahead_attack_not_own_attack(level, xp):
    front, kangaroo = pet("fish", attack=1, health=1), pet("kangaroo", level, attack=0, health=20)
    r = runtime([front, kangaroo], [pet("fish", attack=1, health=2)])
    r.battle()
    events = [line for line in r.trace if "friend_ahead_attacks you:kangaroo" in line]
    assert len(events) == 1
    assert kangaroo.attack == level


@pytest.mark.parametrize("level,xp", LEVELS)
def test_ox_per_turn_limit_carries_from_shop_into_battle(level, xp):
    g = game()
    ox = pet("ox", level)
    g.state.team = [pet("fish"), ox]
    for _ in range(level + 1):
        if g.state.team[0] is ox:
            g.state.team.insert(0, pet("fish"))
        feed(g, "sleeping_pill")
    assert ox.ability_uses == level and ox.attack == 1 + level and ox.perk == "melon"
    r = battle_runtime([pet("fish"), ox], [], catalog(), random.Random(1))
    battle_ox = r.teams[0][1]
    r.faint(0, r.teams[0][0])
    r.drain()
    assert battle_ox.attack == ox.attack and battle_ox.ability_uses == level
    g.opponent_provider = lambda *args: []
    g.step(Action(ActionKind.END_TURN))
    assert ox.ability_uses == 0


@pytest.mark.parametrize("level,damage", [(1, 2), (2, 5), (3, 7)])
@pytest.mark.parametrize("front", [False, True])
def test_badger_adjacency_percentage_and_meat_bone_exclusion(level, damage, front):
    badger = pet("badger", level, attack=5, perk="meat_bone")
    before, behind, enemy = pet("fish", health=30), pet("fish", health=30), pet("fish", health=30)
    team = [badger, behind] if front else [before, badger, behind]
    r = runtime(team, [enemy])
    r.faint(0, badger)
    r.drain()
    assert behind.health == 30 - damage
    assert enemy.health == (30 - damage if front else 30)
    assert before.health == (30 if front else 30 - damage)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_rabbit_three_feeds_including_self_and_counter_reset(level, xp):
    g = game()
    rabbit = pet("rabbit", level)
    g.state.team = [rabbit]
    for _ in range(4):
        feed(g, "apple")
    assert rabbit.health == 2 + 4 + 3 * level and rabbit.ability_uses == 3
    assert rabbit.attack == 1 + 4
    g.opponent_provider = lambda *args: []
    g.step(Action(ActionKind.END_TURN))
    assert rabbit.ability_uses == 0
    before = rabbit.health
    feed(g, "honey")
    assert rabbit.health == before + level


def test_pill_does_not_resurrect_eater_or_waste_rabbit_charge_or_trigger_hurt():
    g = game()
    peacock, rabbit = pet("peacock", perk="melon"), pet("rabbit")
    g.state.team = [peacock, rabbit]
    feed(g, "sleeping_pill")
    assert g.state.team == [rabbit]
    assert rabbit.ability_uses == 0 and peacock.attack == 2


@pytest.mark.parametrize("level,xp", LEVELS)
def test_dog_shop_summon_buff_is_temporary_and_excludes_itself(level, xp):
    g = game()
    dog = pet("dog", level)
    g.state.team = [dog]
    g.state.shop[0] = ShopItem("pet", "fish", 3, pet=pet("fish"))
    g.step(Action(ActionKind.BUY_PET, 0))
    assert (dog.attack, dog.health) == (3, 2)
    assert (dog.temporary_attack, dog.temporary_health) == (2 * level, level)
    g.opponent_provider = lambda *args: []
    g.step(Action(ActionKind.END_TURN))
    assert (dog.attack, dog.health, dog.temporary_attack, dog.temporary_health) == (3, 2, 0, 0)


@pytest.mark.parametrize("level,xp", LEVELS)
def test_sheep_two_rams_at_all_levels_and_horse_dog_react(level, xp):
    sheep, horse, dog = pet("sheep", level), pet("horse"), pet("dog")
    r = runtime([sheep, horse, dog])
    r.faint(0, sheep)
    r.drain()
    rams = [p for p in r.teams[0] if p.spec_id == "ram"]
    assert len(rams) == 2
    assert all((p.attack, p.health) == (2 * level + 1, 2 * level) for p in rams)
    assert (dog.attack, dog.health) == (7, 4)


def test_summons_respect_five_live_slots_even_with_dead_tombstones():
    sheep = pet("sheep", perk="honey")
    r = runtime([sheep] + [pet("fish") for _ in range(4)])
    r.faint(0, sheep)
    r.drain()
    assert len(r.teams[0]) == 5
    assert sum(p.spec_id == "ram" for p in r.teams[0]) == 1
    assert not any(p.spec_id == "bee" for p in r.teams[0])


@pytest.mark.parametrize("species", ["dog", "ox"])
def test_spider_honey_creates_pet_first_but_places_bee_ahead(species):
    class ChoosePet(random.Random):
        def choice(self, values):
            return species if species in values else super().choice(values)

    spider = pet("spider", perk="honey")
    r = runtime([spider])
    r.rng = ChoosePet(6)
    r.faint(0, spider)
    r.drain()
    bee, summoned = r.teams[0]
    assert bee.spec_id == "bee" and summoned.spec_id == species
    if species == "dog":
        assert (summoned.attack, summoned.health) == (4, 3)
    else:
        r.faint(0, bee)
        r.drain()
        assert summoned.attack == 3 and summoned.perk == "melon"


@pytest.mark.parametrize("level,xp", LEVELS)
def test_rat_enemy_summons_do_not_notify_horse_or_dog(level, xp):
    rat, horse, dog = pet("rat", level), pet("horse"), pet("dog")
    r = runtime([rat], [horse, dog])
    r.faint(0, rat)
    r.drain()
    rats = [p for p in r.teams[1] if p.spec_id == "dirty_rat"]
    assert len(rats) == level and all((p.attack, p.health) == (1, 1) for p in rats)
    assert r.teams[1][:level] == rats and (dog.attack, dog.health) == (3, 2)
    assert not any("friend_summoned" in line for line in r.trace)
    g = game()
    g.state.team = [pet("rat", level)]
    feed(g, "sleeping_pill")
    assert not g.state.team


@pytest.mark.parametrize("level,xp", LEVELS)
@pytest.mark.parametrize("species", TIER3)
def test_spider_can_summon_every_real_tier3_species_at_each_ability_level(level, xp, species):
    class PickSpecies(random.Random):
        def choice(self, options):
            return species if species in options else super().choice(options)

    spider = pet("spider", level)
    r = runtime([spider])
    r.rng = PickSpecies(9)
    r.faint(0, spider)
    r.drain()
    assert len(r.teams[0]) == 1
    summoned = r.teams[0][0]
    assert (summoned.spec_id, summoned.attack, summoned.health, summoned.level) == (
        species,
        2 * level,
        2 * level,
        level,
    )
    assert r.catalog.pets[summoned.spec_id].abilities  # Not a blank stand-in.
    assert summoned.ability_uses == 0


def test_spider_summoned_start_battle_pet_does_not_retroactively_fire():
    class PickDolphin(random.Random):
        def choice(self, options):
            return "dolphin" if "dolphin" in options else super().choice(options)

    g = game()
    g.rng = PickDolphin(3)
    g.state.team = [pet("spider")]
    feed(g, "sleeping_pill")
    assert g.state.team[0].spec_id == "dolphin"
    # A shop-summoned Dolphin participates normally in the next battle's start.
    r = battle_runtime(g.state.team, [pet("fish", health=5)], catalog(), random.Random(1))
    r.phase("start_battle")
    assert r.teams[1][0].health == 1
    # A Dolphin summoned during combat has already missed the start phase.
    spider = pet("spider")
    r = runtime([spider], [pet("fish", health=5)])
    r.rng = PickDolphin(4)
    r.faint(0, spider)
    r.drain()
    assert r.teams[1][0].health == 5


def test_missing_spider_pool_raises_and_rolls_back_pill_purchase():
    g = game()
    g.catalog = replace(g.catalog, implemented_shop_tiers=(1, 2))
    g.state.team = [pet("spider")]
    g.state.shop[0] = ShopItem("food", "sleeping_pill", 1)
    before = deepcopy(g.state.to_dict()), g.rng.getstate()
    with pytest.raises(ContentNotReady, match="Tier 3"):
        g.step(Action(ActionKind.BUY_FOOD, 0, 0))
    assert (g.state.to_dict(), g.rng.getstate()) == before


def test_cupcake_expires_but_rabbit_food_buff_is_permanent():
    g = game()
    fish, rabbit = pet("fish"), pet("rabbit")
    g.state.team = [fish, rabbit]
    feed(g, "cupcake")
    assert (fish.attack, fish.health, attack(fish), health(fish)) == (2, 4, 5, 7)
    g.opponent_provider = lambda *args: []
    g.step(Action(ActionKind.END_TURN))
    assert (fish.attack, fish.health, fish.temporary_attack, fish.temporary_health) == (2, 4, 0, 0)


def test_temporary_health_absorbs_shop_damage_before_permanent_health():
    g = game()
    fish = pet("fish")
    g.state.team = [fish, pet("hedgehog")]
    feed(g, "cupcake")
    feed(g, "sleeping_pill", 1)
    assert fish.health == 3 and fish.temporary_health == 1


def test_melon_consumption_blocks_hurt_but_partial_damage_triggers_it():
    peacock = pet("peacock", health=20, perk="melon")
    r = runtime([peacock])
    r.damage_batch([(0, peacock, 4)])
    r.drain()
    assert peacock.health == 20 and peacock.attack == 2 and peacock.perk is None
    peacock.perk = "melon"
    r.damage_batch([(0, peacock, 21)])
    r.drain()
    assert peacock.health == 19 and peacock.attack == 5


def test_meat_bone_adds_three_attack_damage_not_base_attack():
    a, b = pet("fish", attack=1, health=1, perk="meat_bone"), pet("fish", attack=1, health=4)
    r = runtime([a], [b])
    result, _ = r.battle()
    assert result is BattleOutcome.DRAW
    assert a.attack == 1


def test_simultaneous_lethal_hits_do_not_let_ant_resurrect_dead_friend():
    ant, fish = pet("ant", health=1), pet("fish", health=1)
    r = runtime([ant, fish])
    r.damage_batch([(0, ant, 2), (0, fish, 2)])
    r.drain()
    assert not r.teams[0]


def test_gym_observes_perks_and_counters_and_preserves_snapshot_uses():
    pytest.importorskip("gymnasium")
    from sap_rl_lab.env import SapAutoBattlerEnv

    with pytest.raises(ValueError, match="not training-ready"):
        SapAutoBattlerEnv(catalog())
    env = SapAutoBattlerEnv(catalog(), allow_development=True)
    env.reset(seed=6)
    ox = pet("ox", 3, perk="melon", ability_uses=2)
    env.engine.state.team = [ox]
    before = env._observation()
    ox.ability_uses = 3
    after = env._observation()
    assert (before["team"] != after["team"]).sum() == 1
    assert env.observation_space.contains(after)
    ox.perk = "meat_bone"
    assert (after["team"] != env._observation()["team"]).sum() == 2
    assert PetSnapshot.from_pet(ox).to_pet().ability_uses == 3


@pytest.mark.parametrize("seed", range(20))
def test_whole_episode_mask_state_and_deterministic_replay(seed):
    recorder = ReplayRecorder(game(max_actions_per_turn=12), seed=seed)
    g = recorder.engine
    rng = random.Random(seed + 500)
    count = 0
    while not (g.state.terminated or g.state.truncated):
        ids = g.legal_action_ids()
        assert {i for i, allowed in enumerate(g.action_mask()) if allowed} == set(ids)
        result = recorder.step(rng.choice(ids))
        count += 1
        assert count <= g.config.max_turns * g.config.max_actions_per_turn
        assert len(g.state.team) <= 5 and len(g.state.shop) == 9 and g.state.gold >= 0
        assert all(0 < health(p) <= 50 and 0 <= attack(p) <= 50 for p in g.state.team)
        assert result.info.get("reason") != "shop_action_limit"
        assert g.unlocked_tier <= 2
    verify_replay(recorder.finish())


@pytest.mark.parametrize("seed", range(25))
def test_generated_battles_cover_full_roster_without_mutating_inputs(seed):
    rng = random.Random(seed + 8000)
    c = catalog()
    for _ in range(40):
        teams = [
            [
                pet(
                    rng.choice(c.rollable_pet_ids),
                    rng.randint(1, 3),
                    attack=rng.randint(1, 30),
                    health=rng.randint(1, 50),
                    perk=rng.choice([None, "honey", "meat_bone", "melon"]),
                )
                for _ in range(rng.randint(1, 5))
            ]
            for _ in range(2)
        ]
        before = deepcopy(teams)
        r = battle_runtime(*teams, c, rng)
        outcome, attacks = r.battle()
        assert isinstance(outcome, BattleOutcome) and attacks <= 200
        assert teams == before
        assert all(len(t) <= 5 for t in r.teams)
        assert all(0 < health(p) <= 50 and 0 <= attack(p) <= 50 for t in r.teams for p in t)
