"""Full-pack specification tests; not official-client parity recordings."""

import random
from dataclasses import asdict

import numpy as np
import pytest

from sap_rl_lab.fullpack.actions import Action, ActionKind
from sap_rl_lab.fullpack.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.fullpack.domain import GameConfig, GameState, ShopItem
from sap_rl_lab.fullpack.engine import AutoBattler, make_pet, resolve_battle
from sap_rl_lab.fullpack.env import SapAutoBattlerEnv
from sap_rl_lab.fullpack.late_events import LateGameEventRuntime, late_battle_runtime


def catalog(tier=6):
    return load_catalog_by_id("turtle-v0.46-full-v8" if tier == 6 else "turtle-v0.46-tier5-v7")


def pet(name, level=1, **kwargs):
    p = make_pet(catalog(), name)
    p.experience = (1, 3, 6)[level - 1]
    for key, value in kwargs.items():
        setattr(p, key, value)
    return p


def runtime(team, enemies=None, combat=True):
    return LateGameEventRuntime(
        catalog(),
        random.Random(17),
        [team, enemies or []],
        combat=combat,
        state=None if combat else GameState(team=team, shop=[None] * 9, turn=11),
    )


def shop(team, tier=6):
    e = AutoBattler(catalog(tier))
    e.reset(17)
    e.state = GameState(turn=11, gold=100, team=team, shop=[None] * 9)
    return e


def feed(e, name, target=0):
    food = e.catalog.foods[name]
    e.state.shop[0] = ShopItem("food", name, food.cost)
    e.step(Action(ActionKind.BUY_FOOD, 0, target))


def test_complete_roster_contract_preserves_historical_catalog():
    from sap_rl_lab.catalog import load_catalog_by_id as historical

    old = historical("turtle-v0.46-tier4-v6")
    before = catalog_digest(old)
    for tier in (5, 6):
        c = catalog(tier)
        assert len(c.rollable_pet_ids) == 60
        assert len(c.rollable_food_ids) == 18
        assert {
            t: sum(p.tier == t and not p.token for p in c.pets.values()) for t in range(1, 7)
        } == dict.fromkeys(range(1, 7), 10)
        assert all(asdict(c.pets[k]) == asdict(v) for k, v in old.pets.items())
        assert AutoBattler(c).config == GameConfig.turtle_full(tier)
    assert catalog_digest(old) == before
    assert catalog_digest(catalog(5)) != catalog_digest(catalog(6))
    with pytest.raises(ValueError, match="contract"):
        AutoBattler(catalog(), GameConfig.turtle_tier4())


def test_historical_battles_without_late_triggers_preserve_outcomes():
    from sap_rl_lab.catalog import load_catalog_by_id as historical_catalog
    from sap_rl_lab.engine import resolve_battle as historical_battle

    old, new = historical_catalog("turtle-v0.46-tier4-v6"), catalog()
    rng = random.Random(91841)
    # Rat's enemy-summon notification is an explicitly corrected rule; Parrot
    # could copy Rat. Keep those out of this unchanged-behavior comparison.
    roster = [p for p in old.rollable_pet_ids if p not in {"rat", "parrot"}]
    for seed in range(300):
        teams = [
            [
                pet(
                    rng.choice(roster),
                    rng.randint(1, 3),
                    attack=rng.randint(1, 30),
                    health=rng.randint(1, 30),
                )
                for _ in range(rng.randint(1, 5))
            ]
            for _ in range(2)
        ]
        a = historical_battle(*teams, old, random.Random(seed))
        b = resolve_battle(*teams, new, random.Random(seed))
        # Late runtime labels summons differently; compare battle results, not
        # cosmetic trace text. Determinism is checked separately by stress.
        old_result, new_result = asdict(a), asdict(b)
        old_result.pop("trace")
        new_result.pop("trace")
        assert old_result == new_result, f"Historical differential case {seed}"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_boar_before_attack_changes_this_attack_and_health(level):
    owner, enemy = pet("boar", level), pet("fish", attack=1, health=50)
    r = runtime([owner], [enemy])
    r.before_exchange(owner, enemy)
    assert (owner.attack, owner.health) == (10 + 4 * level, 6 + 2 * level)
    assert r.attack_damage(owner) == 10 + 4 * level


@pytest.mark.parametrize(
    "name,trigger,atk,hp", [("mammoth", "faint", 2, 2), ("piranha", "hurt", 3, 0)]
)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_all_friend_buffs_exclude_owner_and_dead(name, trigger, atk, hp, level):
    owner, friend, dead = pet(name, level), pet("fish"), pet("fish", health=0)
    r = runtime([owner, friend, dead])
    r.emit(0, owner, trigger)
    r.drain()
    assert (friend.attack, friend.health) == (2 + atk * level, 3 + hp * level)
    assert dead.health <= 0


@pytest.mark.parametrize("level", [1, 2, 3])
def test_leopard_unique_targets_ceil_and_melon(level):
    owner = pet("leopard", level, attack=7)
    enemies = [pet("fish", health=30) for _ in range(5)]
    r = runtime([owner], enemies)
    r.phase("start_battle")
    assert sum(p.health == 26 for p in enemies) == level
    assert sum(p.health == 30 for p in enemies) == 5 - level
    shielded = pet("fish", health=30, perk="melon")
    runtime([owner], [shielded]).phase("start_battle")
    assert shielded.health == 30 and shielded.perk is None


@pytest.mark.parametrize("level", [1, 2, 3])
def test_gorilla_quota_coconut_and_no_hurt_for_blocked_damage(level):
    owner = pet("gorilla", level, health=40)
    r = runtime([owner])
    for _ in range(level):
        r.damage_batch([(0, owner, 1)])
        r.drain()
        assert owner.perk == "coconut"
        before = owner.health
        r.damage_batch([(0, owner, 100)])
        r.drain()
        assert owner.health == before and owner.perk is None
    assert owner.ability_uses == level
    r.damage_batch([(0, owner, 1)])
    r.drain()
    assert owner.perk is None


def test_coconut_blocks_peanut_and_steak_once_without_mutating_input():
    left, right = pet("scorpion", attack=1), pet("fish", attack=1, health=20, perk="coconut")
    r = runtime([left], [right])
    left.perk = "peanut"
    r.exchange_damage(left, right, 1, 1)
    r.drain()
    assert right.health == 20 and right.perk is None
    steak = pet("fish", attack=3, perk="steak")
    assert r.attack_damage(steak) == 23 and r.attack_damage(steak) == 3
    original = [pet("boar", perk="steak")]
    before = [asdict(p) for p in original]
    resolve_battle(original, [pet("fish")], catalog(), random.Random(1))
    assert [asdict(p) for p in original] == before


@pytest.mark.parametrize("level", [1, 2, 3])
def test_cat_additive_multipliers_two_foods_not_two_targets(level):
    cats = [pet("cat", level), pet("cat")]
    friend = pet("fish")
    e = shop([friend] + cats)
    feed(e, "pear")
    assert (friend.attack, friend.health) == (2 + 2 * (level + 2), 3 + 2 * (level + 2))
    feed(e, "pizza")
    assert [p.ability_uses for p in cats] == [2, 2]
    before = friend.attack
    feed(e, "pear")
    assert friend.attack == before + 2
    feed(e, "chocolate")
    assert [p.ability_uses for p in cats] == [2, 2]


def test_chocolate_is_xp_not_merge_and_level6_reward_pool():
    friend = pet("fish", experience=2)
    other = pet("ant")
    e = shop([friend, other])
    feed(e, "chocolate")
    assert friend.experience == 3 and (friend.attack, friend.health) == (3, 4)
    assert (other.attack, other.health) == (3, 3)
    choices = [s for s in e.state.shop if s and s.choice_group is not None]
    assert len(choices) == 2 and all(e.catalog.pets[s.item_id].tier == 6 for s in choices)
    assert choices[0].choice_group == choices[1].choice_group


def test_cow_chocolate_milk_and_cat_does_not_multiply_xp():
    cow, cat = pet("cow"), pet("cat", 3)
    e = shop([cow, cat])
    feed(e, "chocolate")
    assert cow.experience == 2 and cat.ability_uses == 0
    assert [s.item_id for s in e.state.shop if s and s.kind == "food"] == ["chocolate_milk"] * 2


def test_max_level_chocolate_still_buffs_and_does_not_refresh_quota():
    dragon, cat = pet("dragon", 3, ability_uses=4), pet("cat", 3)
    e = shop([dragon, cat])
    feed(e, "chocolate")
    assert (dragon.attack, dragon.health, dragon.experience) == (4, 9, 6)
    assert dragon.ability_uses == 4 and cat.ability_uses == 0
    assert not any(s and s.choice_group is not None for s in e.state.shop)


def test_chocolate_levelup_refreshes_cow_and_milk_uses_new_level():
    cow = pet("cow", experience=2, ability_uses=1)
    e = shop([cow])
    feed(e, "chocolate")
    assert cow.level == 2 and cow.ability_uses == 0
    assert [s.item_id for s in e.state.shop if s and s.kind == "food"] == [
        "better_chocolate_milk",
        "better_chocolate_milk",
    ]


@pytest.mark.parametrize("level", [1, 2, 3])
def test_tiger_fly_repeated_summons_use_tiger_level_and_consume_quota(level):
    fly, tiger, friend = pet("fly"), pet("tiger", level), pet("fish")
    r = runtime([friend, fly, tiger])
    r.faint(0, friend)
    r.drain()
    tokens = [p for p in r.alive(0) if p.spec_id == "zombie_fly"]
    assert sorted(p.attack for p in tokens) == sorted([4, 4 * level])
    assert fly.ability_uses == 2
    extra = pet("fish")
    r.teams[0].insert(0, extra)
    r.faint(0, extra)
    r.drain()
    assert fly.ability_uses == 3
    assert sum(p.spec_id == "zombie_fly" for p in r.alive(0)) == 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_dragon_four_buys_including_shop_merge_not_team_merge(level):
    dragon = pet("dragon", level)
    e = shop([dragon])
    for i in range(5):
        e.state.shop[0] = ShopItem("pet", "fish", 3, pet=pet("fish"))
        e.step(Action(ActionKind.BUY_PET, 0) if i == 0 else Action(ActionKind.MERGE, 0, 1))
    # Fish gains four merging stat points and four Dragon buffs, Dragon excludes itself.
    assert e.state.team[1].attack == 2 + 4 + 4 * level
    assert dragon.attack == 3 + 1  # One departing-level-1 Fish buff; not a Dragon self-buff.
    assert dragon.ability_uses == 4


def test_fly_tokens_do_not_retrigger_and_shop_quota_survives_into_battle():
    fly, friend = pet("fly", 2), pet("fish")
    r = runtime([friend, fly], combat=False)
    r.faint(0, friend)
    r.drain()
    token = r.teams[0][0]
    assert (token.spec_id, token.attack, token.health, token.level) == ("zombie_fly", 8, 8, 2)
    assert fly.ability_uses == 1
    r.faint(0, token)
    r.drain()
    assert fly.ability_uses == 1 and r.teams[0] == [fly]
    b = late_battle_runtime([fly], [], catalog(), random.Random(1))
    assert b.teams[0][0].ability_uses == 1


def test_fly_waits_for_space_without_spending_charge_and_mushroom_order():
    fly = pet("fly")
    rooster = pet("rooster", 3)
    r = runtime([rooster, pet("fish"), pet("fish"), pet("fish"), fly])
    r.faint(0, rooster)
    r.drain()
    assert len(r.alive(0)) == 5 and fly.ability_uses == 0
    snake = pet("snake", perk="mushroom")
    r = runtime([snake, fly])
    r.faint(0, snake)
    r.drain()
    assert [p.spec_id for p in r.teams[0]] == ["zombie_fly", "snake", "fly"]
    revived = r.teams[0][1]
    assert (revived.attack, revived.health, revived.perk) == (1, 1, None)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_tiger_uses_own_level_and_cannot_extend_snake_quota(level):
    snake, tiger, target = pet("snake"), pet("tiger", level), pet("fish", health=50)
    r = runtime([snake, tiger], [target])
    r.emit(0, snake, "friend_ahead_attacks")
    r.drain()
    assert target.health == 50 - 5 - 5 * level and snake.ability_uses == 2
    snake.ability_uses = 4
    before = target.health
    r.emit(0, snake, "friend_ahead_attacks")
    r.drain()
    assert target.health == before - 5 and snake.ability_uses == 5
    r.emit(0, snake, "friend_ahead_attacks")
    r.drain()
    assert snake.ability_uses == 5


def test_tiger_does_not_repeat_shop_or_perks_and_repeats_faint_summons():
    monkey, tiger = pet("monkey"), pet("tiger", 3)
    runtime([monkey, tiger], combat=False).phase("end_turn")
    assert (monkey.attack, monkey.health) == (3, 4)
    cricket = pet("cricket", perk="mushroom")
    r = runtime([cricket, tiger])
    r.faint(0, cricket)
    r.drain()
    assert len(r.teams[0]) == 4
    assert sum(p.spec_id == "cricket" for p in r.teams[0]) == 1
    assert len([p for p in r.teams[0] if p.spec_id == "zombie_cricket"]) == 2


def test_fullpack_observation_distinguishes_all_perks_and_four_five_uses():
    env = SapAutoBattlerEnv(catalog=catalog(), allow_development=True)
    env.reset(seed=1)
    env.engine.state.team = [pet("snake", ability_uses=4)]
    a = env._observation()
    env.engine.state.team[0].ability_uses = 5
    b = env._observation()
    assert not np.array_equal(a["team"], b["team"])
    for perk in ("peanut", "bread", "coconut", "mushroom", "steak"):
        env.engine.state.team[0].perk = perk
        obs = env._observation()
        assert env.observation_space.contains(obs)
        assert obs["team"][0, -5:].sum() == 1
    env.close()


def test_rat_does_not_give_enemy_summon_buffs():
    rat, turkey = pet("rat"), pet("turkey")
    r = runtime([rat], [turkey])
    r.faint(0, rat)
    r.drain()
    dirty_rat = r.teams[1][0]
    assert (dirty_rat.spec_id, dirty_rat.attack, dirty_rat.health) == ("dirty_rat", 1, 1)


@pytest.mark.parametrize("tiger_level", [1, 2, 3])
def test_tiger_whale_releases_both_swallowed_friends(tiger_level):
    first, second, whale, tiger = (
        pet("fish"),
        pet("beaver"),
        pet("whale"),
        pet("tiger", tiger_level),
    )
    r = runtime([first, second, whale, tiger])
    r.phase("start_battle")
    assert [p.spec_id for p in r.teams[0]] == ["whale", "tiger"]
    r.faint(0, whale)
    r.drain()
    assert [p.spec_id for p in r.teams[0]] == ["beaver", "fish", "tiger"]
    assert [p.level for p in r.teams[0]][:2] == [1, tiger_level]


@pytest.mark.parametrize("tiger_level", [1, 2, 3])
def test_tiger_crocodile_repeats_at_tiger_level(tiger_level):
    r = runtime(
        [pet("crocodile", 3), pet("tiger", tiger_level)], [pet("fish", health=50) for _ in range(5)]
    )
    r.phase("start_battle")
    assert sum("damage_last_enemy" in row for row in r.trace) == 3 + tiger_level


@pytest.mark.parametrize("tier", [5, 6])
@pytest.mark.parametrize("family", ["full_stats", "full_summon", "full_tempo"])
def test_reachable_teacher_episode_and_exact_replay(tier, family):
    from sap_rl_lab.fullpack.baselines import scripted_policy
    from sap_rl_lab.fullpack.replay import ReplayRecorder, verify_replay

    engine = AutoBattler(catalog(tier))
    recorder = ReplayRecorder(engine, seed=801001)
    policy, rng = scripted_policy(family), random.Random(801002)
    for _ in range(900):
        action = policy.choose(engine, rng)
        assert action in engine.legal_actions()
        step = recorder.step(engine.codec.encode(action))
        assert not step.truncated
        if step.terminated:
            break
    else:
        pytest.fail("Teacher episode failed to finish")
    verify_replay(recorder.finish())


def test_late_shop_progression_and_rewards_never_substitute_tiers():
    for tier in (5, 6):
        e = shop([pet("fish")], tier=tier)
        assert e.unlocked_tier == tier
        e._stock_level_up_choices()
        choices = [s for s in e.state.shop if s and s.choice_group]
        assert len(choices) == 2
        assert all(e.catalog.pets[s.item_id].tier == 6 for s in choices)
        seen = set()
        foods = set()
        for _ in range(200):
            e._roll_shop(free=True)
            seen.update(
                e.catalog.pets[s.item_id].tier for s in e.state.shop if s and s.kind == "pet"
            )
            foods.update(
                e.catalog.foods[s.item_id].tier for s in e.state.shop if s and s.kind == "food"
            )
        assert seen == set(range(1, tier + 1))
        assert foods == set(range(1, tier + 1))
