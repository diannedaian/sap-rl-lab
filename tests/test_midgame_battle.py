"""Version-5 combat/copy specifications; component tests, not client recordings."""

import random

import pytest
from test_midgame_components import catalog, pet, runtime

from sap_rl_lab.domain import BattleOutcome
from sap_rl_lab.midgame_events import midgame_battle_runtime


@pytest.mark.parametrize("level", [1, 2, 3])
def test_deer_summons_level_matched_bus_with_chili(level):
    owner = pet("deer", level)
    r = runtime([owner])
    r.faint(0, owner)
    r.drain()
    (bus,) = r.teams[0]
    assert (bus.spec_id, bus.attack, bus.health, bus.level, bus.perk) == (
        "bus",
        5 * level,
        3 * level,
        level,
        "chili",
    )


def test_chili_targets_pre_damage_second_enemy_not_new_summons():
    bus = pet("bus", attack=20, health=20, perk="chili")
    deer, second, third = pet("deer"), pet("fish", health=8), pet("fish", health=9)
    r = runtime([bus], [deer, second, third], combat=True)
    r.exchange_damage(bus, deer, 20, 2)
    r.drain()
    assert second.health == 3 and third.health == 9
    summoned = r.teams[1][0]
    assert (summoned.spec_id, summoned.health, summoned.perk) == ("bus", 3, "chili")


@pytest.mark.parametrize("level", [1, 2, 3])
def test_hippo_knockouts_limit_three_and_multiple_chili_kills(level):
    hippo = pet("hippo", level, attack=15, health=40, perk="chili")
    r = runtime([hippo], [pet("fish") for _ in range(5)], combat=True)
    for _ in range(3):
        opponent = r.alive(1)[0]
        r.exchange_damage(hippo, opponent, hippo.attack, 2)
        r.drain()
    assert hippo.ability_uses == 3
    assert hippo.attack == 15 + 9 * level
    assert not r.teams[1]


def test_hippo_knockout_on_mutual_faint_runs_without_resurrection():
    hippo, enemy = pet("hippo", health=1), pet("fish", attack=20, health=1)
    r = runtime([hippo], [enemy], combat=True)
    r.exchange_damage(hippo, enemy, 4, 20)
    r.drain()
    assert not r.teams[0] and not r.teams[1]
    assert hippo.ability_uses == 1 and hippo.health <= 0
    assert any("knock_out" in line for line in r.trace)


def test_hippo_knockout_waits_for_hurt_and_faint_chain():
    hippo = pet("hippo", attack=10, health=2)
    enemy = pet("blowfish", health=1)
    r = runtime([hippo], [enemy], combat=True)
    r.exchange_damage(hippo, enemy, 10, 1)
    r.drain()
    # Blowfish's lethal Hurt resolves before Hippo's heal. It cannot heal back
    # out of the faint queue and be resurrected by the delayed Knockout.
    assert not r.teams[0]
    hurt_index = next(i for i, s in enumerate(r.trace) if s.startswith("hurt"))
    knockout_index = next(i for i, s in enumerate(r.trace) if s.startswith("knock_out"))
    assert hurt_index < knockout_index


@pytest.mark.parametrize(
    "amount,damage", [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 3), (20, 18)]
)
def test_garlic_version040_floor_does_not_amplify_one_damage(amount, damage):
    target = pet("fish", health=40, perk="garlic")
    r = runtime([target], combat=True)
    r.damage_batch([(0, target, amount)])
    r.drain()
    assert target.health == 40 - damage and target.perk == "garlic"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_parrot_copies_own_level_and_resets_before_start_turn(level):
    dolphin, parrot = pet("dolphin", 3), pet("parrot", level)
    r = runtime([dolphin, parrot])
    r.phase("end_turn")
    assert parrot.copied_ability == "dolphin"
    battle = midgame_battle_runtime([parrot], [pet("fish", health=40)], catalog(), random.Random(1))
    battle.phase("start_battle")
    assert battle.teams[1][0].health == 40 - 4 * level
    assert parrot.copied_ability == "dolphin"  # Battle uses a copy.
    r.phase("start_turn")
    assert parrot.copied_ability is None


def test_parrot_no_immediate_end_turn_buff_or_copied_start_turn_gold():
    bison, parrot, friend = pet("bison"), pet("parrot", 3), pet("fish", 3)
    r = runtime([bison, parrot, friend])
    r.phase("end_turn")
    assert parrot.copied_ability == "bison"
    assert (parrot.attack, parrot.health) == (4, 2)
    parrot.copied_ability = "swan"
    before = r.state.gold
    r.phase("start_turn")
    assert r.state.gold == before


def test_parrot_chain_depends_on_end_turn_attack_priority():
    front = pet("dolphin")
    first, second = pet("parrot", attack=10), pet("parrot", attack=5)
    r = runtime([front, first, second])
    r.phase("end_turn")
    assert first.copied_ability == second.copied_ability == "dolphin"
    first, second = pet("parrot", attack=5), pet("parrot", attack=10)
    runtime([front, first, second]).phase("end_turn")
    assert first.copied_ability == "dolphin" and second.copied_ability == "parrot"


def test_parrot_copied_faint_ability_summons_bus_not_parrot():
    parrot = pet("parrot", 2, copied_ability="deer")
    r = runtime([parrot])
    r.faint(0, parrot)
    r.drain()
    assert [(p.spec_id, p.attack, p.health, p.perk) for p in r.teams[0]] == [
        ("bus", 10, 6, "chili")
    ]


@pytest.mark.parametrize("level", [1, 2, 3])
def test_whale_retains_displayed_stats_not_perks_or_old_memory(level):
    prey = pet(
        "parrot",
        3,
        attack=17,
        health=23,
        temporary_attack=2,
        temporary_health=4,
        perk="melon",
        copied_ability="deer",
        sell_bonus=8,
    )
    whale = pet("whale", level)
    r = runtime([prey, whale], combat=True)
    r.phase("start_battle")
    assert [p.spec_id for p in r.teams[0]] == ["bus", "whale"]
    r.faint(0, whale)
    r.drain()
    released = r.teams[0][-1]
    assert (released.spec_id, released.attack, released.health, released.level) == (
        "parrot",
        19,
        27,
        level,
    )
    assert released.perk is None and released.copied_ability is None
    assert released.ability_uses == released.sell_bonus == 0


def test_whale_turtle_faint_gives_whale_melon_but_released_turtle_no_perk():
    turtle, whale = pet("turtle", attack=12, health=18, perk="honey"), pet("whale")
    r = runtime([turtle, whale], combat=True)
    r.phase("start_battle")
    assert whale.perk == "melon"
    assert r.teams[0][0].spec_id == "bee"
    r.faint(0, whale)
    r.drain()
    released = r.teams[0][-1]
    assert (released.spec_id, released.attack, released.health, released.perk) == (
        "turtle",
        12,
        18,
        None,
    )


def test_whale_pilled_before_battle_cannot_release_an_unswallowed_pet():
    whale = pet("whale")
    r = runtime([pet("fish"), whale])
    r.faint(0, whale)
    r.drain()
    assert [p.spec_id for p in r.teams[0]] == ["fish"]


def test_nested_whale_release_has_no_old_swallow_memory():
    prey, inner, outer = pet("fish", attack=8), pet("whale", attack=10), pet("whale", attack=5)
    r = runtime([prey, inner, outer], combat=True)
    r.phase("start_battle")
    # Both saved starts resolve before faints; outer swallows inner, whose
    # original faint releases its fish. Outer later releases a *fresh* Whale.
    assert [p.spec_id for p in r.teams[0]] == ["fish", "whale"]
    r.faint(0, outer)
    r.drain()
    new_whale = r.teams[0][-1]
    assert new_whale is not inner and new_whale.spec_id == "whale"
    r.faint(0, new_whale)
    r.drain()
    assert [p.spec_id for p in r.teams[0]] == ["fish"]


def test_battle_copies_are_independent_and_hippo_quota_resets_each_battle():
    owner = pet("hippo", attack=40, health=40, ability_uses=3)
    r = midgame_battle_runtime([owner], [pet("fish")], catalog(), random.Random(1))
    outcome, _ = r.battle()
    assert outcome is BattleOutcome.WIN and r.teams[0][0].ability_uses == 1
    assert owner.ability_uses == 3 and (owner.attack, owner.health) == (40, 40)
