"""Observed numeric components and explicit per-ability rounding contracts."""

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from sap_rl_lab.catalog import _ability, catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import Pet
from sap_rl_lab.events import EventRuntime, percent_amount


def catalog():
    return load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")


def fixture():
    return json.loads(
        (Path(__file__).parent / "fixtures/real_game/dodo_percentage_components.json").read_text()
    )


def test_observed_dodo_chain_uses_updated_attack_and_floors_halves():
    case = fixture()["chain"]
    # Only the passive numeric recipient is replaced; no Scorpion/Crocodile
    # behavior is simulated or claimed to match the full recorded battle.
    team = [
        Pet(
            "swan" if row["species"] == "scorpion" else row["species"],
            row["attack"],
            row["health"],
            experience={1: 1, 2: 3, 3: 6}[row["level"]],
            perk=row.get("perk"),
        )
        for row in case["front_to_back"]
    ]
    runtime = EventRuntime(catalog(), random.Random(9), [team, []], combat=True)
    runtime.phase("start_battle")
    assert [pet.attack for pet in team] == case["observed_attack_front_to_back"]


def test_observed_level_three_dodo_tooltip_distinguishes_floor_from_nearest_even():
    case = fixture()["tooltip"]
    ahead = Pet("swan", 1, 2)
    dodo = Pet("dodo", case["attack"], 7, experience={1: 1, 2: 3, 3: 6}[case["level"]])
    runtime = EventRuntime(catalog(), random.Random(9), [[ahead, dodo], []], combat=True)
    runtime.phase("start_battle")
    assert ahead.attack - 1 == case["observed_current_ability_value"]
    assert round(case["attack"] * case["percent"] / 100) == 14


@pytest.mark.parametrize(
    "value,percent,floor,ceil", [(1, 50, 0, 1), (5, 50, 2, 3), (9, 150, 13, 14)]
)
def test_percentage_primitives_have_explicit_distinct_rounding(value, percent, floor, ceil):
    # Specification boundary checks, not additional recorded-game observations.
    assert percent_amount(value, percent, "floor") == floor
    assert percent_amount(value, percent, "ceil") == ceil


def test_rounding_is_fingerprinted_and_not_uniform_across_pets():
    current = catalog()
    dodo = current.pets["dodo"]
    ability = dodo.abilities[0]
    assert ability.params["rounding"] == "floor"
    assert current.pets["crab"].abilities[0].params["rounding"] == "ceil"
    assert current.pets["badger"].abilities[0].params["rounding"] == "floor"
    old_params = {k: v for k, v in ability.params.items() if k != "rounding"}
    pets = dict(current.pets)
    pets["dodo"] = replace(dodo, abilities=(replace(ability, params=old_params),))
    assert catalog_digest(current) != catalog_digest(replace(current, pets=pets))


@pytest.mark.parametrize("rounding", ["nearest", "", None])
def test_invalid_rounding_is_rejected(rounding):
    with pytest.raises(ValueError, match="rounding"):
        _ability({"trigger": "start_battle", "effect": "share_attack", "rounding": rounding})


def test_rounding_on_non_percentage_effect_is_rejected():
    with pytest.raises(ValueError, match="percentage effect"):
        _ability({"trigger": "start_battle", "effect": "buff", "rounding": "floor"})


def percentage_cases():
    return json.loads(
        (Path(__file__).parent / "fixtures/real_game/crab_badger_percentage_2026.json").read_text()
    )


def test_observed_crab_gains_two_health_from_six_health_friend():
    case = percentage_cases()["crab"]
    crab = Pet("crab", case["attack"], case["health_before"])
    friend = Pet("swan", 5, case["highest_friend_health"])
    runtime = EventRuntime(catalog(), random.Random(9), [[crab, friend], []], combat=True)
    runtime.phase("start_battle")
    assert crab.health == case["health_after"]
    assert crab.health - case["health_before"] == case["observed_health_gain"]


def test_observed_crab_non_half_gain_distinguishes_ceiling_from_nearest():
    case = percentage_cases()["crab_non_half"]
    crab = Pet("crab", case["attack"], case["health_before"], experience=6)
    assert crab.level == case["level"] == 3
    # Unsupported recorded pets have only their observed health transferred.
    friends = [Pet("swan", 1, hp) for hp in case["other_friend_healths"]]
    runtime = EventRuntime(catalog(), random.Random(9), [[crab] + friends, []], combat=True)
    runtime.phase("start_battle")
    assert max(case["other_friend_healths"]) == case["highest_friend_health"]
    assert crab.health == case["health_after"] == 29
    assert crab.health - case["health_before"] == case["observed_health_gain"] == 15
    assert round(case["highest_friend_health"] * 0.75) == 14


def test_observed_badger_twenty_three_attack_deals_eleven_to_friend():
    case = percentage_cases()["badger"]
    badger = Pet("badger", case["attack"], 1)
    friend = Pet("swan", 4, case["friend_health_before"])
    runtime = EventRuntime(catalog(), random.Random(9), [[badger, friend], []], combat=True)
    runtime.faint(0, badger)
    runtime.drain()
    assert friend.health == case["friend_health_after"]
    assert case["friend_health_before"] - friend.health == case["observed_friend_damage"]
