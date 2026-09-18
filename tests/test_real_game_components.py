"""Limited observed historical components; never full client-parity certification."""

import json
import random
from pathlib import Path

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import load_catalog_by_id
from sap_rl_lab.domain import BattleOutcome, GameConfig, GameState, Pet, ShopItem
from sap_rl_lab.engine import AutoBattler, make_pet
from sap_rl_lab.events import EventRuntime, battle_runtime


def test_draw_limit_component_matches_observed_historical_count():
    case = json.loads(
        (Path(__file__).parent / "fixtures/real_game/long_battle_tapir_2022.json").read_text()
    )
    count = case["observed_result"]["attack_exchanges"]
    times = case["approximate_attack_frame_seconds"]
    assert len(times) == len(set(times)) == count
    assert times == sorted(times)
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    assert catalog.battle_attack_limit == count
    # Transfer only the observed generic boundary/queue-order component to
    # supported pets. These are NOT the teams or full trace from the recording.
    runtime = battle_runtime(
        [Pet("cricket", 1, count)], [Pet("swan", 1, 50)], catalog, random.Random(9)
    )
    assert runtime.battle() == (BattleOutcome.DRAW, count)
    assert runtime.attack_limit_reached
    assert runtime.teams[0][0].spec_id == "zombie_cricket"
    summon = next(i for i, line in enumerate(runtime.trace) if "summons" in line)
    draw = next(i for i, line in enumerate(runtime.trace) if "attack limit reached" in line)
    assert summon < draw


def test_observed_cricket_pill_level_and_ox_before_summon():
    case = json.loads(
        (Path(__file__).parent / "fixtures/real_game/cricket_ox_pill_2022.json").read_text()
    )
    team = [Pet(**p) for p in case["team_front_to_back"]]
    r = EventRuntime(
        load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4"),
        random.Random(9),
        [team, []],
        combat=False,
    )
    r.faint(0, team[case["action"]["target"]])
    r.drain()
    for species, expected in case["observed_result"].items():
        if species == "trigger_order":
            continue
        matching = [p for p in team if p.spec_id == species]
        assert len(matching) == 1
        for field, value in expected.items():
            assert getattr(matching[0], field) == value
    ox_event = next(i for i, t in enumerate(r.trace) if "friend_ahead_faints" in t)
    summon_event = next(i for i, t in enumerate(r.trace) if "summons" in t)
    assert ox_event < summon_event


def test_observed_recent_giraffe_worm_start_turn_not_end_turn():
    case = json.loads(
        (Path(__file__).parent / "fixtures/real_game/giraffe_worm_start_2026.json").read_text()
    )
    team = [Pet(**p) for p in case["team_front_to_back"]]
    state = GameState(turn=4, team=team, shop=[None] * 9)
    runtime = EventRuntime(
        load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4"),
        random.Random(9),
        [team, []],
        combat=False,
        state=state,
    )
    runtime.phase("end_turn")
    assert [(p.attack, p.health) for p in team] == [(6, 6), (1, 2), (3, 2), (2, 2), (2, 5)]
    assert not any(state.shop)
    state.turn = 5
    runtime.phase("start_turn")
    assert [(p.attack, p.health) for p in team] == [(7, 7), (1, 2), (3, 2), (2, 2), (2, 5)]
    stocked = [item for item in state.shop if item]
    assert len(stocked) == 1
    assert stocked[0].item_id == "apple" and stocked[0].cost == 2
    assert [line.split()[1] for line in runtime.trace] == ["you:worm", "you:giraffe"]


def test_observed_recent_ant_merges_stock_linked_paid_choices():
    case = json.loads(
        (Path(__file__).parent / "fixtures/real_game/ant_upgrade_choices_2026.json").read_text()
    )
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")

    class ObservedOffers(random.Random):
        def __init__(self):
            super().__init__(9)
            self.offers = iter(case["observed_choices"])

        def choice(self, candidates):
            chosen = next(self.offers)
            assert chosen in candidates
            assert all(catalog.pets[p].tier == 3 for p in candidates)
            return chosen

    engine = AutoBattler(catalog, GameConfig.turtle_curriculum())
    engine.reset(9)
    engine.rng = ObservedOffers()
    engine.state = GameState(
        turn=4,
        gold=10,
        team=[Pet(**p) for p in case["team_front_to_back"]],
        shop=[
            ShopItem("pet", "ant", 3, frozen=True, pet=make_pet(catalog, "ant")),
            ShopItem("pet", "ant", 3, frozen=True, pet=make_pet(catalog, "ant")),
            ShopItem("pet", "pig", 3, pet=make_pet(catalog, "pig")),
            ShopItem("food", "honey", 3),
            None,
            ShopItem("food", "apple", 2),
            None,
            None,
            None,
        ],
    )
    engine.step(Action(ActionKind.MERGE, 0, 0))
    assert engine.state.gold == 7
    assert (
        engine.state.team[0].attack,
        engine.state.team[0].health,
        engine.state.team[0].level,
    ) == (4, 4, 1)
    engine.step(Action(ActionKind.MERGE, 1, 0))
    assert engine.state.gold == 4
    assert (
        engine.state.team[0].attack,
        engine.state.team[0].health,
        engine.state.team[0].level,
    ) == (5, 5, 2)
    choices = [item for item in engine.state.shop if item and item.choice_group]
    assert [item.item_id for item in choices] == case["observed_choices"]
    assert choices[0].choice_group == choices[1].choice_group
    assert all(item.cost == 3 for item in choices)
    engine.step(Action(ActionKind.BUY_FOOD, 5, 0))
    assert engine.state.gold == 2
    assert (engine.state.team[0].attack, engine.state.team[0].health) == (6, 6)
    engine.step(Action(ActionKind.SELL, 3))
    assert engine.state.gold == 3
    engine.step(Action(ActionKind.BUY_PET, 0))
    assert engine.state.gold == 0
    assert any(p.spec_id == "giraffe" for p in engine.state.team)
    assert not any(s and s.item_id in case["observed_choices"] for s in engine.state.shop)
    assert any(s and s.item_id == "pig" for s in engine.state.shop)
