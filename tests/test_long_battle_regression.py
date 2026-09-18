"""Boundary tests for the provisional draw rule; not official-client fixtures."""

import json
import random
from dataclasses import asdict, replace
from hashlib import sha256
from importlib import resources

import pytest

from sap_rl_lab.catalog import catalog_digest, catalog_from_dict, load_catalog_by_id
from sap_rl_lab.domain import BattleOutcome, Pet
from sap_rl_lab.events import battle_runtime


def test_identical_low_attack_high_health_teams_finish_without_exception():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    team = [Pet("swan", 1, 50) for _ in range(5)]
    runtime = battle_runtime(team, team, catalog, random.Random(1))
    # Symmetry implies a draw whether through simultaneous fainting or a real
    # attack-limit draw. Do not invent the official limit from conflicting posts.
    outcome, attacks = runtime.battle()
    assert outcome == BattleOutcome.DRAW
    assert attacks == catalog.battle_attack_limit == 30
    assert all(p.health == 20 for p in (runtime.teams[0][0], runtime.teams[1][0]))
    assert "battle draw: attack limit reached (30 exchanges)" in runtime.trace
    assert all(p.attack == 1 and p.health == 50 for p in team)


@pytest.mark.parametrize("count", [29, 30])
@pytest.mark.parametrize("outcome", [BattleOutcome.WIN, BattleOutcome.LOSS, BattleOutcome.DRAW])
def test_natural_result_at_or_before_limit_has_priority(count, outcome):
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    friendly_health = 50 if outcome == BattleOutcome.WIN else count
    enemy_health = 50 if outcome == BattleOutcome.LOSS else count
    runtime = battle_runtime(
        [Pet("swan", 1, friendly_health)],
        [Pet("swan", 1, enemy_health)],
        catalog,
        random.Random(1),
    )
    assert runtime.battle() == (outcome, count)
    assert not any("attack limit reached" in item for item in runtime.trace)


def test_events_on_last_exchange_finish_before_draw_check():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    # Last exchange kills the Cricket. Its Zombie is still summoned before draw.
    runtime = battle_runtime(
        [Pet("cricket", 1, 30)],
        [Pet("swan", 1, 50)],
        catalog,
        random.Random(1),
    )
    assert runtime.battle() == (BattleOutcome.DRAW, 30)
    assert runtime.teams[0][0].spec_id == "zombie_cricket"
    assert runtime.teams[1][0].health == 20
    assert runtime.trace.index("you:cricket summons you:zombie_cricket") < runtime.trace.index(
        "battle draw: attack limit reached (30 exchanges)"
    )


def test_ability_events_do_not_count_as_attack_exchanges():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    runtime = battle_runtime(
        [Pet("mosquito", 1, 50)],
        [Pet("swan", 1, 50)],
        catalog,
        random.Random(1),
    )
    assert runtime.battle() == (BattleOutcome.DRAW, 30)
    assert runtime.teams[1][0].health == 19


def test_unconfigured_historical_runtime_still_raises_technical_guard():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    team = [Pet("swan", 1, 50) for _ in range(5)]
    runtime = battle_runtime(
        team, team, replace(catalog, battle_attack_limit=None), random.Random(1)
    )
    with pytest.raises(RuntimeError, match="attack safety limit"):
        runtime.battle()


def test_limit_is_fingerprinted_but_legacy_fingerprint_stays_identical():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    assert catalog_digest(catalog) != catalog_digest(replace(catalog, battle_attack_limit=40))
    for name in ("turtle-v0.46-tier1", "turtle-v0.46-tier1-rules-v2"):
        legacy = load_catalog_by_id(name)
        payload = asdict(legacy)
        del payload["battle_attack_limit"]
        old_digest = sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        assert catalog_digest(legacy) == old_digest
        assert legacy.battle_attack_limit is None


@pytest.mark.parametrize("limit", [True, False, 0, -1, 201, 30.5, "30"])
def test_catalog_rejects_invalid_draw_limits(limit):
    path = resources.files("sap_rl_lab").joinpath(
        "catalogs", "turtle_v0_46_tier12_curriculum_v4.json"
    )
    raw = json.loads(path.read_text())
    raw["battle_attack_limit"] = limit
    with pytest.raises(ValueError, match="battle_attack_limit"):
        catalog_from_dict(raw)


def test_draw_rule_does_not_hide_event_queue_exceptions(monkeypatch):
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    runtime = battle_runtime([Pet("swan", 1, 50)], [Pet("swan", 1, 50)], catalog, random.Random(1))

    def fail():
        raise RuntimeError("event failure")

    monkeypatch.setattr(runtime, "drain", fail)
    with pytest.raises(RuntimeError, match="event failure"):
        runtime.battle()


def test_attack_limit_draw_advances_turn_without_trophy_life_loss_or_episode_cutoff():
    from sap_rl_lab.actions import Action, ActionKind
    from sap_rl_lab.domain import GameConfig
    from sap_rl_lab.engine import AutoBattler

    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    engine = AutoBattler(
        catalog,
        GameConfig.turtle_curriculum(),
        opponent_provider=lambda *args: [Pet("swan", 1, 50)],
    )
    engine.reset(11)
    engine.state.team = [Pet("swan", 1, 50)]
    result = engine.step(Action(ActionKind.END_TURN))
    assert result.reward == 0
    assert not result.terminated and not result.truncated
    assert (engine.state.turn, engine.state.wins, engine.state.lives) == (2, 0, 5)
    assert engine.state.previous_outcome == BattleOutcome.DRAW
    assert engine.state.team[0].health == 50
    assert result.info["battle_attacks"] == 30
    assert result.info["battle_attack_limit_reached"]
    assert not result.info.get("forced_end_turn")


def test_evaluation_counts_draw_caps_separately_from_forced_shop_endings(monkeypatch):
    from types import SimpleNamespace

    from sap_rl_lab import baselines
    from sap_rl_lab.actions import Action, ActionKind
    from sap_rl_lab.domain import GameConfig
    from sap_rl_lab.env import SapAutoBattlerEnv
    from sap_rl_lab.evaluation import evaluate_policy

    original_reset = SapAutoBattlerEnv.reset

    def reset_with_long_battle(env, *, seed=None, options=None):
        _, info = original_reset(env, seed=seed, options=options)
        env.engine.state.team = [Pet("swan", 1, 50)]
        env.engine.opponent_provider = lambda *args: [Pet("swan", 1, 50)]
        return env._observation(), info

    monkeypatch.setattr(SapAutoBattlerEnv, "reset", reset_with_long_battle)
    monkeypatch.setattr(
        baselines,
        "scripted_policy",
        lambda name: SimpleNamespace(choose=lambda *args: Action(ActionKind.END_TURN)),
    )
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    result = evaluate_policy(
        "random",
        episodes=2,
        seed=50,
        env_kwargs={
            "catalog": catalog,
            "allow_development": True,
            "config": GameConfig.turtle_curriculum(max_turns=2),
        },
    )
    assert result["battle_attack_limit"] == 30
    assert result["battle_attack_limit_draws"] == 4
    assert result["battle_attack_limit_draw_rate"] == 1
    assert result["battle_attack_limit_episode_rate"] == 1
    assert result["forced_end_turns"] == 0
    assert result["environment_contract"]["catalog_sha256"] == catalog_digest(catalog)
    assert all(row["battle_attack_limit_draws"] == 2 for row in result["episode_results"])
    # Only the explicitly configured episode turn bound truncates these runs.
    assert all(row["final_turn"] == 3 for row in result["episode_results"])
