"""Transparent policies used as tests, curricula, and evaluation baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .actions import Action, ActionKind
from .engine import AutoBattler


class RandomPolicy:
    def choose(self, engine: AutoBattler, rng: random.Random) -> Action:
        return rng.choice(engine.legal_actions())


class SpendGoldPolicy:
    """A readable heuristic: merge, fill the team, buy food, then reroll or finish."""

    def choose(self, engine: AutoBattler, rng: random.Random) -> Action:
        by_kind: Dict[ActionKind, List[Action]] = {}
        for action in engine.legal_actions():
            by_kind.setdefault(action.kind, []).append(action)
        priorities = [ActionKind.MERGE, ActionKind.BUY_PET, ActionKind.BUY_FOOD]
        for kind in priorities:
            choices = by_kind.get(kind, [])
            if choices:
                return rng.choice(choices)
        if engine.state.gold >= 3 and by_kind.get(ActionKind.ROLL):
            return by_kind[ActionKind.ROLL][0]
        return by_kind[ActionKind.END_TURN][0]


class FocusedPolicy:
    """Two deliberately simple, frozen priorities, not optimized search agents.

    Stats buys high base-stat pets and feeds its strongest pet. Summon prefers
    one or two Horses, Crickets and Honey, and keeps Horses behind other pets.
    Both obey the public action API and bound ordering work before the cutoff.
    """

    def __init__(self, focus: str):
        if focus not in {"stats", "summon"}:
            raise ValueError("focus must be stats or summon")
        self.focus = focus

    def choose(self, engine: AutoBattler, rng: random.Random) -> Action:
        state = engine.state
        if state.actions_this_turn >= engine.config.max_actions_per_turn - 1:
            return Action(ActionKind.END_TURN)
        horses = sum(p.spec_id == "horse" for p in state.team)

        def preference(pet_id: str) -> float:
            if self.focus == "summon":
                return {
                    "horse": 20 if horses == 0 else (12 if horses < 2 else 0),
                    "cricket": 16,
                    "ant": 5,
                }.get(pet_id, 0)
            pet = engine.catalog.pets[pet_id]
            return pet.attack * pet.health + (4 if pet_id == "fish" else 0)

        scored = []
        for action in engine.legal_actions():
            score = None
            if action.kind is ActionKind.BUY_PET:
                score = 30 + preference(state.shop[action.source].item_id)
            elif action.kind is ActionKind.MERGE:
                pet = state.team[action.target]
                level_bonus = 8 if pet.experience in {2, 5} else 0
                score = 20 + preference(pet.spec_id) + level_bonus
            elif action.kind is ActionKind.BUY_FOOD:
                food = engine.catalog.foods[state.shop[action.source].item_id]
                pet = state.team[action.target]
                if food.effect == "buff":
                    if pet.attack < 50 or pet.health < 50:
                        score = 18 + (pet.attack + pet.health) / 100
                elif pet.perk != "honey":
                    score = 32 if self.focus == "summon" and pet.spec_id != "horse" else 2
            if score is not None:
                scored.append((score, action))
        if scored:
            best = max(score for score, _ in scored)
            return rng.choice([action for score, action in scored if score == best])
        if state.gold >= 4:
            return Action(ActionKind.ROLL)

        def order(pet):
            if self.focus == "summon":
                return (pet.spec_id == "horse", -pet.attack, pet.spec_id)
            return (0, -pet.attack * pet.health, pet.spec_id)

        for i in range(len(state.team) - 1):
            if order(state.team[i]) > order(state.team[i + 1]):
                return Action(ActionKind.SWAP, source=i, target=i + 1)
        return Action(ActionKind.END_TURN)


def scripted_policy(name: str):
    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return SpendGoldPolicy()
    return FocusedPolicy(name)


@dataclass(frozen=True)
class EpisodeSummary:
    seed: int
    return_: float
    wins: int
    lives: int
    turns: int
    success: bool
    truncated: bool


def play_episode(
    engine: AutoBattler,
    policy: object,
    seed: int,
    render: bool = False,
) -> EpisodeSummary:
    rng = random.Random(seed + 1_000_003)
    engine.reset(seed=seed)
    episode_return = 0.0
    while not (engine.state.terminated or engine.state.truncated):
        action = policy.choose(engine, rng)  # type: ignore[attr-defined]
        if render:
            print(engine.codec.describe(engine.codec.encode(action)))
        transition = engine.step(action)
        episode_return += transition.reward
    return EpisodeSummary(
        seed=seed,
        return_=episode_return,
        wins=engine.state.wins,
        lives=engine.state.lives,
        turns=engine.state.turn,
        success=engine.state.wins >= engine.config.target_wins,
        truncated=engine.state.truncated,
    )
