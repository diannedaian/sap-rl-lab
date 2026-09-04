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
