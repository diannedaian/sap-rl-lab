"""Serializable opponent snapshots and mixtures for reproducible league training."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import Catalog
from .domain import GameConfig, Pet
from .engine import OpponentProvider, default_opponent


@dataclass(frozen=True)
class PetSnapshot:
    spec_id: str
    attack: int
    health: int
    experience: int = 1
    perk: Optional[str] = None

    @classmethod
    def from_pet(cls, pet: Pet) -> PetSnapshot:
        return cls(
            spec_id=pet.spec_id,
            attack=pet.attack,
            health=pet.health,
            experience=pet.experience,
            perk=pet.perk,
        )

    def to_pet(self) -> Pet:
        return Pet(
            spec_id=self.spec_id,
            attack=self.attack,
            health=self.health,
            experience=self.experience,
            perk=self.perk,
        )


@dataclass(frozen=True)
class TeamSnapshot:
    turn: int
    label: str
    pets: Tuple[PetSnapshot, ...]

    @classmethod
    def capture(cls, turn: int, team: Sequence[Pet], label: str) -> TeamSnapshot:
        return cls(turn=turn, label=label, pets=tuple(PetSnapshot.from_pet(pet) for pet in team))


class SnapshotLeague:
    """Samples immutable team snapshots for the requested game turn.

    An exact-turn bucket is preferred. If it is empty, the nearest earlier
    bucket is used; when the league has no usable snapshot, the fallback
    provider supplies the opponent. Returned pets are always fresh objects.
    """

    def __init__(
        self,
        fallback: OpponentProvider = default_opponent,
        catalog_id: Optional[str] = None,
    ) -> None:
        self.fallback = fallback
        self.catalog_id = catalog_id
        self._by_turn: Dict[int, List[TeamSnapshot]] = {}

    def add(self, turn: int, team: Sequence[Pet], label: str = "snapshot") -> None:
        if turn < 1:
            raise ValueError("turn must be positive")
        snapshot = TeamSnapshot.capture(turn, team, label)
        self._by_turn.setdefault(turn, []).append(snapshot)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._by_turn.values())

    def __call__(
        self, turn: int, rng: random.Random, catalog: Catalog, config: GameConfig
    ) -> Sequence[Pet]:
        if self.catalog_id is not None and self.catalog_id != catalog.catalog_id:
            raise ValueError(
                f"Opponent league needs catalog {self.catalog_id}, got {catalog.catalog_id}"
            )
        eligible = [saved_turn for saved_turn in self._by_turn if saved_turn <= turn]
        if not eligible:
            return self.fallback(turn, rng, catalog, config)
        snapshot = rng.choice(self._by_turn[max(eligible)])
        return [pet.to_pet() for pet in snapshot.pets]

    def save(self, path: str | Path) -> None:
        snapshots = [
            {
                "turn": snapshot.turn,
                "label": snapshot.label,
                "pets": [asdict(pet) for pet in snapshot.pets],
            }
            for turn in sorted(self._by_turn)
            for snapshot in self._by_turn[turn]
        ]
        payload = {
            "schema_version": 1,
            "catalog_id": self.catalog_id,
            "snapshots": snapshots,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path, fallback: OpponentProvider = default_opponent
    ) -> SnapshotLeague:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError(f"Unsupported league schema: {data['schema_version']}")
        league = cls(fallback=fallback, catalog_id=data.get("catalog_id"))
        for item in data["snapshots"]:
            pets = [PetSnapshot(**pet).to_pet() for pet in item["pets"]]
            league.add(int(item["turn"]), pets, str(item["label"]))
        return league


def build_spend_gold_league(episodes: int, seed: int = 0) -> SnapshotLeague:
    """Capture round-indexed teams produced by the readable heuristic policy."""

    if episodes < 1:
        raise ValueError("episodes must be positive")

    from .actions import ActionKind
    from .baselines import SpendGoldPolicy
    from .engine import AutoBattler

    engine = AutoBattler()
    policy = SpendGoldPolicy()
    league = SnapshotLeague(catalog_id=engine.catalog.catalog_id)
    for episode in range(episodes):
        episode_seed = seed + episode
        policy_rng = random.Random(episode_seed + 1_000_003)
        engine.reset(seed=episode_seed)
        while not (engine.state.terminated or engine.state.truncated):
            action = policy.choose(engine, policy_rng)
            if action.kind is ActionKind.END_TURN:
                league.add(
                    engine.state.turn,
                    engine.state.team,
                    label=f"spend-gold-seed-{episode_seed}",
                )
            engine.step(action)
    return league


class OpponentMixture:
    """Chooses among scripted or snapshot providers using fixed weights."""

    def __init__(self, providers: Sequence[Tuple[float, OpponentProvider]]) -> None:
        if not providers or any(weight <= 0 for weight, _ in providers):
            raise ValueError("providers must contain positive weights")
        self.providers = tuple(providers)

    def __call__(
        self, turn: int, rng: random.Random, catalog: Catalog, config: GameConfig
    ) -> Sequence[Pet]:
        weights = [weight for weight, _ in self.providers]
        provider = rng.choices([provider for _, provider in self.providers], weights=weights, k=1)[
            0
        ]
        return provider(turn, rng, catalog, config)
