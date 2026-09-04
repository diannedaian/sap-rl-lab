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

    def __init__(self, fallback: OpponentProvider = default_opponent) -> None:
        self.fallback = fallback
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
        Path(path).write_text(json.dumps({"snapshots": snapshots}, indent=2) + "\n")

    @classmethod
    def load(
        cls, path: str | Path, fallback: OpponentProvider = default_opponent
    ) -> SnapshotLeague:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        league = cls(fallback=fallback)
        for item in data["snapshots"]:
            pets = [PetSnapshot(**pet).to_pet() for pet in item["pets"]]
            league.add(int(item["turn"]), pets, str(item["label"]))
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
