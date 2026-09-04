"""Small serializable domain objects; no RL library dependency lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, List, Optional


class BattleOutcome(IntEnum):
    LOSS = -1
    DRAW = 0
    WIN = 1


@dataclass
class Pet:
    spec_id: str
    attack: int
    health: int
    experience: int = 1
    perk: Optional[str] = None
    temporary_attack: int = 0
    temporary_health: int = 0

    @property
    def level(self) -> int:
        if self.experience >= 6:
            return 3
        if self.experience >= 3:
            return 2
        return 1

    def clone(self) -> Pet:
        return replace(self)

    def battle_copy(self) -> Pet:
        copy = self.clone()
        copy.attack += copy.temporary_attack
        copy.health += copy.temporary_health
        copy.temporary_attack = 0
        copy.temporary_health = 0
        return copy

    def clear_temporary_stats(self) -> None:
        self.temporary_attack = 0
        self.temporary_health = 0


@dataclass
class ShopItem:
    kind: str
    item_id: str
    cost: int
    frozen: bool = False


@dataclass(frozen=True)
class GameConfig:
    max_team_size: int = 5
    max_shop_size: int = 5
    starting_gold: int = 10
    starting_lives: int = 5
    target_wins: int = 10
    max_turns: int = 30
    max_actions_per_turn: int = 30
    pet_cost: int = 3
    pet_shop_slots: int = 4
    food_shop_slots: int = 1

    def __post_init__(self) -> None:
        if self.pet_shop_slots + self.food_shop_slots != self.max_shop_size:
            raise ValueError("Pet and food shop slots must add up to max_shop_size")
        if min(self.max_team_size, self.max_shop_size, self.starting_gold) <= 0:
            raise ValueError("Game sizes and starting gold must be positive")


@dataclass
class GameState:
    turn: int = 1
    gold: int = 10
    lives: int = 5
    wins: int = 0
    team: List[Pet] = field(default_factory=list)
    shop: List[Optional[ShopItem]] = field(default_factory=list)
    previous_outcome: Optional[BattleOutcome] = None
    actions_this_turn: int = 0
    terminated: bool = False
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.previous_outcome is not None:
            result["previous_outcome"] = int(self.previous_outcome)
        return result
