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
    ability_uses: int = 0
    copied_ability: Optional[str] = None
    sell_bonus: int = 0

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
    pet: Optional[Pet] = None
    choice_group: Optional[int] = None


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
    shop_action_limit_mode: str = "truncate"
    shop_pet_stats: bool = False
    shop_rules: str = "legacy"
    max_shop_tier: int = 6

    def __post_init__(self) -> None:
        if self.shop_rules not in {
            "legacy",
            "turtle_v046_development",
            "turtle_tier12_curriculum",
            "turtle_midgame",
            "turtle_tier4",
            "turtle_tier5",
            "turtle_full",
        }:
            raise ValueError("Unknown shop_rules")
        if not 1 <= self.max_shop_tier <= 6:
            raise ValueError("max_shop_tier must be 1 through 6")
        if self.shop_rules == "turtle_tier12_curriculum" and self.max_shop_tier != 2:
            raise ValueError("This curriculum requires a Tier-2 normal-shop cap")
        if self.shop_rules == "turtle_midgame" and self.max_shop_tier != 3:
            raise ValueError("Midgame requires a Tier-3 normal-shop cap and Tier-4 rewards")
        if self.shop_rules == "turtle_tier4" and self.max_shop_tier != 4:
            raise ValueError("Tier4 curriculum requires Tier-4 shops and Tier-5 rewards")
        if self.shop_rules == "turtle_tier5" and self.max_shop_tier != 5:
            raise ValueError("Tier5 curriculum requires Tier-5 shops and Tier-6 rewards")
        if self.shop_rules == "turtle_full" and self.max_shop_tier != 6:
            raise ValueError("Full Turtle Pack requires Tier-6 shops")
        if self.shop_rules == "legacy":
            if self.pet_shop_slots + self.food_shop_slots != self.max_shop_size:
                raise ValueError("Pet and food shop slots must add up to max_shop_size")
        elif (self.max_shop_size, self.pet_shop_slots, self.food_shop_slots) != (9, 5, 2):
            raise ValueError("Development shop requires 5 pet + 2 food + 2 extra slots")
        elif not self.shop_pet_stats:
            raise ValueError("Development shop requires observable shop pet stats")
        if min(self.max_team_size, self.max_shop_size, self.starting_gold) <= 0:
            raise ValueError("Game sizes and starting gold must be positive")
        if self.max_actions_per_turn < 1 or self.max_turns < 1:
            raise ValueError("Action and turn limits must be positive")
        if self.shop_action_limit_mode not in {"truncate", "force_battle"}:
            raise ValueError("shop_action_limit_mode must be truncate or force_battle")

    @classmethod
    def turtle_development(cls, **overrides: Any) -> GameConfig:
        """Opt-in development contract; never changes the eight-pet defaults."""
        values = dict(
            max_shop_size=9,
            pet_shop_slots=5,
            food_shop_slots=2,
            shop_pet_stats=True,
            shop_rules="turtle_v046_development",
            shop_action_limit_mode="force_battle",
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def turtle_curriculum(cls, **overrides: Any) -> GameConfig:
        values = dict(shop_rules="turtle_tier12_curriculum", max_shop_tier=2)
        values.update(overrides)
        return cls.turtle_development(**values)

    @classmethod
    def turtle_midgame(cls, **overrides: Any) -> GameConfig:
        values = dict(shop_rules="turtle_midgame", max_shop_tier=3)
        values.update(overrides)
        return cls.turtle_development(**values)

    @classmethod
    def turtle_tier4(cls, **overrides: Any) -> GameConfig:
        values = dict(shop_rules="turtle_tier4", max_shop_tier=4)
        values.update(overrides)
        return cls.turtle_development(**values)

    @classmethod
    def turtle_full(cls, tier=6, **overrides: Any) -> GameConfig:
        if tier not in (5, 6):
            raise ValueError("Full-pack curriculum tier must be 5 or 6")
        values = dict(shop_rules="turtle_tier5" if tier == 5 else "turtle_full", max_shop_tier=tier)
        values.update(overrides)
        return cls.turtle_development(**values)


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
    shop_attack_bonus: int = 0
    shop_health_bonus: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for field_name in ("shop_attack_bonus", "shop_health_bonus"):
            if result[field_name] == 0:
                result.pop(field_name)
        for pet in result["team"]:
            if pet["ability_uses"] == 0:
                pet.pop("ability_uses")
            if pet["copied_ability"] is None:
                pet.pop("copied_ability")
            if pet["sell_bonus"] == 0:
                pet.pop("sell_bonus")
        # Keep legacy eight-pet state/replay payloads byte-for-byte compatible.
        for item in result["shop"]:
            if item is not None:
                if item["pet"] is not None and item["pet"]["ability_uses"] == 0:
                    item["pet"].pop("ability_uses")
                if item["pet"] is not None:
                    if item["pet"]["copied_ability"] is None:
                        item["pet"].pop("copied_ability")
                    if item["pet"]["sell_bonus"] == 0:
                        item["pet"].pop("sell_bonus")
                for optional in ("pet", "choice_group"):
                    if item[optional] is None:
                        item.pop(optional)
        if self.previous_outcome is not None:
            result["previous_outcome"] = int(self.previous_outcome)
        return result
