"""Versioned, validated game-content catalogs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Dict, Optional, Tuple

SUPPORTED_TRIGGERS = {
    "buy",
    "sell",
    "level_up",
    "start_battle",
    "faint",
    "friend_summoned",
    "start_turn",
    "end_turn",
    "hurt",
    "after_attack",
    "friend_ahead_attacks",
    "friend_ahead_faints",
    "friendly_ate_food",
    "knock_out",
    "summoned",
    "friend_faints",
}
SUPPORTED_EFFECTS = {
    "buff",
    "set_perk",
    "buff_random_friend",
    "buff_subject",
    "damage_random_enemy",
    "gain_gold",
    "stock_food",
    "summon",
    "buff_shop_pets",
    "buff_self",
    "buff_position",
    "gain_health_percent",
    "share_attack",
    "damage_lowest_enemy",
    "damage_behind",
    "damage_all",
    "damage_adjacent",
    "summon_enemy",
    "summon_tier",
    "buff_eater",
    "gain_melon",
    "faint_pet",
    "buff_level_friends",
    "buff_if_level_friend",
    "discount_shop_food",
    "perk_position",
    "reduce_highest_health",
    "copy_ahead_ability",
    "swallow_ahead",
    "buff_random_team",
    "buff_all_pets",
    "buff_front_pet",
    "replace_milk",
    "damage_last_enemy",
    "rhino_snipe",
    "rooster_chicks",
    "gain_peanut",
    "buff_future_shop",
}


@dataclass(frozen=True)
class AbilitySpec:
    trigger: str
    effect: str
    params: Mapping[str, Any]

    def at_level(self, key: str, level: int, default: int = 0) -> int:
        if level not in {1, 2, 3}:
            raise ValueError("Ability level must be 1, 2, or 3")
        values = self.params.get(f"{key}_by_level")
        if values is None:
            return int(self.params.get(key, default))
        return int(values[level - 1])


@dataclass(frozen=True)
class PetSpec:
    id: str
    name: str
    tier: int
    attack: int
    health: int
    token: bool
    abilities: Tuple[AbilitySpec, ...]


@dataclass(frozen=True)
class FoodSpec:
    id: str
    name: str
    tier: int
    cost: int
    effect: str
    params: Mapping[str, Any]
    token: bool = False
    curriculum_only: bool = False


@dataclass(frozen=True)
class Catalog:
    catalog_id: str
    game_version: str
    checked_on: str
    pets: Mapping[str, PetSpec]
    foods: Mapping[str, FoodSpec]
    rules_version: int = 1
    development_only: bool = False
    implemented_shop_tiers: Tuple[int, ...] = ()
    battle_attack_limit: Optional[int] = None

    @property
    def rollable_pet_ids(self) -> Tuple[str, ...]:
        return tuple(spec.id for spec in self.pets.values() if not spec.token)

    @property
    def rollable_food_ids(self) -> Tuple[str, ...]:
        return tuple(spec.id for spec in self.foods.values() if not spec.token)

    def pet_ids_through_tier(self, tier: int) -> Tuple[str, ...]:
        return tuple(spec.id for spec in self.pets.values() if not spec.token and spec.tier <= tier)

    def food_ids_through_tier(self, tier: int) -> Tuple[str, ...]:
        return tuple(
            spec.id for spec in self.foods.values() if not spec.token and spec.tier <= tier
        )


def _unique_ids(records: Iterable[Mapping[str, Any]], record_type: str) -> None:
    seen = set()
    for record in records:
        record_id = record.get("id")
        if not record_id or record_id in seen:
            raise ValueError(f"{record_type} IDs must be non-empty and unique: {record_id!r}")
        seen.add(record_id)


def _ability(raw: Mapping[str, Any]) -> AbilitySpec:
    trigger = str(raw["trigger"])
    effect = str(raw["effect"])
    if trigger not in SUPPORTED_TRIGGERS:
        raise ValueError(f"Unsupported trigger: {trigger}")
    if effect not in SUPPORTED_EFFECTS:
        raise ValueError(f"Unsupported effect: {effect}")
    params = {key: value for key, value in raw.items() if key not in {"trigger", "effect"}}
    if "rounding" in params:
        if effect not in {
            "gain_health_percent",
            "share_attack",
            "damage_adjacent",
            "reduce_highest_health",
        }:
            raise ValueError("Percentage rounding requires a percentage effect")
        if params["rounding"] not in {"floor", "ceil"}:
            raise ValueError("Unsupported percentage rounding")
    for key, value in params.items():
        if key.endswith("_by_level") and len(value) != 3:
            raise ValueError(f"{key} must contain exactly three level values")
    return AbilitySpec(trigger=trigger, effect=effect, params=params)


def catalog_from_dict(raw: Mapping[str, Any]) -> Catalog:
    rules_version = int(raw.get("rules_version", 1))
    if rules_version not in {1, 2, 3, 4, 5, 6}:
        raise ValueError(f"Unsupported rules version: {rules_version}")
    battle_attack_limit = raw.get("battle_attack_limit")
    if battle_attack_limit is not None:
        if type(battle_attack_limit) is not int or not 1 <= battle_attack_limit <= 200:
            raise ValueError("battle_attack_limit must be an integer from 1 through 200")
        if rules_version < 4:
            raise ValueError("Battle draw limits are opt-in expanded rules only")
    raw_pets = list(raw["pets"])
    raw_foods = list(raw["foods"])
    _unique_ids(raw_pets, "Pet")
    _unique_ids(raw_foods, "Food")

    pets: Dict[str, PetSpec] = {}
    for entry in raw_pets:
        spec = PetSpec(
            id=str(entry["id"]),
            name=str(entry["name"]),
            tier=int(entry["tier"]),
            attack=int(entry["attack"]),
            health=int(entry["health"]),
            token=bool(entry.get("token", False)),
            abilities=tuple(_ability(item) for item in entry.get("abilities", [])),
        )
        if spec.attack < 0 or spec.health <= 0 or not 0 <= spec.tier <= 6:
            raise ValueError(f"Invalid pet stats for {spec.id}")
        pets[spec.id] = spec

    foods: Dict[str, FoodSpec] = {}
    for entry in raw_foods:
        effect = str(entry["effect"])
        if effect not in SUPPORTED_EFFECTS:
            raise ValueError(f"Unsupported food effect: {effect}")
        spec = FoodSpec(
            id=str(entry["id"]),
            name=str(entry["name"]),
            tier=int(entry["tier"]),
            cost=int(entry["cost"]),
            effect=effect,
            params={
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "id",
                    "name",
                    "tier",
                    "cost",
                    "effect",
                    "token",
                    "curriculum_only",
                }
            },
            token=bool(entry.get("token", False)),
            curriculum_only=bool(entry.get("curriculum_only", False)),
        )
        foods[spec.id] = spec

    for pet in pets.values():
        for ability in pet.abilities:
            pet_id = ability.params.get("summon_id")
            food_id = ability.params.get("food_id")
            if pet_id is not None and pet_id not in pets:
                raise ValueError(f"{pet.id} references missing summoned pet {pet_id}")
            if food_id is not None and food_id not in foods:
                raise ValueError(f"{pet.id} references missing food {food_id}")
            for food_id in ability.params.get("food_id_by_level", []):
                if food_id not in foods:
                    raise ValueError(f"{pet.id} references missing food {food_id}")

    return Catalog(
        catalog_id=str(raw["catalog_id"]),
        game_version=str(raw["game_version"]),
        checked_on=str(raw["checked_on"]),
        pets=pets,
        foods=foods,
        rules_version=rules_version,
        development_only=bool(raw.get("development_only", False)),
        implemented_shop_tiers=tuple(int(tier) for tier in raw.get("implemented_shop_tiers", [])),
        battle_attack_limit=battle_attack_limit,
    )


def load_catalog(name: str = "turtle_v0_46_tier1_rules_v2.json") -> Catalog:
    catalog_file = resources.files("sap_rl_lab").joinpath("catalogs", name)
    with catalog_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "extends" in raw:
        # v6 is an append-only content extension, not an edit of the frozen v5 file.
        if raw["extends"] != "turtle_v0_46_midgame_v5.json" or raw["rules_version"] != 6:
            raise ValueError("Unsupported catalog inheritance")
        base_file = resources.files("sap_rl_lab").joinpath("catalogs", raw["extends"])
        with base_file.open("r", encoding="utf-8") as handle:
            base = json.load(handle)
        raw = {
            **base,
            **raw,
            "pets": base["pets"] + raw["pets"],
            "foods": base["foods"] + raw["foods"],
        }
    return catalog_from_dict(raw)


def catalog_digest(catalog: Catalog) -> str:
    """Fingerprint content AND insertion order, which defines observation IDs."""
    payload = asdict(catalog)
    # Unconfigured historical catalogs retain their exact old fingerprints.
    if payload["battle_attack_limit"] is None:
        del payload["battle_attack_limit"]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_catalog_by_id(catalog_id: str) -> Catalog:
    """Resolve recorded rules explicitly; never replay old data under new rules."""
    names = {
        "turtle-v0.46-tier1": "turtle_v0_46_tier1.json",
        "turtle-v0.46-tier1-rules-v2": "turtle_v0_46_tier1_rules_v2.json",
        "turtle-v0.46-tier12-dev-v3": "turtle_v0_46_tier12_dev_v3.json",
        "turtle-v0.46-tier12-curriculum-v4": "turtle_v0_46_tier12_curriculum_v4.json",
        "turtle-v0.46-midgame-v5": "turtle_v0_46_midgame_v5.json",
        "turtle-v0.46-tier4-v6": "turtle_v0_46_tier4_v6.json",
    }
    if catalog_id not in names:
        raise ValueError(f"Unknown recorded catalog: {catalog_id}")
    return load_catalog(names[catalog_id])
