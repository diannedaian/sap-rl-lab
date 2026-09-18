"""Pure-Python shop and battle engine with explicit, atomic transitions."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .actions import Action, ActionCodec, ActionKind
from .catalog import AbilitySpec, Catalog, load_catalog
from .domain import BattleOutcome, GameConfig, GameState, Pet, ShopItem
from .shop import ContentNotReady, merged_pet, shop_slots, stock_items


class InvalidAction(ValueError):
    """Raised before state mutation when an action is not legal."""


@dataclass(frozen=True)
class Transition:
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BattleResult:
    outcome: BattleOutcome
    trace: List[str]
    attacks: int
    attack_limit_reached: bool = False


OpponentProvider = Callable[[int, random.Random, Catalog, GameConfig], Sequence[Pet]]


def make_pet(catalog: Catalog, pet_id: str) -> Pet:
    spec = catalog.pets[pet_id]
    return Pet(spec_id=spec.id, attack=spec.attack, health=spec.health)


def default_opponent(
    turn: int, rng: random.Random, catalog: Catalog, config: GameConfig
) -> Sequence[Pet]:
    """A deliberately transparent baseline, not a claim about arena matchmaking."""

    available = catalog.pet_ids_through_tier(min(config.max_shop_tier, 1 + (turn - 1) // 2))
    team_size = min(config.max_team_size, 2 + turn // 2)
    stat_bonus = max(0, (turn - 1) // 2)
    team = []
    for _ in range(team_size):
        pet = make_pet(catalog, rng.choice(available))
        pet.attack += rng.randint(0, stat_bonus)
        pet.health += rng.randint(0, stat_bonus)
        team.append(pet)
    return team


class AutoBattler:
    """Owns one episode and exposes legal actions independent of any RL library."""

    def __init__(
        self,
        catalog: Optional[Catalog] = None,
        config: Optional[GameConfig] = None,
        opponent_provider: Optional[OpponentProvider] = None,
    ) -> None:
        self.catalog = catalog or load_catalog()
        self.fullpack_rules = self.catalog.rules_version in {7, 8}
        self.tier4_rules = self.catalog.rules_version in {6, 7, 8}
        self.midgame_rules = self.catalog.rules_version in {5, 6, 7, 8}
        self.expanded_rules = self.catalog.rules_version in {4, 5, 6, 7, 8}
        self.config = config or (
            GameConfig.turtle_full(tier=self.catalog.rules_version - 2)
            if self.fullpack_rules
            else GameConfig.turtle_tier4()
            if self.tier4_rules
            else GameConfig.turtle_midgame()
            if self.midgame_rules
            else GameConfig.turtle_curriculum()
            if self.expanded_rules
            else GameConfig.turtle_development()
            if self.catalog.rules_version == 3
            else GameConfig()
        )
        self.modern_shop = self.config.shop_rules != "legacy"
        expected_shop = {
            3: "turtle_v046_development",
            4: "turtle_tier12_curriculum",
            5: "turtle_midgame",
            6: "turtle_tier4",
            7: "turtle_tier5",
            8: "turtle_full",
        }.get(self.catalog.rules_version, "legacy")
        if self.config.shop_rules != expected_shop:
            raise ValueError("Catalog rules and shop contract must be selected together")
        needs_shop_stats = any(
            ability.effect == "buff_shop_pets"
            for pet in self.catalog.pets.values()
            for ability in pet.abilities
        )
        if needs_shop_stats and not self.config.shop_pet_stats:
            raise ValueError("Shop-buff abilities require observable persistent shop pet stats")
        self.codec = ActionCodec(
            self.config.max_team_size, self.config.max_shop_size, team_merges=self.modern_shop
        )
        self.opponent_provider = opponent_provider or default_opponent
        self.rng = random.Random()
        self.seed: Optional[int] = None
        self.state = GameState()
        self.last_battle: Optional[BattleResult] = None

    def reset(self, seed: Optional[int] = None) -> GameState:
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = GameState(
            turn=1,
            gold=self.config.starting_gold,
            lives=self.config.starting_lives,
        )
        self.last_battle = None
        self._roll_shop(free=True)
        return self.state

    @property
    def unlocked_tier(self) -> int:
        return min(self.config.max_shop_tier, 1 + (self.state.turn - 1) // 2)

    def legal_actions(self) -> Tuple[Action, ...]:
        if self.state.terminated or self.state.truncated:
            return ()
        legal: List[Action] = [Action(ActionKind.END_TURN)]
        if self.state.gold >= 1:
            legal.append(Action(ActionKind.ROLL))

        for shop_index, item in enumerate(self.state.shop):
            if item is None:
                continue
            legal.append(Action(ActionKind.FREEZE, source=shop_index))
            if item.kind == "pet" and self.state.gold >= item.cost:
                if len(self.state.team) < self.config.max_team_size:
                    legal.append(Action(ActionKind.BUY_PET, source=shop_index))
                for team_index, pet in enumerate(self.state.team):
                    if pet.spec_id == item.item_id and pet.experience < 6:
                        legal.append(Action(ActionKind.MERGE, source=shop_index, target=team_index))
            if item.kind == "food" and self.state.gold >= item.cost:
                if (
                    self.tier4_rules
                    and self.catalog.foods[item.item_id].effect == "buff_future_shop"
                ):
                    legal.append(Action(ActionKind.BUY_FOOD, source=shop_index, target=0))
                    continue  # Canned Food targets the shop, including with an empty team.
                for team_index in range(len(self.state.team)):
                    if (
                        self.midgame_rules
                        and team_index > 0
                        and self.catalog.foods[item.item_id].effect == "buff_random_team"
                    ):
                        continue  # Random-recipient food has one canonical buy action.
                    legal.append(Action(ActionKind.BUY_FOOD, source=shop_index, target=team_index))

        for team_index in range(len(self.state.team)):
            legal.append(Action(ActionKind.SELL, source=team_index))
        for team_index in range(max(0, len(self.state.team) - 1)):
            legal.append(Action(ActionKind.SWAP, source=team_index, target=team_index + 1))
        if self.modern_shop:
            for source, incoming in enumerate(self.state.team):
                for target, pet in enumerate(self.state.team):
                    if source != target and incoming.spec_id == pet.spec_id and pet.experience < 6:
                        legal.append(Action(ActionKind.MERGE_TEAM, source, target))
        return tuple(legal)

    def legal_action_ids(self) -> Tuple[int, ...]:
        return tuple(self.codec.encode(action) for action in self.legal_actions())

    def action_mask(self) -> Tuple[bool, ...]:
        legal = set(self.legal_action_ids())
        return tuple(action_id in legal for action_id in range(self.codec.size))

    def step_id(self, action_id: int) -> Transition:
        return self.step(self.codec.decode(action_id))

    def step(self, action: Action) -> Transition:
        if not self.modern_shop:
            return self._step(action)
        # A partial development catalog may discover a missing dependency during
        # the forced battle / next-shop transition. Roll back the WHOLE action,
        # including RNG, instead of leaving a paid-for, half-applied move.
        before = deepcopy(self.state), self.rng.getstate(), self.last_battle
        try:
            return self._step(action)
        except ContentNotReady:
            self.state, rng_state, self.last_battle = before
            self.rng.setstate(rng_state)
            raise

    def _step(self, action: Action) -> Transition:
        legal = self.legal_actions()
        if action not in legal:
            raise InvalidAction(
                f"Illegal action {action}; legal actions are "
                + ", ".join(self.codec.describe(self.codec.encode(item)) for item in legal)
            )

        if action.kind is ActionKind.END_TURN:
            return self._end_turn()
        if action.kind is ActionKind.ROLL:
            self._roll_shop(free=False)
        elif action.kind is ActionKind.BUY_PET:
            self._buy_pet(action.source)
        elif action.kind is ActionKind.BUY_FOOD:
            self._buy_food(action.source, action.target)
        elif action.kind is ActionKind.MERGE:
            self._merge(action.source, action.target)
        elif action.kind is ActionKind.MERGE_TEAM:
            self._merge_team(action.source, action.target)
        elif action.kind is ActionKind.SELL:
            self._sell(action.source)
        elif action.kind is ActionKind.SWAP:
            self.state.team[action.source], self.state.team[action.target] = (
                self.state.team[action.target],
                self.state.team[action.source],
            )
        elif action.kind is ActionKind.FREEZE:
            item = self.state.shop[action.source]
            assert item is not None
            item.frozen = not item.frozen
        else:  # pragma: no cover - the enum and codec make this unreachable
            raise AssertionError(f"Unhandled action kind: {action.kind}")

        self.state.actions_this_turn += 1
        if self.state.actions_this_turn >= self.config.max_actions_per_turn:
            if self.config.shop_action_limit_mode == "force_battle":
                # The selected action happens first, then the ordinary end-turn
                # path runs exactly once. This is an environment transition, not
                # a sampled END_TURN action or an absorbing shop-limit failure.
                budget_info = {
                    "forced_end_turn": True,
                    "forced_end_turn_reason": "shop_action_limit",
                    "shop_actions_before_battle": self.state.actions_this_turn,
                    "gold_before_battle": self.state.gold,
                    "empty_slots_before_battle": self.config.max_team_size - len(self.state.team),
                }
                transition = self._end_turn()
                return Transition(
                    transition.reward,
                    transition.terminated,
                    transition.truncated,
                    {**transition.info, **budget_info},
                )
            self.state.truncated = True
            return Transition(
                reward=-1.0,
                terminated=False,
                truncated=True,
                info={"reason": "shop_action_limit"},
            )
        return Transition(reward=0.0, terminated=False, truncated=False)

    def _buy_pet(self, shop_index: int) -> None:
        item = self.state.shop[shop_index]
        assert item is not None
        self._consume_shop_item(shop_index)
        self.state.gold -= item.cost
        pet = item.pet.clone() if item.pet is not None else make_pet(self.catalog, item.item_id)
        self.state.team.append(pet)
        self._run_shop_trigger(pet, "buy", subject=pet)
        self._notify_friend_summoned(pet, battle=False)

    def _buy_food(self, shop_index: int, team_index: int) -> None:
        item = self.state.shop[shop_index]
        assert item is not None
        self._consume_shop_item(shop_index)
        self.state.gold -= item.cost
        food = self.catalog.foods[item.item_id]
        multiplier = self._food_multiplier(food) if self.fullpack_rules else 1
        if self.tier4_rules and food.effect == "buff_future_shop":
            atk, hp = (
                int(food.params["attack"]) * multiplier,
                int(food.params["health"]) * multiplier,
            )
            self.state.shop_attack_bonus = min(50, self.state.shop_attack_bonus + atk)
            self.state.shop_health_bonus = min(50, self.state.shop_health_bonus + hp)
            runtime = self._shop_runtime()
            for offered in self.state.shop:
                if offered is not None and offered.kind == "pet":
                    runtime.buff(offered.pet, atk, hp)
            return  # No team pet ate this food; do not trigger Rabbit/Seal.
        target = self.state.team[team_index]
        if self.expanded_rules:
            runtime = self._shop_runtime()
            if food.effect == "buff_random_team" and self.midgame_rules:
                friends = runtime.alive(0)
                preferred = [
                    p
                    for p in friends
                    if p.attack + p.temporary_attack < 50 or p.health + p.temporary_health < 50
                ]
                count = min(int(food.params.get("count", 2)), len(friends))
                targets = self.rng.sample(preferred, min(count, len(preferred)))
                chosen = {id(p) for p in targets}
                if len(targets) < count:
                    targets += self.rng.sample(
                        [p for p in friends if id(p) not in chosen], count - len(targets)
                    )
                for recipient in targets:
                    runtime.buff(
                        recipient,
                        int(food.params.get("attack", 0)) * multiplier,
                        int(food.params.get("health", 0)) * multiplier,
                    )
                    runtime.notify_food_eaten(0, recipient)
                runtime.drain()
                return
            if food.effect == "buff":
                runtime.buff(
                    target,
                    int(food.params.get("attack", 0)) * multiplier,
                    int(food.params.get("health", 0)) * multiplier,
                    bool(food.params.get("temporary", False)),
                )
            elif food.effect == "set_perk":
                target.perk = str(food.params["perk"])
            elif food.effect == "faint_pet":
                runtime.faint(0, target)
            elif food.effect == "gain_experience" and self.fullpack_rules:
                self._feed_experience(target, int(food.params["experience"]))
            else:
                raise RuntimeError(f"Unsupported curriculum food: {food.effect}")
            runtime.notify_food_eaten(0, target)
            runtime.drain()
            return
        if food.effect == "buff":
            target.attack = min(50, target.attack + int(food.params.get("attack", 0)))
            target.health = min(50, target.health + int(food.params.get("health", 0)))
        elif food.effect == "set_perk":
            target.perk = str(food.params["perk"])
        else:
            raise RuntimeError(f"Food effect has no engine implementation: {food.effect}")

    def _merge(self, shop_index: int, team_index: int) -> None:
        item = self.state.shop[shop_index]
        assert item is not None
        if self.modern_shop:
            incoming = item.pet if item.pet is not None else make_pet(self.catalog, item.item_id)
            self._consume_shop_item(shop_index)
            self.state.gold -= item.cost
            self._modern_merge(incoming, team_index, bought=True)
            return
        self.state.shop[shop_index] = None
        self.state.gold -= item.cost
        target = self.state.team[team_index]
        previous_level = target.level
        incoming = item.pet if item.pet is not None else make_pet(self.catalog, item.item_id)
        target.attack = min(50, max(target.attack, incoming.attack) + 1)
        target.health = min(50, max(target.health, incoming.health) + 1)
        target.experience = min(6, target.experience + incoming.experience)
        self._run_shop_trigger(target, "buy", subject=target)
        if target.level > previous_level:
            # A level-up ability belongs to the departing level. Other triggers
            # (buy, sell, faint, summon, battle) retain the owner's current level.
            # v1 is intentionally preserved for exact historical replay.
            ability_level = previous_level if self.catalog.rules_version >= 2 else target.level
            self._run_shop_trigger(target, "level_up", subject=target, ability_level=ability_level)

    def _consume_shop_item(self, shop_index: int) -> None:
        item = self.state.shop[shop_index]
        assert item is not None
        self.state.shop[shop_index] = None
        if item.choice_group is not None:
            for index, other in enumerate(self.state.shop):
                if other is not None and other.choice_group == item.choice_group:
                    self.state.shop[index] = None

    def _merge_team(self, source: int, target: int) -> None:
        incoming = self.state.team.pop(source)
        # Index-based removal is intentional: equal-stat pets are distinct pets.
        self._modern_merge(incoming, target - int(source < target), bought=False)

    def _modern_merge(self, incoming: Pet, team_index: int, bought: bool) -> None:
        target = self.state.team[team_index]
        previous_level = max(target.level, incoming.level)
        both_level_two_or_more = min(target.level, incoming.level) >= 2
        refresh_triggers = self.expanded_rules and any(
            ability.params.get("merge_trigger_rule") == "target-reset-on-level-up"
            for ability in self.catalog.pets[target.spec_id].abilities
        )
        result = merged_pet(target, incoming, refresh_triggers=refresh_triggers)
        if self.midgame_rules:
            # Sell bonus follows the larger accumulated value, not a sum that
            # would duplicate value while merging. Target-client fixture pending.
            result.sell_bonus = max(target.sell_bonus, incoming.sell_bonus)
        if self.expanded_rules:
            result.temporary_attack = min(result.temporary_attack, max(0, 50 - result.attack))
            result.temporary_health = min(result.temporary_health, max(0, 50 - result.health))
        self.state.team[team_index] = result
        defer_cow_buy = (
            bought and self.tier4_rules and (result.copied_ability or result.spec_id) == "cow"
        )
        if bought and not defer_cow_buy:
            self._run_shop_trigger(result, "buy", subject=result)
        if result.level > previous_level:
            self._run_shop_trigger(result, "level_up", subject=result, ability_level=previous_level)
            if not both_level_two_or_more:
                self._stock_level_up_choices()
        if defer_cow_buy:
            self._run_shop_trigger(result, "buy", subject=result)

    def _fresh_shop_pet(self, pet_id):
        pet = make_pet(self.catalog, pet_id)
        if self.tier4_rules:
            pet.attack = min(50, pet.attack + self.state.shop_attack_bonus)
            pet.health = min(50, pet.health + self.state.shop_health_bonus)
        return pet

    def _require_shop_tier(self, tier: int) -> None:
        if self.modern_shop and tier not in self.catalog.implemented_shop_tiers:
            raise ContentNotReady(
                f"{self.catalog.catalog_id}: Tier {tier} pool is not implemented; "
                "no lower-tier substitution is allowed. Development catalog is not trainable."
            )

    def _stock_level_up_choices(self) -> None:
        tier = min(6, self.unlocked_tier + 1)
        self._require_shop_tier(tier)
        ids = tuple(
            pet.id for pet in self.catalog.pets.values() if not pet.token and pet.tier == tier
        )
        if not ids:
            raise ContentNotReady(f"Missing exact Tier {tier} level-up reward pool")
        group = 1 + max(
            (item.choice_group or 0 for item in self.state.shop if item is not None), default=0
        )
        choices = []
        for _ in range(2):
            pet_id = self.rng.choice(ids)
            choices.append(
                ShopItem(
                    "pet",
                    pet_id,
                    self.config.pet_cost,
                    pet=self._fresh_shop_pet(pet_id),
                    choice_group=group,
                )
            )
        stock_items(self.state.shop, choices, shop_slots(self.state.turn)[2], "pet")

    def _sell(self, team_index: int) -> None:
        pet = self.state.team.pop(team_index)
        self.state.gold += pet.level + (pet.sell_bonus if self.midgame_rules else 0)
        self._run_shop_trigger(pet, "sell", subject=pet)

    def _run_shop_trigger(
        self, owner: Pet, trigger: str, subject: Pet, ability_level: Optional[int] = None
    ) -> None:
        if self.expanded_rules:
            runtime = self._shop_runtime()
            runtime.emit(0, owner, trigger, subject, ability_level)
            if self.fullpack_rules and trigger == "buy":
                for friend in runtime.alive(0):
                    if friend is not owner:
                        runtime.emit(0, friend, "friend_bought", subject)
            runtime.drain()
            return
        spec = self.catalog.pets[owner.spec_id]
        level = owner.level if ability_level is None else ability_level
        for ability in spec.abilities:
            if ability.trigger != trigger:
                continue
            if ability.effect == "gain_gold":
                self.state.gold += ability.at_level("gold", level)
            elif ability.effect == "stock_food":
                count = ability.at_level("count", level, 1)
                food_id = str(ability.params["food_id"])
                if self.modern_shop:
                    food = self.catalog.foods[food_id]
                    stock_items(
                        self.state.shop,
                        [ShopItem("food", food.id, food.cost) for _ in range(count)],
                        shop_slots(self.state.turn)[2],
                        "food",
                    )
                else:
                    for _ in range(count):
                        self._stock_food(food_id)
            elif ability.effect == "buff_random_friend":
                friends = [pet for pet in self.state.team if pet is not subject]
                count = min(len(friends), ability.at_level("count", level, 1))
                for friend in self.rng.sample(friends, count):
                    friend.attack = min(50, friend.attack + ability.at_level("attack", level))
                    friend.health = min(50, friend.health + ability.at_level("health", level))
            elif ability.effect == "buff_shop_pets":
                for item in self.state.shop:
                    if item is not None and item.kind == "pet":
                        if item.pet is None:
                            item.pet = make_pet(self.catalog, item.item_id)
                        item.pet.attack = min(
                            50, item.pet.attack + ability.at_level("attack", level)
                        )
                        item.pet.health = min(
                            50, item.pet.health + ability.at_level("health", level)
                        )
            else:
                raise RuntimeError(
                    f"Effect {ability.effect} is not valid for shop trigger {trigger}"
                )

    def _notify_friend_summoned(self, subject: Pet, battle: bool) -> None:
        if self.expanded_rules:
            runtime = self._shop_runtime()
            runtime.notify_summon(0, subject)
            runtime.drain()
            return
        for owner in self.state.team:
            if owner is subject:
                continue
            spec = self.catalog.pets[owner.spec_id]
            for ability in spec.abilities:
                if ability.trigger != "friend_summoned":
                    continue
                attack = ability.at_level("attack", owner.level)
                health = ability.at_level("health", owner.level)
                if battle:
                    subject.attack += attack
                    subject.health += health
                else:
                    subject.temporary_attack += attack
                    subject.temporary_health += health

    def _stock_food(self, food_id: str) -> None:
        food = self.catalog.foods[food_id]
        stocked = ShopItem("food", food.id, food.cost)
        if self.modern_shop:
            stock_items(self.state.shop, [stocked], shop_slots(self.state.turn)[2], "food")
            return
        for index, item in enumerate(self.state.shop):
            if item is None:
                self.state.shop[index] = stocked
                return
        for index in range(len(self.state.shop) - 1, -1, -1):
            item = self.state.shop[index]
            if item is not None and not item.frozen:
                self.state.shop[index] = stocked
                return

    def _roll_shop(self, free: bool) -> None:
        if self.modern_shop:
            self._roll_development_shop(free)
            return
        if not free:
            self.state.gold -= 1
        old_shop = list(self.state.shop)
        new_shop: List[Optional[ShopItem]] = []
        pet_ids = self.catalog.pet_ids_through_tier(self.unlocked_tier)
        food_ids = self.catalog.food_ids_through_tier(self.unlocked_tier)
        for index in range(self.config.max_shop_size):
            old_item = old_shop[index] if index < len(old_shop) else None
            if old_item is not None and old_item.frozen:
                new_shop.append(old_item)
                continue
            if index < self.config.pet_shop_slots or not food_ids:
                pet_id = self.rng.choice(pet_ids)
                new_shop.append(
                    ShopItem(
                        "pet",
                        pet_id,
                        self.config.pet_cost,
                        pet=make_pet(self.catalog, pet_id) if self.config.shop_pet_stats else None,
                    )
                )
            else:
                food_id = self.rng.choice(food_ids)
                new_shop.append(ShopItem("food", food_id, self.catalog.foods[food_id].cost))
        self.state.shop = new_shop

    def _roll_development_shop(self, free: bool) -> None:
        # Check every unlocked pool: adding one Tier-2 pet is not a complete
        # Tier-2 roll distribution. The catalog declares implemented pools.
        for tier in range(1, self.unlocked_tier + 1):
            self._require_shop_tier(tier)
        pet_ids = self.catalog.pet_ids_through_tier(self.unlocked_tier)
        food_ids = self.catalog.food_ids_through_tier(self.unlocked_tier)
        if not pet_ids or not food_ids:
            raise ContentNotReady("Development shop requires pet and food pools")
        pet_slots, food_slots, active = shop_slots(self.state.turn)
        frozen_pets = [x for x in self.state.shop if x and x.frozen and x.kind == "pet"]
        frozen_foods = [x for x in self.state.shop if x and x.frozen and x.kind == "food"]
        pets = []
        foods = []
        remaining = active - len(frozen_pets) - len(frozen_foods)
        for _ in range(min(max(0, pet_slots - len(frozen_pets)), remaining)):
            pet_id = self.rng.choice(pet_ids)
            pets.append(
                ShopItem("pet", pet_id, self.config.pet_cost, pet=self._fresh_shop_pet(pet_id))
            )
        remaining -= len(pets)
        for _ in range(min(max(0, food_slots - len(frozen_foods)), remaining)):
            food_id = self.rng.choice(food_ids)
            foods.append(ShopItem("food", food_id, self.catalog.foods[food_id].cost))
        pets.sort(key=lambda item: self.catalog.pets[item.item_id].tier)
        foods.sort(key=lambda item: self.catalog.foods[item.item_id].tier)
        new_shop: List[Optional[ShopItem]] = [None] * self.config.max_shop_size
        for index, item in enumerate(frozen_pets + pets):
            new_shop[index] = item
        for index, item in enumerate(frozen_foods + foods):
            new_shop[active - 1 - index] = item
        self.state.shop = new_shop
        if not free:
            self.state.gold -= 1

    def _end_turn(self) -> Transition:
        if self.expanded_rules:
            self._shop_runtime().phase("end_turn")
        opponent = list(
            self.opponent_provider(self.state.turn, self.rng, self.catalog, self.config)
        )
        self.last_battle = resolve_battle(
            self.state.team,
            opponent,
            self.catalog,
            self.rng,
            max_team_size=self.config.max_team_size,
        )
        outcome = self.last_battle.outcome
        self.state.previous_outcome = outcome
        if outcome is BattleOutcome.WIN:
            self.state.wins += 1
        elif outcome is BattleOutcome.LOSS:
            self.state.lives -= 1

        for pet in self.state.team:
            pet.clear_temporary_stats()

        self.state.terminated = self.state.wins >= self.config.target_wins or self.state.lives <= 0
        info: Dict[str, Any] = {
            "battle_outcome": outcome.name.lower(),
            "opponent": [pet.spec_id for pet in opponent],
            "battle_trace": list(self.last_battle.trace),
        }
        if self.expanded_rules and self.catalog.battle_attack_limit is not None:
            info["battle_attacks"] = self.last_battle.attacks
            info["battle_attack_limit_reached"] = self.last_battle.attack_limit_reached
        if self.state.terminated:
            info["success"] = self.state.wins >= self.config.target_wins
            return Transition(float(outcome), True, False, info)

        self.state.turn += 1
        if self.state.turn > self.config.max_turns:
            self.state.truncated = True
            info["reason"] = "turn_limit"
            return Transition(float(outcome), False, True, info)

        self.state.gold = self.config.starting_gold
        self.state.actions_this_turn = 0
        if self.expanded_rules:
            for pet in self.state.team:
                pet.ability_uses = 0
            if self.state.turn == 3 and self.state.lives < self.config.starting_lives:
                self.state.lives += 1
        self._roll_shop(free=True)
        if self.expanded_rules:
            self._shop_runtime().phase("start_turn")
        return Transition(float(outcome), False, False, info)

    def _shop_runtime(self):
        from .events import EventRuntime
        from .late_events import LateGameEventRuntime
        from .midgame_events import MidgameEventRuntime
        from .tier4_events import Tier4EventRuntime

        runtime_type = (
            LateGameEventRuntime
            if self.fullpack_rules
            else Tier4EventRuntime
            if self.tier4_rules
            else MidgameEventRuntime
            if self.midgame_rules
            else EventRuntime
        )
        return runtime_type(
            self.catalog,
            self.rng,
            [self.state.team, []],
            combat=False,
            state=self.state,
            max_team_size=self.config.max_team_size,
        )

    def _food_multiplier(self, food):
        if food.effect not in {"buff", "buff_random_team", "buff_future_shop"}:
            return 1
        multiplier = 1
        for pet in self.state.team:
            for ability in self.catalog.pets[pet.copied_ability or pet.spec_id].abilities:
                if ability.effect == "multiply_food" and pet.ability_uses < 2:
                    multiplier += ability.at_level("multiplier", pet.level) - 1
                    pet.ability_uses += 1
        return multiplier

    def _feed_experience(self, target, amount):
        from .catalog import AbilitySpec
        from .events import Event

        before = target.level
        # XP grants +1/+1 per point even when already level three. Unlike a
        # merge, it never copies another pet's larger stats or held perk.
        self._shop_runtime().buff(target, amount, amount)
        target.experience = min(6, target.experience + amount)
        if target.level > before:
            target.ability_uses = 0
        if target.spec_id == "cow":
            ability = AbilitySpec(
                "friendly_ate_food",
                "replace_milk",
                {
                    "food_id_by_level": [
                        "chocolate_milk",
                        "better_chocolate_milk",
                        "best_chocolate_milk",
                    ]
                },
            )
            self._shop_runtime().apply_effect(Event("friendly_ate_food", 0, target, ability))
        if target.level > before:
            self._run_shop_trigger(target, "level_up", target, ability_level=before)
            self._stock_level_up_choices()


def _abilities(catalog: Catalog, pet: Pet, trigger: str) -> Iterable[AbilitySpec]:
    return (
        ability for ability in catalog.pets[pet.spec_id].abilities if ability.trigger == trigger
    )


def resolve_battle(
    friendly: Sequence[Pet],
    enemy: Sequence[Pet],
    catalog: Catalog,
    rng: random.Random,
    max_team_size: int = 5,
) -> BattleResult:
    """Resolve one battle on copies and return a human-readable event trace."""

    if catalog.rules_version in {4, 5, 6, 7, 8}:
        from .events import battle_runtime
        from .late_events import late_battle_runtime
        from .midgame_events import midgame_battle_runtime
        from .tier4_events import tier4_battle_runtime

        make_runtime = (
            late_battle_runtime
            if catalog.rules_version in {7, 8}
            else tier4_battle_runtime
            if catalog.rules_version == 6
            else midgame_battle_runtime
            if catalog.rules_version == 5
            else battle_runtime
        )
        runtime = make_runtime(friendly, enemy, catalog, rng, max_team_size)
        outcome, attacks = runtime.battle()
        return BattleResult(outcome, runtime.trace, attacks, runtime.attack_limit_reached)

    teams = [
        [pet.battle_copy() for pet in friendly],
        [pet.battle_copy() for pet in enemy],
    ]
    trace: List[str] = []

    def label(side: int, pet: Pet) -> str:
        return f"{'you' if side == 0 else 'opponent'}:{catalog.pets[pet.spec_id].name}"

    def notify_summon(side: int, subject: Pet) -> None:
        for owner in teams[side]:
            if owner is subject or owner.health <= 0:
                continue
            for ability in _abilities(catalog, owner, "friend_summoned"):
                subject.attack += ability.at_level("attack", owner.level)
                subject.health += ability.at_level("health", owner.level)
                trace.append(f"{label(side, owner)} buffs summoned {label(side, subject)}")

    def resolve_faints(side: int) -> None:
        guard = 0
        while any(pet.health <= 0 for pet in teams[side]):
            guard += 1
            if guard > 50:
                raise RuntimeError("Faint resolution exceeded safety limit")
            faint_index = next(i for i, pet in enumerate(teams[side]) if pet.health <= 0)
            fainted = teams[side].pop(faint_index)
            trace.append(f"{label(side, fainted)} faints")
            summons: List[Pet] = []
            for ability in _abilities(catalog, fainted, "faint"):
                if ability.effect == "buff_random_friend" and teams[side]:
                    count = min(len(teams[side]), ability.at_level("count", fainted.level, 1))
                    for target in rng.sample(teams[side], count):
                        target.attack += ability.at_level("attack", fainted.level)
                        target.health += ability.at_level("health", fainted.level)
                        trace.append(f"{label(side, fainted)} buffs {label(side, target)}")
                elif ability.effect == "summon":
                    count = ability.at_level("count", fainted.level, 1)
                    for _ in range(count):
                        token = make_pet(catalog, str(ability.params["summon_id"]))
                        token.attack = ability.at_level(
                            "summon_attack", fainted.level, token.attack
                        )
                        token.health = ability.at_level(
                            "summon_health", fainted.level, token.health
                        )
                        summons.append(token)
            if fainted.perk == "honey":
                summons.append(make_pet(catalog, "bee"))
            for token in reversed(summons):
                if len(teams[side]) >= max_team_size:
                    break
                teams[side].insert(min(faint_index, len(teams[side])), token)
                notify_summon(side, token)
                trace.append(f"{label(side, fainted)} summons {label(side, token)}")

    # Start-of-battle effects are ordered by attack, with seeded tie breaking.
    starters: List[Tuple[float, int, Pet, AbilitySpec]] = []
    for side in (0, 1):
        for pet in list(teams[side]):
            for ability in _abilities(catalog, pet, "start_battle"):
                starters.append((pet.attack + rng.random() * 1e-6, side, pet, ability))
    starters.sort(key=lambda entry: entry[0], reverse=True)
    for _, side, owner, ability in starters:
        opponents = [pet for pet in teams[1 - side] if pet.health > 0]
        if ability.effect == "damage_random_enemy" and opponents:
            count = min(len(opponents), ability.at_level("count", owner.level, 1))
            for target in rng.sample(opponents, count):
                damage = ability.at_level("damage", owner.level)
                target.health -= damage
                trace.append(f"{label(side, owner)} deals {damage} to {label(1 - side, target)}")
    resolve_faints(0)
    resolve_faints(1)

    attacks = 0
    while teams[0] and teams[1]:
        attacks += 1
        if attacks > 200:
            raise RuntimeError("Battle exceeded attack safety limit")
        left, right = teams[0][0], teams[1][0]
        left_damage, right_damage = left.attack, right.attack
        left.health -= right_damage
        right.health -= left_damage
        trace.append(
            f"{label(0, left)} and {label(1, right)} attack for {left_damage}/{right_damage}"
        )
        resolve_faints(0)
        resolve_faints(1)

    if teams[0] and not teams[1]:
        outcome = BattleOutcome.WIN
    elif teams[1] and not teams[0]:
        outcome = BattleOutcome.LOSS
    else:
        outcome = BattleOutcome.DRAW
    trace.append(f"battle ends: {outcome.name.lower()}")
    return BattleResult(outcome=outcome, trace=trace, attacks=attacks)
