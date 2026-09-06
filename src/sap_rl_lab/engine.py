"""Pure-Python shop and battle engine with explicit, atomic transitions."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .actions import Action, ActionCodec, ActionKind
from .catalog import AbilitySpec, Catalog, load_catalog
from .domain import BattleOutcome, GameConfig, GameState, Pet, ShopItem


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


OpponentProvider = Callable[[int, random.Random, Catalog, GameConfig], Sequence[Pet]]


def make_pet(catalog: Catalog, pet_id: str) -> Pet:
    spec = catalog.pets[pet_id]
    return Pet(spec_id=spec.id, attack=spec.attack, health=spec.health)


def default_opponent(
    turn: int, rng: random.Random, catalog: Catalog, config: GameConfig
) -> Sequence[Pet]:
    """A deliberately transparent baseline, not a claim about arena matchmaking."""

    available = catalog.pet_ids_through_tier(min(6, 1 + (turn - 1) // 2))
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
        self.config = config or GameConfig()
        self.codec = ActionCodec(self.config.max_team_size, self.config.max_shop_size)
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
        return min(6, 1 + (self.state.turn - 1) // 2)

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
                for team_index in range(len(self.state.team)):
                    legal.append(Action(ActionKind.BUY_FOOD, source=shop_index, target=team_index))

        for team_index in range(len(self.state.team)):
            legal.append(Action(ActionKind.SELL, source=team_index))
        for team_index in range(max(0, len(self.state.team) - 1)):
            legal.append(Action(ActionKind.SWAP, source=team_index, target=team_index + 1))
        return tuple(legal)

    def legal_action_ids(self) -> Tuple[int, ...]:
        return tuple(self.codec.encode(action) for action in self.legal_actions())

    def action_mask(self) -> Tuple[bool, ...]:
        legal = set(self.legal_action_ids())
        return tuple(action_id in legal for action_id in range(self.codec.size))

    def step_id(self, action_id: int) -> Transition:
        return self.step(self.codec.decode(action_id))

    def step(self, action: Action) -> Transition:
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
        self.state.shop[shop_index] = None
        self.state.gold -= item.cost
        pet = make_pet(self.catalog, item.item_id)
        self.state.team.append(pet)
        self._run_shop_trigger(pet, "buy", subject=pet)
        self._notify_friend_summoned(pet, battle=False)

    def _buy_food(self, shop_index: int, team_index: int) -> None:
        item = self.state.shop[shop_index]
        assert item is not None
        self.state.shop[shop_index] = None
        self.state.gold -= item.cost
        food = self.catalog.foods[item.item_id]
        target = self.state.team[team_index]
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
        self.state.shop[shop_index] = None
        self.state.gold -= item.cost
        target = self.state.team[team_index]
        previous_level = target.level
        incoming = make_pet(self.catalog, item.item_id)
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

    def _sell(self, team_index: int) -> None:
        pet = self.state.team.pop(team_index)
        self.state.gold += pet.level
        self._run_shop_trigger(pet, "sell", subject=pet)

    def _run_shop_trigger(
        self, owner: Pet, trigger: str, subject: Pet, ability_level: Optional[int] = None
    ) -> None:
        spec = self.catalog.pets[owner.spec_id]
        level = owner.level if ability_level is None else ability_level
        for ability in spec.abilities:
            if ability.trigger != trigger:
                continue
            if ability.effect == "gain_gold":
                self.state.gold += ability.at_level("gold", level)
            elif ability.effect == "stock_food":
                count = ability.at_level("count", level, 1)
                for _ in range(count):
                    self._stock_food(str(ability.params["food_id"]))
            elif ability.effect == "buff_random_friend":
                friends = [pet for pet in self.state.team if pet is not subject]
                count = min(len(friends), ability.at_level("count", level, 1))
                for friend in self.rng.sample(friends, count):
                    friend.attack = min(50, friend.attack + ability.at_level("attack", level))
                    friend.health = min(50, friend.health + ability.at_level("health", level))
            else:
                raise RuntimeError(
                    f"Effect {ability.effect} is not valid for shop trigger {trigger}"
                )

    def _notify_friend_summoned(self, subject: Pet, battle: bool) -> None:
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
                new_shop.append(ShopItem("pet", pet_id, self.config.pet_cost))
            else:
                food_id = self.rng.choice(food_ids)
                new_shop.append(ShopItem("food", food_id, self.catalog.foods[food_id].cost))
        self.state.shop = new_shop

    def _end_turn(self) -> Transition:
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
        self._roll_shop(free=True)
        return Transition(float(outcome), False, False, info)


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
