"""Version-4 event runtime shared by combat and shop-side faint effects.

Pets are tracked by object identity. Death effects retain their owner's position
until after-faint summons; simultaneous hits are applied before resolving events.
The frozen eight-pet resolver does not call this module. See TIER12_RULES.md for
the source ledger and client-parity limitations of this executable specification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .catalog import AbilitySpec, Catalog
from .domain import BattleOutcome, GameState, Pet, ShopItem
from .shop import ContentNotReady, shop_slots, stock_items

PRIORITY = {
    "start_turn": -30,
    "end_turn": -30,
    "start_battle": -30,
    "buy": -30,
    "sell": -30,
    "level_up": -30,
    "after_attack": -20,
    "friend_ahead_attacks": -10,
    "hurt": 0,
    "friend_summoned": 1,
    "faint": 2,
    "friend_ahead_faints": 3,
    "after_faint": 4,
    "friendly_ate_food": 5,
}
SUMMON_EFFECTS = {"summon", "summon_tier", "summon_enemy"}
PERKS = ("honey", "meat_bone", "melon")


def attack(pet: Pet) -> int:
    return pet.attack + pet.temporary_attack


def health(pet: Pet) -> int:
    return pet.health + pet.temporary_health


def percent_amount(value: int, percent: int, rounding: str = "ceil") -> int:
    """Per-ability rounding; Dodo's observed floor is not a global rule."""
    if rounding == "floor":
        return value * percent // 100
    if rounding == "ceil":
        return (value * percent + 99) // 100
    raise ValueError(f"Unsupported percentage rounding: {rounding}")


def index_of(team: Sequence[Pet], pet: Pet) -> Optional[int]:
    return next((i for i, other in enumerate(team) if other is pet), None)


@dataclass
class Event:
    trigger: str
    side: int
    owner: Pet
    ability: Optional[AbilitySpec] = None
    subject: Optional[Pet] = None
    level: Optional[int] = None
    tie: float = 0.0


class EventRuntime:
    """One event queue, either operating on shop pets or on battle copies."""

    priorities = PRIORITY
    dead_owner_triggers = {"sell", "hurt", "faint", "after_attack", "start_battle"}

    def __init__(
        self,
        catalog: Catalog,
        rng: random.Random,
        teams: List[List[Pet]],
        *,
        combat: bool,
        state: Optional[GameState] = None,
        max_team_size: int = 5,
    ) -> None:
        self.catalog, self.rng, self.teams = catalog, rng, teams
        self.combat, self.state, self.max_team_size = combat, state, max_team_size
        self.queue: List[Event] = []
        self.faint_queued = {}  # Keep references: Python must not recycle dead-pet IDs.
        self.trace: List[str] = []
        self.events_processed = 0
        self.attack_limit_reached = False

    def label(self, side: int, pet: Pet) -> str:
        return f"{'you' if side == 0 else 'opponent'}:{pet.spec_id}"

    def alive(self, side: int) -> List[Pet]:
        return [pet for pet in self.teams[side] if health(pet) > 0]

    def abilities(self, pet: Pet):
        return self.catalog.pets[pet.spec_id].abilities

    def emit(
        self,
        side: int,
        owner: Pet,
        trigger: str,
        subject: Optional[Pet] = None,
        level: Optional[int] = None,
    ) -> None:
        for ability in self.abilities(owner):
            if ability.trigger == trigger and not (
                trigger == "faint" and ability.effect in SUMMON_EFFECTS
            ):
                self.queue.append(
                    Event(trigger, side, owner, ability, subject, level, self.rng.random())
                )

    def phase(self, trigger: str) -> None:
        for side in (0, 1):
            for pet in self.alive(side):
                self.emit(side, pet, trigger)
        self.drain()

    def nearby(self, side: int, owner: Pet, direction: str, count: int) -> List[Pet]:
        team = self.teams[side]
        index = index_of(team, owner)
        if index is None:
            return []
        candidates = list(reversed(team[:index])) if direction == "ahead" else team[index + 1 :]
        return [pet for pet in candidates if health(pet) > 0][:count]

    def buff(self, pet: Pet, atk: int = 0, hp: int = 0, temporary: bool = False) -> None:
        # Temporary health absorbs shop damage first. Permanent gains keep their
        # duration even if temporary stats already bring the displayed total to 50.
        if temporary and not self.combat:
            pet.temporary_attack = min(max(0, 50 - pet.attack), pet.temporary_attack + atk)
            pet.temporary_health = min(max(0, 50 - pet.health), pet.temporary_health + hp)
        else:
            pet.attack = min(50, pet.attack + atk)
            pet.health = min(50, pet.health + hp)
            pet.temporary_attack = min(pet.temporary_attack, max(0, 50 - pet.attack))
            pet.temporary_health = min(pet.temporary_health, max(0, 50 - pet.health))

    def notify_summon(self, side: int, subject: Pet) -> None:
        for owner in self.alive(side):
            if owner is not subject:
                self.emit(side, owner, "friend_summoned", subject)

    def notify_food_eaten(self, side: int, subject: Pet) -> None:
        for owner in self.alive(side):
            self.emit(side, owner, "friendly_ate_food", subject)

    def gain_perk(self, side: int, subject: Pet, perk: str, *, notify_food: bool) -> None:
        # Ability-gained perks skip an identical held perk (official 0.24).
        # A newly gained perk counts as food eaten since official 0.36.
        # Purchased perk foods notify separately, exactly once even if identical.
        if subject.perk == perk:
            return
        subject.perk = perk
        if notify_food:
            self.notify_food_eaten(side, subject)

    def damage_batch(self, hits: Sequence[Tuple[int, Pet, int]]) -> None:
        # The hit list is selected before applying any damage. No intervening
        # buff, death or summon can retroactively change the opposing attack.
        for side, target, amount in hits:
            if amount <= 0 or health(target) <= 0:
                continue
            if target.perk == "melon":
                target.perk = None
                amount = max(0, amount - 20)
                self.trace.append(f"{self.label(side, target)} consumes melon")
            if not amount:
                continue  # Fully blocked damage is not Hurt.
            absorbed = min(target.temporary_health, amount)
            target.temporary_health -= absorbed
            target.health -= amount - absorbed
            self.trace.append(f"{self.label(side, target)} takes {amount} damage")
            self.emit(side, target, "hurt")  # Lethal damage still triggers Hurt.
        self.queue_faints()

    def queue_faints(self) -> None:
        for side in (0, 1):
            team = self.teams[side]
            for index, pet in enumerate(team):
                if health(pet) > 0 or id(pet) in self.faint_queued:
                    continue
                self.faint_queued[id(pet)] = pet
                self.emit(side, pet, "faint")
                # Snapshot adjacency before removal, so multiple simultaneous
                # faints cannot manufacture extra "immediately ahead" triggers.
                if index + 1 < len(team):
                    self.emit(side, team[index + 1], "friend_ahead_faints", pet)
                self.queue.append(Event("after_faint", side, pet, tie=self.rng.random()))

    def faint(self, side: int, pet: Pet) -> None:
        # Sleeping Pill is direct faint, not damage: ignores armor, no Hurt.
        pet.health = pet.temporary_health = 0
        self.queue_faints()

    def drain(self) -> None:
        while self.queue:
            self.events_processed += 1
            if self.events_processed > 10_000:
                raise RuntimeError("Event safety limit exceeded; simulator error, not game loss")
            index = min(
                range(len(self.queue)),
                key=lambda i: (
                    self.priorities[self.queue[i].trigger],
                    -attack(self.queue[i].owner),
                    self.queue[i].tie,
                ),
            )
            event = self.queue.pop(index)
            if event.trigger == "after_faint":
                self.after_faint(event.side, event.owner)
                continue
            # Already queued start abilities precede faint resolution, including
            # those of lethally sniped owners. They cannot revive dead targets.
            if event.trigger not in self.dead_owner_triggers and (
                health(event.owner) <= 0 or index_of(self.teams[event.side], event.owner) is None
            ):
                continue
            self.apply(event)

    def apply(self, event: Event) -> None:
        ability, owner, side = event.ability, event.owner, event.side
        assert ability is not None
        level = event.level or owner.level

        def value(key, default=0):
            return ability.at_level(key, level, default)

        if ability.params.get("requires_previous_loss") and (
            self.state is None or self.state.previous_outcome is not BattleOutcome.LOSS
        ):
            return
        if ability.effect == "buff_eater" and (
            event.subject is None
            or health(event.subject) <= 0
            or index_of(self.teams[side], event.subject) is None
        ):
            return
        limit = value("max_uses")
        if limit:
            if owner.ability_uses >= limit:
                return
            owner.ability_uses += 1
        self.trace.append(f"{event.trigger} {self.label(side, owner)} L{level}: {ability.effect}")
        effect = ability.effect
        atk, hp, count = value("attack"), value("health"), value("count", 1)
        temporary = bool(ability.params.get("temporary", False))
        if effect == "buff_self":
            self.buff(owner, atk, hp if health(owner) > 0 else 0, temporary)
        elif effect in {"buff_subject", "buff_eater"}:
            if event.subject is not None and health(event.subject) > 0:
                self.buff(event.subject, atk, hp, temporary)
        elif effect == "buff_random_friend":
            friends = [pet for pet in self.alive(side) if pet is not owner]
            preferred = friends
            if ability.params.get("prefer_nonmaxed"):
                # Official 0.29: random buffs prioritize non-maxed pets. This
                # uses displayed 50/50, not a speculative per-stat optimizer.
                preferred = [pet for pet in friends if attack(pet) < 50 or health(pet) < 50]
            selected = self.rng.sample(preferred, min(count, len(preferred)))
            if len(selected) < min(count, len(friends)):
                selected_ids = {id(pet) for pet in selected}
                remaining = [pet for pet in friends if id(pet) not in selected_ids]
                selected += self.rng.sample(remaining, min(count - len(selected), len(remaining)))
            for pet in selected:
                self.buff(pet, atk, hp, temporary)
        elif effect == "buff_position":
            for pet in self.nearby(side, owner, str(ability.params["direction"]), count):
                self.buff(pet, atk, hp, temporary)
        elif effect == "gain_gold":
            assert self.state is not None
            self.state.gold += value("gold")
        elif effect == "stock_food":
            assert self.state is not None
            ids = ability.params.get("food_id_by_level")
            food_id = str(ids[level - 1] if ids else ability.params["food_id"])
            food = self.catalog.foods[food_id]
            items = [ShopItem("food", food_id, value("cost", food.cost)) for _ in range(count)]
            stock_items(self.state.shop, items, shop_slots(self.state.turn)[2], "food")
        elif effect == "buff_shop_pets":
            assert self.state is not None
            for item in self.state.shop:
                if item and item.kind == "pet":
                    assert item.pet is not None
                    self.buff(item.pet, atk, hp)
        elif effect == "gain_health_percent":
            friends = [pet for pet in self.alive(side) if pet is not owner]
            if friends and health(owner) > 0:
                self.buff(
                    owner,
                    hp=percent_amount(
                        max(health(pet) for pet in friends),
                        value("percent"),
                        ability.params.get("rounding", "ceil"),
                    ),
                )
        elif effect == "share_attack":
            for pet in self.nearby(side, owner, "ahead", 1):
                self.buff(
                    pet,
                    atk=percent_amount(
                        attack(owner), value("percent"), ability.params.get("rounding", "ceil")
                    ),
                )
        elif effect == "damage_random_enemy":
            enemies = self.alive(1 - side)
            self.damage_batch(
                [
                    (1 - side, pet, value("damage"))
                    for pet in self.rng.sample(enemies, min(count, len(enemies)))
                ]
            )
        elif effect in {"damage_lowest_enemy", "damage_behind"}:
            for _ in range(value("repeats", 1)):
                if effect == "damage_behind":
                    targets, target_side = self.nearby(side, owner, "behind", 1), side
                else:
                    enemies, target_side = self.alive(1 - side), 1 - side
                    lowest = min((health(pet) for pet in enemies), default=0)
                    targets = (
                        [self.rng.choice([pet for pet in enemies if health(pet) == lowest])]
                        if enemies
                        else []
                    )
                self.damage_batch([(target_side, pet, value("damage")) for pet in targets])
        elif effect == "damage_all":
            self.damage_batch(
                [
                    (target_side, pet, value("damage"))
                    for target_side in (0, 1)
                    for pet in self.alive(target_side)
                    if pet is not owner
                ]
            )
        elif effect == "damage_adjacent":
            targets = [
                (side, pet)
                for direction in ("ahead", "behind")
                for pet in self.nearby(side, owner, direction, 1)
            ]
            if index_of(self.teams[side], owner) == 0 and self.combat:
                targets += [(1 - side, pet) for pet in self.alive(1 - side)[:1]]
            self.damage_batch(
                [
                    (
                        s,
                        p,
                        percent_amount(
                            attack(owner), value("percent"), ability.params.get("rounding", "ceil")
                        ),
                    )
                    for s, p in targets
                ]
            )
        elif effect == "gain_melon":
            self.buff(owner, atk=atk)
            self.gain_perk(
                side, owner, "melon", notify_food=bool(ability.params.get("perk_counts_as_food"))
            )
        else:
            raise RuntimeError(f"No version-4 implementation for {effect}")

    def after_faint(self, side: int, owner: Pet) -> None:
        index = index_of(self.teams[side], owner)
        if index is None:
            return
        self.teams[side].pop(index)
        self.trace.append(f"{self.label(side, owner)} faints")
        summons = []
        for ability in self.abilities(owner):
            if ability.trigger != "faint" or ability.effect not in SUMMON_EFFECTS:
                continue
            if ability.effect == "summon_enemy" and not self.combat:
                continue  # There is no opposing shop team for a pilled Rat.
            for _ in range(ability.at_level("count", owner.level, 1)):
                if ability.effect == "summon_tier":
                    tier = int(ability.params["tier"])
                    ids = [
                        pet.id
                        for pet in self.catalog.pets.values()
                        if pet.tier == tier and not pet.token
                    ]
                    if tier not in self.catalog.implemented_shop_tiers or not ids:
                        raise ContentNotReady(f"Missing exact Tier {tier} summon pool")
                    pet_id = self.rng.choice(ids)
                else:
                    pet_id = str(ability.params["summon_id"])
                spec = self.catalog.pets[pet_id]
                pet = Pet(
                    pet_id,
                    ability.at_level("summon_attack", owner.level, spec.attack),
                    ability.at_level("summon_health", owner.level, spec.health),
                    experience=(1, 3, 6)[owner.level - 1]
                    if ability.params.get("match_level")
                    else 1,
                    perk=ability.params.get("summon_perk"),
                )
                enemy = ability.effect == "summon_enemy"
                summons.append((1 - side if enemy else side, pet, not enemy, False))
        summons.extend(self.extra_faint_summons(side, owner))
        if owner.perk == "honey":
            spec = self.catalog.pets["bee"]
            # The pet's own summon is created first; the later Bee is inserted
            # ahead of it. This matters for Spider -> Ox and -> Dog interactions.
            summons.append((side, Pet("bee", spec.attack, spec.health), True, True))
        offset = 0
        for target_side, pet, notify, ahead_of_own_summons in summons:
            if len(self.alive(target_side)) >= self.max_team_size:
                continue
            position = index + (0 if ahead_of_own_summons else offset) if target_side == side else 0
            self.teams[target_side].insert(min(position, len(self.teams[target_side])), pet)
            if target_side == side:
                offset += 1
            self.trace.append(f"{self.label(side, owner)} summons {self.label(target_side, pet)}")
            if notify:
                self.notify_summon(target_side, pet)
                if pet.perk is not None:
                    self.notify_food_eaten(target_side, pet)

    def extra_faint_summons(self, side: int, owner: Pet):
        return []

    def battle(self) -> Tuple[BattleOutcome, int]:
        self.phase("start_battle")
        attacks = 0
        while self.alive(0) and self.alive(1):
            # A rules-level draw is distinct from the technical safety guard.
            # Resolve an exchange and its entire event queue before checking
            # this boundary, so a win on the final allowed exchange stays a win.
            limit = self.catalog.battle_attack_limit
            if limit is not None and attacks >= limit:
                self.attack_limit_reached = True
                self.trace.append(f"battle draw: attack limit reached ({attacks} exchanges)")
                self.trace.append("battle ends: draw")
                return BattleOutcome.DRAW, attacks
            attacks += 1
            if attacks > 200:
                raise RuntimeError("Battle exceeded attack safety limit")
            left, right = self.alive(0)[0], self.alive(1)[0]
            left_damage = attack(left) + (3 if left.perk == "meat_bone" else 0)
            right_damage = attack(right) + (3 if right.perk == "meat_bone" else 0)
            for side, pet in ((0, left), (1, right)):
                self.emit(side, pet, "after_attack")
                for friend in self.nearby(side, pet, "behind", 1):
                    self.emit(side, friend, "friend_ahead_attacks", pet)
            self.trace.append(
                f"attack {self.label(0, left)} / {self.label(1, right)}: "
                f"{left_damage}/{right_damage}"
            )
            self.exchange_damage(left, right, left_damage, right_damage)
            self.drain()
        outcome = (
            BattleOutcome.WIN
            if self.alive(0)
            else BattleOutcome.LOSS
            if self.alive(1)
            else BattleOutcome.DRAW
        )
        self.trace.append(f"battle ends: {outcome.name.lower()}")
        return outcome, attacks

    def exchange_damage(self, left: Pet, right: Pet, left_damage: int, right_damage: int):
        self.damage_batch([(0, left, right_damage), (1, right, left_damage)])


def battle_runtime(
    friendly: Sequence[Pet],
    enemy: Sequence[Pet],
    catalog: Catalog,
    rng: random.Random,
    max_team_size: int = 5,
) -> EventRuntime:
    teams = [[pet.battle_copy() for pet in team] for team in (friendly, enemy)]
    for team in teams:
        for pet in team:
            pet.attack, pet.health = min(50, pet.attack), min(50, pet.health)
    return EventRuntime(catalog, rng, teams, combat=True, max_team_size=max_team_size)
