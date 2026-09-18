"""Midgame event extensions under construction, isolated from frozen v4 runs.

This is not yet an episode-training environment. See MIDGAME_RULES.md for the
versioned specification, unresolved dependencies, and explicit integration gate.
"""

from __future__ import annotations

from .catalog import AbilitySpec
from .domain import Pet
from .events import Event, EventRuntime, attack, health, percent_amount


class MidgameEventRuntime(EventRuntime):
    """Reuse v4 scheduling while implementing the new midgame effect primitives."""

    priorities = {**EventRuntime.priorities, "knock_out": 4.5}
    dead_owner_triggers = EventRuntime.dead_owner_triggers | {"knock_out"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_limits = set()
        self.swallowed = {}
        self.damage_owner = None

    def abilities(self, pet):
        return self.catalog.pets[pet.copied_ability or pet.spec_id].abilities

    def emit(self, side, owner, trigger, subject=None, level=None):
        super().emit(side, owner, trigger, subject, level)
        if trigger == "end_turn" and owner.perk == "cake":
            self.queue.append(
                Event(
                    trigger,
                    side,
                    owner,
                    AbilitySpec(trigger, "increase_sell_value", {}),
                    tie=self.rng.random(),
                )
            )

    def phase(self, trigger: str) -> None:
        self.phase_limits.clear()
        if trigger == "start_turn":
            for team in self.teams:
                for pet in team:
                    pet.copied_ability = None
        super().phase(trigger)

    def apply(self, event: Event) -> None:
        previous = self.damage_owner
        self.damage_owner = (event.side, event.owner)
        try:
            self.apply_effect(event)
        finally:
            self.damage_owner = previous

    def apply_effect(self, event: Event) -> None:
        ability, owner, side = event.ability, event.owner, event.side
        assert ability is not None
        effect = ability.effect
        handled = {
            "buff_level_friends",
            "buff_if_level_friend",
            "discount_shop_food",
            "perk_position",
            "reduce_highest_health",
            "copy_ahead_ability",
            "swallow_ahead",
            "increase_sell_value",
        }
        if effect not in handled:
            return super().apply(event)
        level = event.level or owner.level

        def value(key, default=0):
            return ability.at_level(key, level, default)

        self.trace.append(f"{event.trigger} {self.label(side, owner)} L{level}: {effect}")
        if effect == "copy_ahead_ability":
            targets = self.nearby(side, owner, "ahead", 1)
            if targets:
                target = targets[0]
                owner.copied_ability = target.copied_ability or target.spec_id
                owner.ability_uses = 0
        elif effect == "swallow_ahead":
            targets = self.nearby(side, owner, "ahead", 1)
            if targets:
                target = targets[0]
                # A released pet is new, not a restored object graph. Preserve
                # displayed stats only, and give it the swallowing ability's level.
                released = Pet(
                    target.spec_id, attack(target), health(target), experience=(1, 3, 6)[level - 1]
                )
                self.swallowed.setdefault(id(owner), []).append(released)
                self.faint(side, target)
        elif effect == "increase_sell_value":
            owner.sell_bonus += 1
        elif effect == "buff_if_level_friend":
            # A level-three Bison is not its own level-three friend. The shared
            # limit is per team and phase, not a separate allowance per copy.
            friends = [p for p in self.alive(side) if p is not owner]
            if not any(p.level >= value("friend_level", 3) for p in friends):
                return
            group = ability.params.get("team_limit_group")
            key = (side, event.trigger, group)
            if group is not None and key in self.phase_limits:
                return
            if group is not None:
                self.phase_limits.add(key)
            self.buff(owner, value("attack"), value("health"))
        elif effect == "buff_level_friends":
            friends = [
                p
                for p in self.alive(side)
                if p is not owner and p.level >= value("friend_level", 2)
            ]
            # Match the inherited random-buff preference: level eligibility is
            # applied first, then prefer non-maxed pets, without duplicates.
            preferred = [p for p in friends if attack(p) < 50 or health(p) < 50]
            count = min(value("count", 2), len(friends))
            selected = self.rng.sample(preferred, min(count, len(preferred)))
            if len(selected) < count:
                chosen = {id(p) for p in selected}
                remaining = [p for p in friends if id(p) not in chosen]
                selected += self.rng.sample(remaining, count - len(selected))
            for target in selected:
                self.buff(target, value("attack"), value("health"))
        elif effect == "discount_shop_food":
            if self.state is None or self.combat:
                raise RuntimeError("Shop food discounts require shop state")
            for item in self.state.shop:
                if item is not None and item.kind == "food":
                    item.cost = max(0, item.cost - value("discount"))
        elif effect == "perk_position":
            perk = str(ability.params["perk"])
            # An identical perk is skipped when choosing recipients, rather
            # than spending a target on a friend who cannot gain it (0.24).
            friends = self.nearby(
                side,
                owner,
                str(ability.params.get("direction", "behind")),
                self.max_team_size,
            )
            targets = [p for p in friends if p.perk != perk][: value("count", 1)]
            for target in targets:
                self.gain_perk(side, target, perk, notify_food=True)
        elif effect == "reduce_highest_health":
            enemies = self.alive(1 - side)
            if not enemies:
                return
            highest = max(health(p) for p in enemies)
            target = self.rng.choice([p for p in enemies if health(p) == highest])
            removed = min(
                highest - 1,
                percent_amount(highest, value("percent"), ability.params.get("rounding", "ceil")),
            )
            # Health removal is NOT a hit: bypass armor without consuming it,
            # does not trigger Hurt, and cannot kill a one-health target.
            temporary = min(target.temporary_health, removed)
            target.temporary_health -= temporary
            target.health -= removed - temporary
            self.trace.append(f"{self.label(1 - side, target)} loses {removed} health")

    def extra_faint_summons(self, side, owner):
        return [(side, pet, True, False) for pet in self.swallowed.pop(id(owner), [])]

    def damage_batch(self, hits, *, sources=None):
        """Simultaneous damage with source attribution; health removal stays separate."""
        if sources is None:
            sources = [self.damage_owner] * len(hits)
        if len(hits) != len(sources):
            raise ValueError("Every hit needs its own source entry")
        knockouts = []
        for (side, target, amount), source in zip(hits, sources):
            if amount <= 0 or health(target) <= 0:
                continue
            if target.perk == "melon":
                target.perk = None
                amount = max(0, amount - 20)
                self.trace.append(f"{self.label(side, target)} consumes melon")
            elif target.perk == "garlic":
                # 0.40+: minimum reduced damage is two. A one-damage hit is
                # not amplified; zero damage remains zero and cannot cause Hurt.
                amount = min(amount, max(2, amount - 2))
            if not amount:
                continue
            absorbed = min(target.temporary_health, amount)
            target.temporary_health -= absorbed
            target.health -= amount - absorbed
            self.trace.append(f"{self.label(side, target)} takes {amount} damage")
            self.emit(side, target, "hurt")
            if health(target) <= 0 and source is not None:
                knockouts.append((*source, target))
        self.queue_faints()
        for side, source, target in knockouts:
            self.emit(side, source, "knock_out", target)

    def exchange_damage(self, left, right, left_damage, right_damage):
        hits = [(0, left, right_damage), (1, right, left_damage)]
        sources = [(1, right), (0, left)]
        # Select both splash targets before any damage/faint/summon resolution.
        for side, owner in ((0, left), (1, right)):
            enemies = self.alive(1 - side)
            if owner.perk == "chili" and len(enemies) >= 2:
                hits.append((1 - side, enemies[1], 5))
                sources.append((side, owner))
        self.damage_batch(hits, sources=sources)


def midgame_battle_runtime(friendly, enemy, catalog, rng, max_team_size=5):
    teams = [[pet.battle_copy() for pet in team] for team in (friendly, enemy)]
    for team in teams:
        for pet in team:
            pet.attack, pet.health = min(50, pet.attack), min(50, pet.health)
            if (pet.copied_ability or pet.spec_id) == "hippo":
                pet.ability_uses = 0  # Hippo's quota is per battle, not per shop.
    return MidgameEventRuntime(catalog, rng, teams, combat=True, max_team_size=max_team_size)
