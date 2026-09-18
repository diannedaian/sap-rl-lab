"""Opt-in v6: Tier5 reward abilities and Tier4 perks; v4/v5 are unchanged."""

from .catalog import AbilitySpec
from .domain import Pet, ShopItem
from .events import Event, attack, health, percent_amount
from .midgame_events import MidgameEventRuntime
from .shop import shop_slots


class Tier4EventRuntime(MidgameEventRuntime):
    priorities = {**MidgameEventRuntime.priorities, "summoned": 1, "friend_faints": 3}

    def emit(self, side, owner, trigger, subject=None, level=None):
        before = len(self.queue)
        super().emit(side, owner, trigger, subject, level)
        # Each Crocodile projectile is a distinct trigger and reselects a living target.
        for event in self.queue[before:]:
            if event.ability and event.ability.effect == "damage_last_enemy":
                for _ in range(event.ability.at_level("repeats", level or owner.level, 1) - 1):
                    self.queue.append(
                        Event(
                            trigger, side, owner, event.ability, subject, level, self.rng.random()
                        )
                    )
        if trigger == "end_turn" and owner.perk == "bread":
            self.queue.append(
                Event(
                    trigger,
                    side,
                    owner,
                    AbilitySpec(trigger, "bread_health", {}),
                    tie=self.rng.random(),
                )
            )

    def notify_summon(self, side, subject):
        self.emit(side, subject, "summoned", subject)
        super().notify_summon(side, subject)

    def queue_faints(self):
        newly_dead = [
            (side, p)
            for side in (0, 1)
            for p in self.teams[side]
            if health(p) <= 0 and id(p) not in self.faint_queued
        ]
        super().queue_faints()
        for side, subject in newly_dead:
            for friend in self.alive(side):
                if friend is not subject:
                    self.emit(side, friend, "friend_faints", subject)

    def apply_effect(self, event):
        ability, owner, side = event.ability, event.owner, event.side
        level = event.level or owner.level
        effect = ability.effect
        if ability.params.get("eater_is_owner") and event.subject is not owner:
            return
        if effect == "rooster_chicks":
            return  # Resolved in after_faint, where a team slot has actually opened.
        handled = {
            "buff_all_pets",
            "buff_front_pet",
            "replace_milk",
            "damage_last_enemy",
            "rhino_snipe",
            "gain_peanut",
            "bread_health",
        }
        if effect not in handled:
            return super().apply_effect(event)
        self.trace.append(f"{event.trigger} {self.label(side, owner)} L{level}: {effect}")
        if effect == "buff_all_pets":
            for target_side in (0, 1):
                for target in self.alive(target_side):
                    self.buff(target, hp=ability.at_level("health", level))
        elif effect == "buff_front_pet":
            targets = self.alive(side)[:1]  # Can target itself if it is front-most.
            for target in targets:
                self.buff(
                    target, ability.at_level("attack", level), ability.at_level("health", level)
                )
        elif effect == "bread_health":
            self.buff(owner, hp=7, temporary=True)
        elif effect == "gain_peanut":
            self.gain_perk(side, owner, "peanut", notify_food=True)
        elif effect == "replace_milk":
            if self.state is None or self.combat:
                raise RuntimeError("Milk replacement requires a shop")
            for i, item in enumerate(self.state.shop):
                if item is not None and item.kind == "food":
                    self.state.shop[i] = None  # Replacement includes frozen food.
            milk = ability.params["food_id_by_level"][level - 1]
            active = shop_slots(self.state.turn)[2]
            for i in (active - 1, active - 2):
                self.state.shop[i] = ShopItem("food", milk, 0)
        else:
            enemies = self.alive(1 - side)
            if not enemies:
                return
            target = enemies[-1] if effect == "damage_last_enemy" else enemies[0]
            amount = ability.at_level("damage", level)
            if effect == "rhino_snipe":
                spec = self.catalog.pets[target.spec_id]
                # Legacy tokens use tier 0 in the catalog, but count as Tier1 in-game.
                if spec.token or spec.tier == 1:
                    amount *= 2
            self.damage_batch([(1 - side, target, amount)])

    def extra_faint_summons(self, side, owner):
        summons = []
        for ability in self.abilities(owner):
            if ability.effect == "rooster_chicks":
                for _ in range(ability.at_level("count", owner.level, 1)):
                    chick = Pet("chick", max(1, percent_amount(attack(owner), 50, "ceil")), 1)
                    summons.append((side, chick, True, False))
        return summons + super().extra_faint_summons(side, owner)

    def exchange_damage(self, left, right, left_damage, right_damage):
        # Peanut applies only to a direct attack that penetrates armor, never an
        # ability snipe. Capture eligibility before either attack mutates a perk.
        lethal = []
        for side, source, target, amount in (
            (0, left, right, left_damage),
            (1, right, left, right_damage),
        ):
            penetrates = amount > (20 if target.perk == "melon" else 0)
            if source.perk == "peanut" and penetrates and amount > 0:
                lethal.append((side, source, target))
        super().exchange_damage(left, right, left_damage, right_damage)
        # Hurt is already queued by the attack. Do not create a second damage hit
        # or duplicate a knockout when the direct attack was lethal on its own.
        for side, source, target in lethal:
            if health(target) > 0:
                target.health = target.temporary_health = 0
                self.trace.append(f"{self.label(side, source)} peanut knocks out {target.spec_id}")
                self.queue_faints()
                self.emit(side, source, "knock_out", target)


def tier4_battle_runtime(friendly, enemy, catalog, rng, max_team_size=5):
    teams = [[p.battle_copy() for p in team] for team in (friendly, enemy)]
    for team in teams:
        for pet in team:
            pet.attack, pet.health = min(50, pet.attack), min(50, pet.health)
            if (pet.copied_ability or pet.spec_id) == "hippo":
                pet.ability_uses = 0
    return Tier4EventRuntime(catalog, rng, teams, combat=True, max_team_size=max_team_size)
