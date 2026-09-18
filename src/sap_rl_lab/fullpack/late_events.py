"""Late-game triggers, finite repeats and ordered summons for the full-pack runtime."""

from dataclasses import replace

from .domain import Pet
from .events import SUMMON_EFFECTS, attack, health, index_of, percent_amount
from .midgame_events import MidgameEventRuntime
from .tier4_events import Tier4EventRuntime


class LateGameEventRuntime(Tier4EventRuntime):
    priorities = {
        **Tier4EventRuntime.priorities,
        "before_attack": -25,
        "friend_bought": -29,
        "fly_summon": 4.25,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faint_repeats = {}
        self.faint_anchors = {}

    def tiger_level(self, side, owner):
        if not self.combat:
            return None
        i = index_of(self.teams[side], owner)
        if i is None or i + 1 >= len(self.teams[side]):
            return None
        friend = self.teams[side][i + 1]
        if health(friend) > 0 and any(a.effect == "repeat_ahead" for a in self.abilities(friend)):
            return friend.level
        return None

    def emit(self, side, owner, trigger, subject=None, level=None):
        start = len(self.queue)
        super().emit(side, owner, trigger, subject, level)
        repeat = self.tiger_level(side, owner)
        if trigger == "faint" and repeat is not None:
            self.faint_repeats[id(owner)] = repeat
        # Trigger repetitions (e.g. Crocodile) are one ability activation. A
        # Tiger repeats that activation at the Tiger's level, not its perk.
        seen = set()
        for event in self.queue[start:]:
            ability = event.ability
            if ability is None:
                continue
            if ability.effect == "fly_summon":
                event.trigger = "fly_summon"  # After the dead pet's own summons.
            if (
                repeat is not None
                and trigger not in {"start_turn", "end_turn", "buy", "sell", "level_up"}
                and ability in self.abilities(owner)
                and ability.effect != "repeat_ahead"
                and id(ability) not in seen
            ):
                event.repeat_level = repeat
                seen.add(id(ability))

    def can_apply(self, event):
        ability, owner = event.ability, event.owner
        if ability.params.get("subject_tier") is not None:
            if event.subject is None:
                return False
            spec = self.catalog.pets[event.subject.spec_id]
            if (1 if spec.token else spec.tier) != ability.params["subject_tier"]:
                return False
        limit = ability.at_level("max_uses", owner.level)
        if limit and owner.ability_uses >= limit:
            return False
        if ability.effect == "fly_summon":
            return (
                event.subject is not None
                and event.subject.spec_id != "zombie_fly"
                and len(self.alive(event.side)) < self.max_team_size
            )
        if ability.effect == "gain_coconut":
            return health(owner) > 0 and owner.perk != "coconut"
        return True

    def apply(self, event):
        if not self.can_apply(event):
            return
        super().apply(event)
        if event.repeat_level is not None:
            count = event.ability.at_level("repeats", event.repeat_level, 1)
            for _ in range(count):
                self.queue.append(
                    replace(
                        event, level=event.repeat_level, repeat_level=None, tie=self.rng.random()
                    )
                )

    def apply_effect(self, event):
        ability, owner, side = event.ability, event.owner, event.side
        level = event.level or owner.level
        effect = ability.effect
        if effect not in {"gain_coconut", "damage_attack_percent", "fly_summon"}:
            return super().apply_effect(event)
        if ability.at_level("max_uses", owner.level):
            owner.ability_uses += 1
        self.trace.append(f"{event.trigger} {self.label(side, owner)} L{level}: {effect}")
        if effect == "gain_coconut":
            self.gain_perk(side, owner, "coconut", notify_food=True)
        elif effect == "damage_attack_percent":
            enemies = self.alive(1 - side)
            count = min(len(enemies), ability.at_level("count", level, 1))
            amount = percent_amount(
                attack(owner),
                ability.at_level("percent", level),
                ability.params.get("rounding", "ceil"),
            )
            self.damage_batch([(1 - side, p, amount) for p in self.rng.sample(enemies, count)])
        else:
            anchors = self.faint_anchors.get(id(event.subject), [])
            position = next(
                (
                    index_of(self.teams[side], p)
                    for p in anchors
                    if index_of(self.teams[side], p) is not None
                ),
                len(self.teams[side]),
            )
            pet = Pet("zombie_fly", 4 * level, 4 * level, experience=(1, 3, 6)[level - 1])
            self.summon(side, owner, pet, position)

    def before_exchange(self, left, right):
        self.emit(0, left, "before_attack")
        self.emit(1, right, "before_attack")
        self.drain()

    def attack_damage(self, pet):
        if pet.perk == "steak":
            pet.perk = None
            self.trace.append(f"{pet.spec_id} consumes steak")
            return attack(pet) + 20
        return super().attack_damage(pet)

    def damage_batch(self, hits, *, sources=None):
        filtered = []
        for side, target, amount in hits:
            if amount > 0 and health(target) > 0 and target.perk == "coconut":
                target.perk = None
                amount = 0
                self.trace.append(f"{self.label(side, target)} consumes coconut")
            filtered.append((side, target, amount))
        super().damage_batch(filtered, sources=sources)

    def exchange_damage(self, left, right, left_damage, right_damage):
        lethal = []
        for side, source, target, amount in (
            (0, left, right, left_damage),
            (1, right, left, right_damage),
        ):
            penetrates = target.perk != "coconut" and amount > (20 if target.perk == "melon" else 0)
            if source.perk == "peanut" and penetrates:
                lethal.append((side, source, target))
        MidgameEventRuntime.exchange_damage(self, left, right, left_damage, right_damage)
        for side, source, target in lethal:
            if health(target) > 0:
                target.health = target.temporary_health = 0
                self.trace.append(f"{self.label(side, source)} peanut knocks out {target.spec_id}")
                self.queue_faints()
                self.emit(side, source, "knock_out", target)

    def summon(self, side, owner, pet, position, notify=True):
        if len(self.alive(side)) >= self.max_team_size:
            return False
        self.teams[side].insert(min(position, len(self.teams[side])), pet)
        self.trace.append(f"{owner.spec_id} summons {self.label(side, pet)}")
        if notify:
            self.notify_summon(side, pet)
            if pet.perk is not None:
                self.notify_food_eaten(side, pet)
        return True

    def own_summons(self, side, owner, level):
        result = []
        for ability in self.abilities(owner):
            if ability.trigger != "faint":
                continue
            if ability.effect == "rooster_chicks":
                result += [
                    (side, Pet("chick", max(1, percent_amount(attack(owner), 50)), 1))
                    for _ in range(ability.at_level("count", level, 1))
                ]
            elif ability.effect in SUMMON_EFFECTS:
                enemy = ability.effect == "summon_enemy"
                if enemy and not self.combat:
                    continue
                for _ in range(ability.at_level("count", level, 1)):
                    if ability.effect == "summon_tier":
                        ids = [
                            p.id
                            for p in self.catalog.pets.values()
                            if not p.token and p.tier == ability.params["tier"]
                        ]
                        if not ids:
                            raise RuntimeError("Missing exact summon tier")
                        name = self.rng.choice(ids)
                    else:
                        name = ability.params["summon_id"]
                    spec = self.catalog.pets[name]
                    pet = Pet(
                        name,
                        ability.at_level("summon_attack", level, spec.attack),
                        ability.at_level("summon_health", level, spec.health),
                        experience=(1, 3, 6)[level - 1] if ability.params.get("match_level") else 1,
                        perk=ability.params.get("summon_perk"),
                    )
                    result.append((1 - side if enemy else side, pet))
        return result

    def after_faint(self, side, owner):
        index = index_of(self.teams[side], owner)
        if index is None:
            return
        self.teams[side].pop(index)
        self.trace.append(f"{self.label(side, owner)} faints")
        levels = [owner.level]
        if id(owner) in self.faint_repeats:
            levels.append(self.faint_repeats.pop(id(owner)))
        summons = [item for level in levels for item in self.own_summons(side, owner, level)]
        summons += [(side, pet) for pet in self.swallowed.pop(id(owner), [])]
        offset = 0
        for target_side, pet in summons:
            if self.summon(
                target_side,
                owner,
                pet,
                index + offset if target_side == side else 0,
                notify=target_side == side,
            ):
                offset += int(target_side == side)
        if owner.perk == "honey":
            self.summon(side, owner, Pet("bee", 1, 1), index)
        elif owner.perk == "mushroom":
            # A new instance, not a restored battle object or reusable Mushroom.
            self.summon(side, owner, Pet(owner.spec_id, 1, 1, experience=owner.experience), index)
        self.faint_anchors[id(owner)] = list(self.teams[side][index:])


def late_battle_runtime(friendly, enemy, catalog, rng, max_team_size=5):
    teams = [[p.battle_copy() for p in team] for team in (friendly, enemy)]
    for team in teams:
        for pet in team:
            pet.attack, pet.health = min(50, pet.attack), min(50, pet.health)
            if (pet.copied_ability or pet.spec_id) in {"hippo", "snake"}:
                pet.ability_uses = 0
    return LateGameEventRuntime(catalog, rng, teams, combat=True, max_team_size=max_team_size)
