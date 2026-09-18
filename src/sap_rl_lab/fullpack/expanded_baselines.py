"""Transparent thirty-pet heuristics, frozen before learning experiments.

These are diagnostic opponents, not a claim of optimal SAP play. They use only
public current state and legal actions; no future rolls or opponent inspection.
"""

from __future__ import annotations

from .actions import Action, ActionKind
from .events import attack, health


class ExpandedPolicy:
    def __init__(self, focus: str):
        if focus not in {"stats", "summon", "tempo"}:
            raise ValueError("Expanded policy needs stats, summon or tempo")
        self.focus = focus

    def value(self, engine, species):
        spec = engine.catalog.pets[species]
        score = spec.attack + spec.health + 2 * spec.tier
        preferences = {
            "stats": {
                "fish": 6,
                "peacock": 8,
                "crab": 8,
                "giraffe": 14,
                "rabbit": 10,
                "camel": 8,
                "elephant": 7,
                "dodo": 6,
                "worm": 5,
            },
            "summon": {
                "horse": 11,
                "cricket": 8,
                "spider": 10,
                "sheep": 17,
                "dog": 16,
                "ox": 13,
                "rat": -8,
                "hedgehog": -8,
                "badger": -8,
            },
            "tempo": {
                "swan": 8,
                "worm": 9,
                "fish": 7,
                "crab": 9,
                "dolphin": 12,
                "giraffe": 13,
                "rabbit": 8,
                "dodo": 7,
            },
        }
        score += preferences[self.focus].get(species, 0)
        copies = sum(p.spec_id == species for p in engine.state.team)
        if species in {"horse", "rabbit", "giraffe", "swan", "worm"}:
            score -= 8 * copies
        return score

    def rank(self, pet):
        species = pet.spec_id
        if species == "elephant":
            return (-5, -health(pet))
        if species == "camel":
            return (-4, -health(pet))
        if species in {"ant", "flamingo", "cricket", "spider", "sheep", "hedgehog"}:
            return (-3, -attack(pet))
        if species in {"kangaroo", "ox"}:
            return (-2, -attack(pet))
        if species in {"dodo", "giraffe", "snail"}:
            return (1, -attack(pet))
        if species in {"horse", "rabbit", "worm", "swan", "dolphin"}:
            return (2, -attack(pet))
        return (0, -attack(pet) * health(pet))

    def choose(self, engine, rng):
        state = engine.state
        if state.actions_this_turn >= engine.config.max_actions_per_turn - 1:
            return Action(ActionKind.END_TURN)
        legal = engine.legal_actions()
        scored = []
        for action in legal:
            score = None
            if action.kind is ActionKind.BUY_PET:
                item = state.shop[action.source]
                score = 30 + self.value(engine, item.item_id)
            elif action.kind in {ActionKind.MERGE, ActionKind.MERGE_TEAM}:
                target = state.team[action.target]
                score = 55 + self.value(engine, target.spec_id)
                if target.experience in {2, 5}:
                    score += 15
            elif action.kind is ActionKind.BUY_FOOD:
                item, target = state.shop[action.source], state.team[action.target]
                food = engine.catalog.foods[item.item_id]
                if food.effect == "buff" and (attack(target) < 50 or health(target) < 50):
                    score = 12 + (attack(target) + health(target)) / 20
                    if target.spec_id in {"peacock", "camel", "elephant", "dog", "crab"}:
                        score += 4
                    if not item.cost:
                        score += 40
                elif food.effect == "set_perk" and target.perk != food.params["perk"]:
                    if target.perk != "melon":
                        score = 10 + attack(target) / 10
                        if food.params["perk"] == "honey":
                            score = (
                                16 if target.spec_id in {"spider", "badger", "cricket"} else None
                            )
                elif food.effect == "faint_pet":
                    # Only clear, positive shop uses; never pill an arbitrary
                    # carry simply because the food is cheap and legal.
                    species, behind = target.spec_id, len(state.team) - action.target - 1
                    if species == "ant" and len(state.team) > 1:
                        score = 22
                    elif species == "flamingo" and behind >= 2:
                        score = 24
                    elif species == "spider":
                        score = 24
                    elif species == "sheep" and len(state.team) <= 4:
                        score = 20
            if score is not None:
                scored.append((score, action))
        # Upgrade a full team's weakest uninvested member when a substantially
        # more useful pet is actually affordable in the current shop.
        if len(state.team) == engine.config.max_team_size:
            offers = [s for s in state.shop if s and s.kind == "pet" and s.cost <= state.gold]
            if offers:
                best = max(self.value(engine, s.item_id) for s in offers)
                for index, p in enumerate(state.team):
                    investment = attack(p) + health(p) + (p.level - 1) * 8
                    if (
                        p.level == 1
                        and investment < 14
                        and best > self.value(engine, p.spec_id) + 7
                    ):
                        scored.append((38, Action(ActionKind.SELL, index)))
        if scored:
            maximum = max(score for score, _ in scored)
            return rng.choice([action for score, action in scored if score == maximum])
        if state.gold >= 4:
            return Action(ActionKind.ROLL)
        for index in range(len(state.team) - 1):
            if self.rank(state.team[index]) > self.rank(state.team[index + 1]):
                return Action(ActionKind.SWAP, index, index + 1)
        return Action(ActionKind.END_TURN)
