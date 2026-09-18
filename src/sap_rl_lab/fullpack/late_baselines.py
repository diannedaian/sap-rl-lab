"""Readable full-pack teachers: numeric scaling, summons, and immediate tempo.

No lookahead, opponent access, or candidate-test tuning. These are not experts.
"""

from .actions import Action, ActionKind
from .tier4_baselines import Tier4Policy


class FullPackPolicy(Tier4Policy):
    def value(self, engine, species):
        preferences = {
            "stats": {"cat": 25, "dragon": 24, "gorilla": 20, "boar": 20},
            "summon": {"fly": 35, "tiger": 28, "snake": 27, "mammoth": 18},
            "tempo": {"leopard": 32, "tiger": 28, "boar": 24, "gorilla": 22, "piranha": 14},
        }
        score = super().value(engine, species) + preferences[self.focus].get(species, 0)
        if species in {"cat", "dragon", "fly", "tiger"}:
            score -= 15 * sum(p.spec_id == species for p in engine.state.team)
        return score

    def rank(self, pet):
        if pet.spec_id == "mammoth":
            return (-3.5, -pet.attack)
        if pet.spec_id in {"boar", "gorilla", "piranha"}:
            return (-2, -pet.health)
        if pet.spec_id in {"snake", "leopard"}:
            return (0.5, -pet.attack)
        if pet.spec_id == "tiger":
            return (1, -pet.attack)
        if pet.spec_id == "fly":
            return (2, -pet.attack)
        if pet.spec_id in {"cat", "dragon"}:
            return (3, -pet.attack)
        return super().rank(pet)

    def choose(self, engine, rng):
        state = engine.state
        if state.actions_this_turn >= engine.config.max_actions_per_turn - 1:
            return Action(ActionKind.END_TURN)
        chosen = super().choose(engine, rng)
        if chosen.kind in {ActionKind.BUY_PET, ActionKind.MERGE, ActionKind.MERGE_TEAM}:
            return chosen
        # Demonstrate a late-game pivot only for a clearly replaceable early pet.
        offers = [
            s
            for s in state.shop
            if s
            and s.kind == "pet"
            and s.cost <= state.gold
            and engine.catalog.pets[s.item_id].tier >= 5
        ]
        if len(state.team) == engine.config.max_team_size and offers:
            best = max(self.value(engine, s.item_id) for s in offers)
            for i, p in enumerate(state.team):
                if (
                    p.level == 1
                    and p.attack + p.health < 25
                    and engine.catalog.pets[p.spec_id].tier <= 3
                    and best > self.value(engine, p.spec_id) + 20
                ):
                    return Action(ActionKind.SELL, i)
        alternatives = []
        for action in engine.legal_actions():
            if action.kind is not ActionKind.BUY_FOOD or not state.team:
                continue
            target = state.team[action.target]
            food = engine.catalog.foods[state.shop[action.source].item_id]
            if food.effect == "gain_experience" and target.experience in {2, 5}:
                alternatives.append((40 + self.value(engine, target.spec_id), action))
            elif food.effect == "buff_random_team" and len(state.team) >= food.params["count"]:
                alternatives.append((20, action))
            elif food.effect == "set_perk" and target.perk not in {"peanut", "melon", "coconut"}:
                if (
                    food.params["perk"] == "mushroom"
                    and target.spec_id in {"mammoth", "snake", "fly", "turkey", "tiger", "scorpion"}
                    and target.perk != "mushroom"
                ):
                    alternatives.append((30, action))
        if alternatives:
            maximum = max(score for score, _ in alternatives)
            return rng.choice([action for score, action in alternatives if score == maximum])
        return chosen
