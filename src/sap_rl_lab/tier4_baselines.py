"""Fixed v6 teacher preferences. Heuristics, not expert or optimal actions."""

from .actions import Action, ActionKind
from .midgame_baselines import MidgamePolicy


class Tier4Policy(MidgamePolicy):
    def value(self, engine, species):
        preferences = {
            "stats": {"monkey": 22, "seal": 18, "cow": 16, "rhino": 10},
            "summon": {"turkey": 30, "shark": 24, "rooster": 22, "cow": 10},
            "tempo": {"crocodile": 22, "scorpion": 18, "rhino": 16, "cow": 12},
        }
        return super().value(engine, species) + preferences[self.focus].get(species, 0)

    def rank(self, pet):
        if pet.spec_id in {"scorpion", "rooster"}:
            return (-3.5, -pet.attack)
        if pet.spec_id in {"turkey", "shark", "monkey", "seal", "cow"}:
            return (3, -pet.attack)
        return super().rank(pet)

    def choose(self, engine, rng):
        state = engine.state
        if state.actions_this_turn >= engine.config.max_actions_per_turn - 1:
            return Action(ActionKind.END_TURN)
        legal = engine.legal_actions()
        canned = [
            a
            for a in legal
            if a.kind is ActionKind.BUY_FOOD and state.shop[a.source].item_id == "canned_food"
        ]
        if not state.team and canned:
            buys = [a for a in legal if a.kind is ActionKind.BUY_PET]
            return rng.choice(buys or canned)
        if canned and state.gold >= 6 and state.turn <= 12 and state.shop_attack_bonus < 3:
            return canned[0]
        chosen = super().choose(engine, rng)
        if chosen.kind is ActionKind.BUY_FOOD:
            target = state.team[chosen.target]
            food = engine.catalog.foods[state.shop[chosen.source].item_id]
            if target.perk == "peanut" and food.effect == "set_perk":
                # Do not teach throwing away a Scorpion's central ability.
                return Action(ActionKind.ROLL) if state.gold >= 4 else Action(ActionKind.END_TURN)
        return chosen
