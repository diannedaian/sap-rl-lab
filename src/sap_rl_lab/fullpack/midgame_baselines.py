"""Fixed forty-pet scripts; deliberately modest opponents, not optimized agents.

Keep the older ExpandedPolicy and its replay behavior unchanged. These preferences
are specified before learning; no candidate test results feed back into them.
"""

from .expanded_baselines import ExpandedPolicy


class MidgamePolicy(ExpandedPolicy):
    def value(self, engine, species):
        score = super().value(engine, species)
        priorities = {
            "stats": {
                "bison": 16,
                "penguin": 16,
                "hippo": 12,
                "squirrel": 8,
                "blowfish": 8,
                "turtle": 6,
            },
            "summon": {"deer": 22, "whale": 16, "parrot": 12, "turtle": 10},
            "tempo": {
                "skunk": 16,
                "hippo": 12,
                "blowfish": 12,
                "deer": 10,
                "squirrel": 9,
                "penguin": 7,
            },
        }
        score += priorities[self.focus].get(species, 0)
        team = engine.state.team
        if species == "bison":
            if not any(p.level == 3 for p in team):
                score -= 12
            score -= 20 * sum(p.spec_id == "bison" for p in team)
        if species == "penguin" and sum(p.level >= 2 for p in team) < 2:
            score -= 6
        if species in {"whale", "parrot"}:
            score -= 10 * sum(p.spec_id == species for p in team)
        return score

    def rank(self, pet):
        if pet.spec_id in {"deer", "turtle"}:
            return (-3, -pet.attack)
        if pet.spec_id == "whale":
            return (-2.5, -pet.attack)
        if pet.spec_id == "parrot":
            return (-1, -pet.attack)
        if pet.spec_id == "blowfish":
            return (-4, -pet.health)
        if pet.spec_id in {"penguin", "squirrel", "skunk"}:
            return (2, -pet.attack)
        return super().rank(pet)
