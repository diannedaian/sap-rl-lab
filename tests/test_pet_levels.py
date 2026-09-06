"""All eight implemented pets: level scaling and trigger-boundary regressions.

Fish/Ant/Otter values are supported by Team Wood Games' 0.27 announcement.
The remaining tests verify this catalog's contract, not unobserved live-game parity.
"""

import random
import unittest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import load_catalog
from sap_rl_lab.domain import Pet, ShopItem
from sap_rl_lab.engine import AutoBattler, resolve_battle
from sap_rl_lab.replay import ReplayRecorder, verify_replay

XP = {1: 1, 2: 3, 3: 6}


class PetLevelTests(unittest.TestCase):
    def prepared(self, catalog=None):
        engine = AutoBattler(catalog=catalog)
        engine.reset(17)
        engine.state.shop = [None] * 5
        return engine

    def test_fish_both_level_boundaries_use_departing_level(self):
        for experience, buff in ((2, 1), (5, 2)):
            with self.subTest(experience=experience):
                engine = self.prepared()
                engine.state.team = [Pet("fish", 4, 6, experience=experience)] + [
                    Pet("pig", 4, 1) for _ in range(3)
                ]
                engine.state.shop[0] = ShopItem("pet", "fish", 3)
                engine.step(Action(ActionKind.MERGE, 0, 0))
                changed = [(p.attack, p.health) for p in engine.state.team[1:]]
                self.assertEqual(changed.count((4 + buff, 1 + buff)), 2)
                self.assertEqual(changed.count((4, 1)), 1)
                self.assertEqual((engine.state.team[0].attack, engine.state.team[0].health), (5, 7))

    def test_fish_non_level_merge_does_not_buff(self):
        for experience in (1, 3, 4):
            engine = self.prepared()
            engine.state.team = [Pet("fish", 2, 3, experience=experience), Pet("pig", 4, 1)]
            engine.state.shop[0] = ShopItem("pet", "fish", 3)
            engine.step(Action(ActionKind.MERGE, 0, 0))
            self.assertEqual(engine.state.team[1], Pet("pig", 4, 1))

    def test_fish_fewer_friends_and_stat_cap(self):
        engine = self.prepared()
        engine.state.team = [Pet("fish", 4, 5, experience=5), Pet("pig", 49, 49)]
        engine.state.shop[0] = ShopItem("pet", "fish", 3)
        engine.step(Action(ActionKind.MERGE, 0, 0))
        self.assertEqual((engine.state.team[1].attack, engine.state.team[1].health), (50, 50))

    def test_fish_only_level_up_pet_and_invalid_levels_fail(self):
        catalog = load_catalog()
        affected = {
            p.id for p in catalog.pets.values() if any(a.trigger == "level_up" for a in p.abilities)
        }
        self.assertEqual(affected, {"fish"})
        for level in (0, -1, 4):
            with self.assertRaises(ValueError):
                catalog.pets["fish"].abilities[0].at_level("attack", level)

    def test_otter_buy_scaling_keeps_current_level(self):
        for level in (1, 2, 3):
            engine = self.prepared()
            otter = Pet("otter", 1, 3, experience=XP[level])
            engine.state.team = [otter] + [Pet("fish", 2, 3) for _ in range(4)]
            engine._run_shop_trigger(otter, "buy", subject=otter)
            self.assertEqual(sum(p.health == 4 for p in engine.state.team[1:]), level)
            self.assertTrue(all(p.attack == 2 for p in engine.state.team[1:]))
            self.assertEqual(otter.health, 3)

    def test_otter_merge_level_crossing_is_not_changed_to_old_level(self):
        engine = self.prepared()
        engine.state.team = [Pet("otter", 1, 3, experience=2)] + [
            Pet("fish", 2, 3) for _ in range(3)
        ]
        engine.state.shop[0] = ShopItem("pet", "otter", 3)
        engine.step(Action(ActionKind.MERGE, 0, 0))
        self.assertEqual(sum(p.health == 4 for p in engine.state.team[1:]), 2)

    def test_pig_and_pigeon_sell_at_every_level(self):
        for level in (1, 2, 3):
            for name in ("pig", "pigeon"):
                engine = self.prepared()
                engine.state.gold = 0
                engine.state.team = [Pet(name, 4, 3, experience=XP[level])]
                engine.step(Action(ActionKind.SELL, 0))
                self.assertEqual(engine.state.gold, level * (2 if name == "pig" else 1))
                if name == "pigeon":
                    self.assertEqual(sum(p is not None for p in engine.state.shop), level)
                    self.assertTrue(
                        all(
                            p.item_id == "bread_crumbs" and p.cost == 0
                            for p in engine.state.shop
                            if p is not None
                        )
                    )

    def test_horse_shop_buff_at_every_level_and_does_not_persist(self):
        for level in (1, 2, 3):
            engine = self.prepared()
            engine.opponent_provider = lambda *args: []
            engine.state.team = [Pet("horse", 2, 1, experience=XP[level])]
            engine.state.shop[0] = ShopItem("pet", "cricket", 3)
            engine.step(Action(ActionKind.BUY_PET, 0))
            target = engine.state.team[1]
            self.assertEqual((target.attack, target.temporary_attack), (1, level))
            engine.step(Action(ActionKind.END_TURN))
            self.assertEqual((target.attack, target.temporary_attack), (1, 0))

    def test_ant_cricket_and_mosquito_battle_scaling(self):
        catalog = load_catalog()
        for level in (1, 2, 3):
            ant = Pet("ant", 0, 0, experience=XP[level])
            # Zero-health setup triggers the actual faint resolver before attacks.
            result = resolve_battle(
                [ant, Pet("fish", 2, 3)], [Pet("pig", 0, 1)], catalog, random.Random(9)
            )
            self.assertIn(f"attack for {2 + level}/0", " ".join(result.trace))
            result = resolve_battle(
                [Pet("cricket", 0, 0, experience=XP[level])],
                [Pet("pig", 0, 1)],
                catalog,
                random.Random(9),
            )
            self.assertIn(f"attack for {level}/0", " ".join(result.trace))
            result = resolve_battle(
                [Pet("mosquito", 1, 1, experience=XP[level])],
                [Pet("fish", 0, 10) for _ in range(4)],
                catalog,
                random.Random(9),
            )
            self.assertEqual(sum("deals 1" in s for s in result.trace), level)

    def test_historical_rules_remain_explicitly_replayable(self):
        legacy = load_catalog("turtle_v0_46_tier1.json")
        self.assertEqual(legacy.rules_version, 1)
        self.assertEqual(load_catalog().rules_version, 2)
        engine = self.prepared(legacy)
        engine.state.team = [Pet("fish", 2, 3, experience=5), Pet("pig", 4, 1)]
        engine.state.shop[0] = ShopItem("pet", "fish", 3)
        engine.step(Action(ActionKind.MERGE, 0, 0))
        self.assertEqual(engine.state.team[1], Pet("pig", 4, 1))
        recorder = ReplayRecorder(AutoBattler(legacy), seed=7)
        recorder.step(0)
        verify_replay(recorder.finish())


if __name__ == "__main__":
    unittest.main()
