import random
import tempfile
import unittest
from pathlib import Path

from sap_rl_lab.catalog import load_catalog
from sap_rl_lab.domain import GameConfig, Pet
from sap_rl_lab.opponents import SnapshotLeague, build_scripted_league, build_spend_gold_league


class OpponentLeagueTests(unittest.TestCase):
    def test_mixture_weights_families_not_snapshot_counts(self):
        from collections import Counter

        from sap_rl_lab.opponents import OpponentMixture

        small, large = SnapshotLeague(), SnapshotLeague()
        small.add(1, [Pet("ant", 2, 2)])
        for _ in range(100):
            large.add(1, [Pet("cricket", 1, 3)])
        mixture = OpponentMixture([(1.0, small), (1.0, large)])
        rng = random.Random(3)
        counts = Counter(mixture(1, rng, self.catalog, self.config)[0].spec_id for _ in range(2000))
        self.assertTrue(900 < counts["ant"] < 1100)
        self.assertTrue(900 < counts["cricket"] < 1100)

    def setUp(self):
        self.catalog = load_catalog()
        self.config = GameConfig()

    def test_uses_nearest_earlier_turn_and_returns_fresh_pets(self):
        league = SnapshotLeague()
        source = [Pet("ant", 7, 8)]
        league.add(3, source, label="checkpoint-1")

        first = league(4, random.Random(1), self.catalog, self.config)
        second = league(4, random.Random(1), self.catalog, self.config)
        self.assertEqual((first[0].spec_id, first[0].attack, first[0].health), ("ant", 7, 8))
        self.assertIsNot(first[0], second[0])
        first[0].attack = 99
        self.assertEqual(second[0].attack, 7)

    def test_json_round_trip(self):
        league = SnapshotLeague()
        league.add(2, [Pet("cricket", 3, 4, experience=3, perk="honey")], "scripted")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "league.json"
            league.save(path)
            restored = SnapshotLeague.load(path)
            team = restored(2, random.Random(2), self.catalog, self.config)
        self.assertEqual(len(restored), 1)
        self.assertIsNone(restored.catalog_id)
        self.assertEqual(team[0], Pet("cricket", 3, 4, experience=3, perk="honey"))

    def test_build_scripted_league_has_round_buckets_and_catalog(self):
        league = build_spend_gold_league(episodes=2, seed=10)
        self.assertEqual(league.catalog_id, self.catalog.catalog_id)
        self.assertGreaterEqual(len(league), 10)
        first_round = league(1, random.Random(3), self.catalog, self.config)
        self.assertTrue(first_round)

    def test_snapshot_preserves_temporary_buffs_and_battle(self):
        from sap_rl_lab.engine import resolve_battle

        team = [Pet("cricket", 1, 3, temporary_attack=3, temporary_health=2), Pet("horse", 2, 1)]
        league = SnapshotLeague(catalog_id=self.catalog.catalog_id)
        league.add(1, team)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            league.save(path)
            restored = SnapshotLeague.load(path)(1, random.Random(0), self.catalog, self.config)
        self.assertEqual(team, restored)
        enemy = [Pet("fish", 5, 6), Pet("ant", 3, 3)]
        self.assertEqual(
            resolve_battle(team, enemy, self.catalog, random.Random(7)),
            resolve_battle(restored, enemy, self.catalog, random.Random(7)),
        )

    def test_legacy_snapshot_loads_without_temporary_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(
                '{"schema_version":1,"snapshots":[{"turn":1,"label":"old",'
                '"pets":[{"spec_id":"ant","attack":2,"health":2}]}]}'
            )
            restored = SnapshotLeague.load(path)(1, random.Random(0), self.catalog, self.config)
        self.assertEqual(restored[0].temporary_attack, 0)

    def test_new_families_are_legal_reproducible_and_distinct(self):
        from collections import Counter

        from sap_rl_lab.baselines import FocusedPolicy
        from sap_rl_lab.engine import AutoBattler

        profiles = {}
        for name in ("stats", "summon"):
            for seed in range(12):
                engine = AutoBattler()
                engine.reset(seed)
                policy = FocusedPolicy(name)
                rng = random.Random(seed)
                while not (engine.state.terminated or engine.state.truncated):
                    action = policy.choose(engine, rng)
                    self.assertIn(action, engine.legal_actions())
                    engine.step(action)
                    self.assertGreaterEqual(engine.state.gold, 0)
                    self.assertLessEqual(len(engine.state.team), 5)
                self.assertFalse(engine.state.truncated)
            first = build_scripted_league(name, 12, seed=91)
            second = build_scripted_league(name, 12, seed=91)
            self.assertEqual(first._by_turn, second._by_turn)
            profiles[name] = Counter(
                p.spec_id for teams in first._by_turn.values() for team in teams for p in team.pets
            )
        self.assertGreater(profiles["summon"]["horse"], profiles["stats"]["horse"])
        self.assertGreater(profiles["summon"]["cricket"], profiles["stats"]["cricket"])


if __name__ == "__main__":
    unittest.main()
