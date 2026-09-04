import random
import tempfile
import unittest
from pathlib import Path

from sap_rl_lab.catalog import load_catalog
from sap_rl_lab.domain import GameConfig, Pet
from sap_rl_lab.opponents import SnapshotLeague, build_spend_gold_league


class OpponentLeagueTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
