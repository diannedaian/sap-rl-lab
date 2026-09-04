import unittest

from sap_rl_lab.catalog import load_catalog


class CatalogTests(unittest.TestCase):
    def test_catalog_is_versioned_and_references_are_valid(self):
        catalog = load_catalog()
        self.assertEqual(catalog.game_version, "0.46")
        self.assertEqual(len(catalog.rollable_pet_ids), 8)
        self.assertIn("pigeon", catalog.rollable_pet_ids)
        self.assertIn("zombie_cricket", catalog.pets)

    def test_only_unlocked_rollable_items_enter_pools(self):
        catalog = load_catalog()
        self.assertNotIn("zombie_cricket", catalog.pet_ids_through_tier(6))
        self.assertNotIn("bread_crumbs", catalog.food_ids_through_tier(6))


if __name__ == "__main__":
    unittest.main()
