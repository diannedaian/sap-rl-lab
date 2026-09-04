import copy
import random
import unittest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import load_catalog
from sap_rl_lab.domain import BattleOutcome, GameConfig, Pet, ShopItem
from sap_rl_lab.engine import AutoBattler, InvalidAction, resolve_battle


def empty_opponent(turn, rng, catalog, config):
    return []


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_reset_is_reproducible(self):
        first = AutoBattler(self.catalog)
        second = AutoBattler(self.catalog)
        self.assertEqual(first.reset(seed=41).to_dict(), second.reset(seed=41).to_dict())

    def test_invalid_action_does_not_mutate_state(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=2)
        before = copy.deepcopy(engine.state.to_dict())
        with self.assertRaises(InvalidAction):
            engine.step(Action(ActionKind.SELL, source=0))
        self.assertEqual(before, engine.state.to_dict())

    def test_shop_actions_have_zero_reward_and_battle_reward_is_incremental(self):
        engine = AutoBattler(self.catalog, opponent_provider=empty_opponent)
        engine.reset(seed=2)
        buy = next(a for a in engine.legal_actions() if a.kind is ActionKind.BUY_PET)
        self.assertEqual(engine.step(buy).reward, 0.0)
        first_win = engine.step(Action(ActionKind.END_TURN))
        self.assertEqual(first_win.reward, 1.0)
        roll = engine.step(Action(ActionKind.ROLL))
        self.assertEqual(roll.reward, 0.0)

    def test_buying_preserves_shop_slot_identity(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=3)
        original = list(engine.state.shop)
        engine.step(Action(ActionKind.BUY_PET, source=1))
        self.assertEqual(engine.state.shop[0], original[0])
        self.assertIsNone(engine.state.shop[1])
        self.assertEqual(engine.state.shop[2], original[2])

    def test_mask_matches_legal_actions(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=5)
        masked = {index for index, allowed in enumerate(engine.action_mask()) if allowed}
        self.assertEqual(masked, set(engine.legal_action_ids()))

    def test_pig_sell_returns_base_and_bonus_gold(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=1)
        engine.state.team = [Pet("pig", 4, 1)]
        engine.state.gold = 0
        engine.step(Action(ActionKind.SELL, source=0))
        self.assertEqual(engine.state.gold, 2)

    def test_pigeon_stocks_free_bread_crumbs(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=1)
        engine.state.team = [Pet("pigeon", 3, 2)]
        engine.state.shop[2] = None
        engine.step(Action(ActionKind.SELL, source=0))
        self.assertEqual(engine.state.shop[2], ShopItem("food", "bread_crumbs", 0))

    def test_horse_buffs_bought_pet_only_for_next_battle(self):
        engine = AutoBattler(self.catalog)
        engine.reset(seed=1)
        engine.state.team = [Pet("horse", 2, 1)]
        engine.state.shop[0] = ShopItem("pet", "cricket", 3)
        engine.step(Action(ActionKind.BUY_PET, source=0))
        cricket = engine.state.team[1]
        self.assertEqual((cricket.attack, cricket.temporary_attack), (1, 1))

    def test_empty_team_draws_empty_team(self):
        result = resolve_battle([], [], self.catalog, random.Random(1))
        self.assertEqual(result.outcome, BattleOutcome.DRAW)

    def test_cricket_summon_can_turn_loss_into_win(self):
        result = resolve_battle(
            [Pet("cricket", 1, 1)],
            [Pet("pig", 1, 1)],
            self.catalog,
            random.Random(1),
        )
        self.assertEqual(result.outcome, BattleOutcome.WIN)
        self.assertTrue(any("summons" in event for event in result.trace))

    def test_shop_action_limit_is_a_true_truncation(self):
        config = GameConfig(max_actions_per_turn=2)
        engine = AutoBattler(self.catalog, config)
        engine.reset(seed=1)
        first = next(a for a in engine.legal_actions() if a.kind is ActionKind.FREEZE)
        self.assertFalse(engine.step(first).truncated)
        second = next(a for a in engine.legal_actions() if a.kind is ActionKind.FREEZE)
        result = engine.step(second)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)


if __name__ == "__main__":
    unittest.main()
