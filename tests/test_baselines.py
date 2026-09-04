import unittest

from sap_rl_lab.baselines import SpendGoldPolicy, play_episode
from sap_rl_lab.engine import AutoBattler


class BaselineTests(unittest.TestCase):
    def test_scripted_policy_finishes_without_truncating(self):
        summary = play_episode(AutoBattler(), SpendGoldPolicy(), seed=12)
        self.assertFalse(summary.truncated)
        self.assertGreater(summary.turns, 0)


if __name__ == "__main__":
    unittest.main()
