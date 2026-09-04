import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "gymnasium extra is not installed")
class EnvironmentTests(unittest.TestCase):
    def test_observation_and_step_match_declared_spaces(self):
        from gymnasium.utils.env_checker import check_env

        from sap_rl_lab.env import SapAutoBattlerEnv

        env = SapAutoBattlerEnv()
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(len(env.action_masks()), env.action_space.n)


if __name__ == "__main__":
    unittest.main()
