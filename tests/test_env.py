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

    def test_training_forfeit_is_absorbing_and_pays_remaining_lives(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        from sap_rl_lab.domain import GameConfig
        from sap_rl_lab.env import SapAutoBattlerEnv

        vec = DummyVecEnv(
            [
                lambda: SapAutoBattlerEnv(
                    config=GameConfig(max_actions_per_turn=1),
                    action_cost=0.005,
                    forfeit_on_limit=True,
                )
            ]
        )
        vec.seed(1)
        vec.reset()
        action = next(i for i in vec.envs[0].engine.legal_action_ids() if i > 1)
        _, rewards, dones, infos = vec.step([action])
        self.assertAlmostEqual(float(rewards[0]), -5.005, places=5)
        self.assertTrue(dones[0])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertEqual(infos[0]["reason"], "shop_action_limit")
        self.assertEqual(infos[0]["game_reward"], -1)
        vec.close()

    def test_shaping_preserves_evaluation_rewards_and_legal_actions(self):
        from sap_rl_lab.env import SapAutoBattlerEnv

        raw = SapAutoBattlerEnv()
        shaped = SapAutoBattlerEnv(action_cost=0.005, forfeit_on_limit=True)
        raw.reset(seed=33)
        shaped.reset(seed=33)
        for _ in range(5):
            self.assertEqual(raw.engine.action_mask(), shaped.engine.action_mask())
            # Buying has an opportunity cost, but the game itself is unchanged.
            action = raw.engine.legal_action_ids()[1]
            _, raw_reward, _, _, _ = raw.step(action)
            _, reward, _, _, info = shaped.step(action)
            self.assertAlmostEqual(reward, raw_reward - 0.005)
            self.assertEqual(info["game_reward"], raw_reward)
            self.assertEqual(raw.engine.state.to_dict(), shaped.engine.state.to_dict())


if __name__ == "__main__":
    unittest.main()
