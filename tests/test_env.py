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

    def test_vector_autoresets_remain_reproducible_across_episodes(self):
        import numpy as np
        from stable_baselines3.common.vec_env import DummyVecEnv

        from sap_rl_lab.domain import GameConfig
        from sap_rl_lab.env import SapAutoBattlerEnv

        def make():
            return SapAutoBattlerEnv(config=GameConfig(max_actions_per_turn=1))

        a, b = DummyVecEnv([make, make]), DummyVecEnv([make, make])
        a.seed(173)
        b.seed(173)
        a.reset()
        b.reset()
        seeds = []
        for _ in range(8):
            actions = [env.engine.legal_action_ids()[1] for env in a.envs]
            obs_a, rewards_a, dones_a, _ = a.step(actions)
            obs_b, rewards_b, dones_b, _ = b.step(actions)
            self.assertTrue(all(dones_a))
            np.testing.assert_equal(rewards_a, rewards_b)
            np.testing.assert_equal(dones_a, dones_b)
            for key in obs_a:
                np.testing.assert_equal(obs_a[key], obs_b[key])
            self.assertEqual(a.reset_infos, b.reset_infos)
            seeds.append(a.reset_infos[0]["seed"])
            self.assertNotEqual(a.reset_infos[0]["seed"], a.reset_infos[1]["seed"])
        self.assertEqual(len(seeds), len(set(seeds)))
        a.close()
        b.close()

    def test_turn_limit_forfeit_retains_last_battle_reward(self):
        from sap_rl_lab.actions import ActionKind
        from sap_rl_lab.domain import GameConfig
        from sap_rl_lab.env import SapAutoBattlerEnv

        env = SapAutoBattlerEnv(
            config=GameConfig(max_turns=1),
            opponent_provider=lambda *args: [],
            action_cost=0.005,
            forfeit_on_limit=True,
        )
        env.reset(seed=1)
        buy = next(a for a in env.engine.legal_actions() if a.kind is ActionKind.BUY_PET)
        env.step(env.engine.codec.encode(buy))
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(info["reason"], "turn_limit")
        self.assertEqual(info["game_reward"], 1)
        self.assertAlmostEqual(reward, 1 - 5 - 0.005)
        self.assertTrue(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
