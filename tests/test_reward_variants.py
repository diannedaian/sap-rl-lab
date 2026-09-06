import importlib.util
import tempfile
import unittest
from pathlib import Path

from sap_rl_lab.training import TrainingConfig


class RewardConfigTests(unittest.TestCase):
    def test_reward_coefficients_are_finite_and_nonnegative(self):
        for field in ("swap_cost", "success_bonus_max", "success_action_cost"):
            for value in (-0.1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    TrainingConfig(**{field: value}).validate()

    def test_success_bonus_requires_observable_history(self):
        with self.assertRaisesRegex(ValueError, "observed count"):
            TrainingConfig(success_bonus_max=1, success_action_cost=0.005).validate()
        TrainingConfig(
            success_bonus_max=1, success_action_cost=0.005, observe_episode_actions=True
        ).validate()


@unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "RL extra is not installed")
class RewardVariantTests(unittest.TestCase):
    def make(self, **kwargs):
        from sap_rl_lab.env import SapAutoBattlerEnv

        env = SapAutoBattlerEnv(**kwargs)
        env.reset(seed=37)
        return env

    def test_swap_cost_only_charges_swaps_and_keeps_game_unchanged(self):
        from sap_rl_lab.actions import Action, ActionKind
        from sap_rl_lab.domain import Pet

        raw, shaped = self.make(), self.make(swap_cost=0.005)
        for env in (raw, shaped):
            env.engine.state.team = [Pet("fish", 2, 3), Pet("fish", 2, 3)]
        for action in (Action(ActionKind.SWAP, 0, 1), Action(ActionKind.ROLL)):
            action_id = raw.engine.codec.encode(action)
            a, b = raw.step(action_id), shaped.step(action_id)
            penalty = 0.005 if action.kind is ActionKind.SWAP else 0
            self.assertAlmostEqual(a[1] - b[1], penalty)
            self.assertEqual(b[4]["swap_penalty"], penalty)
            self.assertEqual(raw.engine.state.to_dict(), shaped.engine.state.to_dict())
            self.assertEqual(raw.engine.action_mask(), shaped.engine.action_mask())

    def test_success_bonus_paid_only_once_at_ten_wins_and_counts_all_decisions(self):
        from sap_rl_lab.actions import Action, ActionKind
        from sap_rl_lab.domain import Pet

        env = self.make(
            success_bonus_max=1,
            success_action_cost=0.005,
            observe_episode_actions=True,
            opponent_provider=lambda *args: [],
        )
        env.engine.state.team = [Pet("fish", 2, 3)]
        env.engine.state.wins = 8
        env.step(env.engine.codec.encode(Action(ActionKind.ROLL)))
        _, reward, terminated, _, info = env.step(0)
        self.assertEqual(reward, 1)
        self.assertFalse(terminated)
        self.assertEqual(info["success_efficiency_bonus"], 0)
        self.assertEqual(env.episode_actions, 2)
        _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["episode_actions"], 3)
        self.assertAlmostEqual(info["success_efficiency_bonus"], 0.985)
        self.assertAlmostEqual(reward, 1.985)
        self.assertEqual(info["game_reward"], 1)
        self.assertEqual(env.step(0)[4].get("success_efficiency_bonus", 0), 0)
        observation, _ = env.reset(seed=37)
        self.assertEqual(env.episode_actions, 0)
        self.assertEqual(observation["global"][-1], 0)

    def test_long_success_bonus_has_zero_floor(self):
        from sap_rl_lab.domain import Pet

        env = self.make(
            success_bonus_max=1,
            success_action_cost=0.005,
            observe_episode_actions=True,
            opponent_provider=lambda *args: [],
        )
        env.engine.state.team = [Pet("fish", 2, 3)]
        env.engine.state.wins = 9
        env.episode_actions = 210
        self.assertEqual(env.step(0)[1], 1)

    def test_failures_and_cutoffs_never_receive_success_bonus(self):
        from sap_rl_lab.domain import GameConfig, Pet

        for kind in ("loss", "shop_limit", "turn_limit", "invalid"):
            config = (
                GameConfig(max_actions_per_turn=1)
                if kind == "shop_limit"
                else (GameConfig(max_turns=1) if kind == "turn_limit" else GameConfig())
            )
            env = self.make(
                config=config,
                success_bonus_max=1,
                success_action_cost=0.005,
                observe_episode_actions=True,
            )
            env.engine.state.lives = 1
            if kind == "turn_limit":
                env.engine.opponent_provider = lambda *args: []
                env.engine.state.team = [Pet("fish", 2, 3)]
            action = 1 if kind == "shop_limit" else 0
            if kind == "invalid":
                action = next(i for i, legal in enumerate(env.engine.action_mask()) if not legal)
            _, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(terminated or truncated)
            self.assertEqual(reward, 1 if kind == "turn_limit" else -1)
            self.assertEqual(info.get("success_efficiency_bonus", 0), 0)

    def test_observation_has_count_and_old_observation_stays_compatible(self):
        from gymnasium.utils.env_checker import check_env

        env = self.make(observe_episode_actions=True)
        check_env(env, skip_render_check=True)
        env.reset(seed=10)
        observation, *_ = env.step(1)
        self.assertEqual(observation["global"].shape, (9,))
        self.assertAlmostEqual(float(observation["global"][-1]), 1 / 900)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(self.make().observation_space["global"].shape, (8,))

    def test_zero_input_migration_preserves_values_and_probabilities(self):
        import numpy as np
        import torch
        from sb3_contrib import MaskablePPO

        from sap_rl_lab.evaluation import evaluate_policy
        from sap_rl_lab.training import copy_initial_policy

        old_env, new_env = self.make(), self.make(observe_episode_actions=True)
        old = MaskablePPO("MultiInputPolicy", old_env, seed=7, n_steps=16, batch_size=16)
        new = MaskablePPO("MultiInputPolicy", new_env, seed=9, n_steps=16, batch_size=16)
        self.assertEqual(copy_initial_policy(new, old), "zero_weight_episode_action_input")
        for seed in range(4):
            old_obs, _ = old_env.reset(seed=seed)
            new_obs, _ = new_env.reset(seed=seed)
            new_obs["global"][-1] = 0.73
            with torch.no_grad():
                a, _ = old.policy.obs_to_tensor(old_obs)
                b, _ = new.policy.obs_to_tensor(new_obs)
                pa = old.policy.get_distribution(a, action_masks=old_env.action_masks())
                pb = new.policy.get_distribution(b, action_masks=new_env.action_masks())
                torch.testing.assert_close(pa.distribution.probs, pb.distribution.probs)
                torch.testing.assert_close(
                    old.policy.predict_values(a), new.policy.predict_values(b)
                )
            np.testing.assert_equal(
                old.predict(old_obs, action_masks=old_env.action_masks(), deterministic=True)[0],
                new.predict(new_obs, action_masks=new_env.action_masks(), deterministic=True)[0],
            )
        a = evaluate_policy(old, episodes=5, seed=51)
        b = evaluate_policy(new, episodes=5, seed=51)
        self.assertEqual(a["episode_results"], b["episode_results"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            new.save(path)
            loaded = MaskablePPO.load(path)
            self.assertEqual(
                b["episode_results"],
                evaluate_policy(loaded, episodes=5, seed=51)["episode_results"],
            )

    def test_two_reward_arms_smoke_training_share_initial_weights(self):
        import json

        from sap_rl_lab.training import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = train(
                TrainingConfig(
                    timesteps=16,
                    environments=1,
                    rollout_steps=16,
                    batch_size=16,
                    output_dir=str(root / "initial"),
                    device="cpu",
                )
            )
            initial_hash = ""
            for name, setting in (
                ("swap", {"swap_cost": 0.005}),
                ("bonus", {"success_bonus_max": 1, "success_action_cost": 0.005}),
            ):
                train(
                    TrainingConfig(
                        timesteps=16,
                        environments=1,
                        rollout_steps=16,
                        batch_size=16,
                        output_dir=str(root / name),
                        initialize_from=str(initial),
                        observe_episode_actions=True,
                        validation_episodes=2,
                        expected_initial_policy_sha256=initial_hash,
                        device="cpu",
                        **setting,
                    )
                )
                manifest = json.loads((root / name / "run_manifest.json").read_text())
                initial_hash = manifest["initial_policy_sha256"]
                self.assertEqual(manifest["input_migration"], "zero_weight_episode_action_input")


if __name__ == "__main__":
    unittest.main()
