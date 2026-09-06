import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "RL extra is not installed")
class EvaluationTests(unittest.TestCase):
    def test_suite_is_equal_family_average_and_compacts_nested_rows(self):
        from unittest.mock import patch

        from sap_rl_lab.evaluation import compact_evaluation, evaluate_suite

        def fake(policy, **kwargs):
            score = {"a.json": 1.0, "b.json": 0.0, "c.json": 0.2}[kwargs["opponent_league"]]
            return {
                "success_rate": score,
                "mean_return": score,
                "mean_wins": score,
                "truncation_rate": 0.0,
                "mean_episode_actions": 1.0,
                "episode_results": [{"seed": kwargs["seed"]}],
                "failure_examples": [],
            }

        with patch("sap_rl_lab.evaluation.evaluate_policy", side_effect=fake) as evaluator:
            suite = evaluate_suite(
                "greedy", {"a": "a.json", "b": "b.json", "c": "c.json"}, episodes=7, seed=99
            )
        self.assertAlmostEqual(suite["success_rate"], 0.4)
        self.assertEqual(evaluator.call_count, 3)
        self.assertEqual(suite["families"]["a"]["episode_results"], [{"seed": 99}])
        self.assertNotIn("episode_results", compact_evaluation(suite)["families"]["a"])

    def test_batch_size_does_not_change_baseline_episodes(self):
        from sap_rl_lab.evaluation import evaluate_policy

        serial = evaluate_policy("greedy", episodes=12, seed=31, batch_size=1)
        batched = evaluate_policy("greedy", episodes=12, seed=31, batch_size=7)
        self.assertEqual(serial["episode_results"], batched["episode_results"])
        self.assertEqual(serial["battle_counts"], batched["battle_counts"])

    def test_diagnostics_report_exact_failure_reason(self):
        from sap_rl_lab.domain import GameConfig
        from sap_rl_lab.evaluation import evaluate_policy

        result = evaluate_policy(
            "greedy", episodes=3, env_kwargs={"config": GameConfig(max_actions_per_turn=1)}
        )
        self.assertEqual(result["ending_reasons"], {"shop_action_limit": 3})
        self.assertEqual(result["truncation_rate"], 1)
        self.assertEqual([r["seed"] for r in result["episode_results"]], [20000, 20001, 20002])
        self.assertEqual(result["mean_return"], -1)

    def test_paired_comparison_rejects_mismatched_pools_and_seeds(self):
        from sap_rl_lab.evaluation import evaluate_policy
        from sap_rl_lab.experiments import paired_comparison

        a = evaluate_policy("greedy", episodes=2, seed=1)
        b = evaluate_policy("greedy", episodes=2, seed=1)
        identical = paired_comparison(a, b, resamples=20)
        self.assertEqual(identical["success"]["mean_difference"], 0)
        self.assertEqual(identical["return"]["episode_bootstrap_95_percent"], [0, 0])
        b["league_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "same opponent league"):
            paired_comparison(a, b)
        b["episode_results"][0]["seed"] = 99
        with self.assertRaisesRegex(ValueError, "identical episode seed"):
            paired_comparison(a, b)
