import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sap_rl_lab.training import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def test_cluster_shape_is_valid(self):
        TrainingConfig(
            timesteps=500_000,
            environments=8,
            rollout_steps=512,
            batch_size=256,
            vector_backend="subproc",
            device="cuda",
        ).validate()

    def test_partial_minibatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            TrainingConfig(environments=3, rollout_steps=10, batch_size=16).validate()

    def test_unknown_vector_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vector_backend"):
            TrainingConfig(vector_backend="threads").validate()

    def test_duplicated_training_pool_is_rejected_as_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "train.json", root / "renamed_copy.json"
            first.write_text('{"snapshots": []}')
            second.write_bytes(first.read_bytes())
            with self.assertRaisesRegex(ValueError, "validation league must differ"):
                TrainingConfig(opponent_league=str(first), validation_league=str(second)).validate()

    def test_mixture_validation_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a.json", root / "b.json"
            a.write_text("a")
            b.write_text("b")
            with self.assertRaisesRegex(ValueError, "validation league must differ"):
                TrainingConfig(
                    opponent_leagues=(str(a), str(b)), validation_leagues={"same": str(a)}
                ).validate()

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "RL extra is not installed")
    def test_fresh_pair_initialization_and_macro_validation(self):
        from sb3_contrib import MaskablePPO

        from sap_rl_lab.evaluation import evaluate_suite
        from sap_rl_lab.opponents import build_scripted_league
        from sap_rl_lab.training import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pools = {}
            for i, name in enumerate(("greedy", "stats", "summon")):
                path = root / f"{name}.json"
                build_scripted_league(name, 2, seed=1100 + i * 100).save(path)
                pools[name] = str(path)
            validation = {"stats": pools["stats"], "summon": pools["summon"]}
            extra = root / "extra_training.json"
            build_scripted_league("stats", 2, seed=9000).save(extra)
            configs = dict(
                timesteps=32,
                environments=2,
                rollout_steps=16,
                batch_size=16,
                device="cpu",
                validation_leagues=validation,
                validation_episodes=2,
                evaluation_interval=16,
            )
            initial_hash = ""
            for arm in ("single", "mixed"):
                train_pools = (
                    (pools["greedy"],) if arm == "single" else (pools["greedy"], str(extra))
                )
                train(
                    TrainingConfig(
                        **configs,
                        opponent_leagues=train_pools,
                        output_dir=str(root / arm),
                        expected_initial_policy_sha256=initial_hash,
                    )
                )
                manifest = json.loads((root / arm / "run_manifest.json").read_text())
                initial_hash = manifest["initial_policy_sha256"]
                self.assertEqual(manifest["initialization"], "from_scratch")
                history = json.loads((root / arm / "validation_history.json").read_text())
                winner = [r for r in history if r["selected"]][-1]
                model = MaskablePPO.load(str(root / arm / "best_model.zip"), device="cpu")
                result = evaluate_suite(model, validation, episodes=2, seed=30000)
                self.assertEqual(result["success_rate"], winner["success_rate"])
                self.assertEqual(result["mean_return"], winner["mean_return"])
                self.assertNotIn("episode_results", winner["families"]["stats"])
            with self.assertRaisesRegex(ValueError, "initial policy weights"):
                train(
                    TrainingConfig(
                        **configs,
                        output_dir=str(root / "bad_pair"),
                        expected_initial_policy_sha256="wrong",
                    )
                )

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "RL extra is not installed")
    def test_continuation_validation_save_load_and_original_weights_are_preserved(self):
        from sb3_contrib import MaskablePPO

        from sap_rl_lab.evaluation import evaluate_policy, file_digest
        from sap_rl_lab.opponents import build_spend_gold_league
        from sap_rl_lab.training import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            league = root / "validation.json"
            build_spend_gold_league(2, seed=456).save(league)
            initial = train(
                TrainingConfig(
                    timesteps=32,
                    environments=2,
                    rollout_steps=16,
                    batch_size=16,
                    device="cpu",
                    output_dir=str(root / "initial"),
                )
            )
            before = file_digest(str(initial))
            out = root / "continued"
            config = TrainingConfig(
                timesteps=32,
                environments=2,
                rollout_steps=16,
                batch_size=16,
                device="cpu",
                output_dir=str(out),
                initialize_from=str(initial),
                validation_league=str(league),
                validation_episodes=2,
                evaluation_interval=16,
                action_cost=0.005,
                forfeit_on_limit=True,
            )
            train(config)
            self.assertEqual(before, file_digest(str(initial)))
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["initial_model_sha256"], before)
            history = json.loads((out / "validation_history.json").read_text())
            self.assertEqual(len(list(out.glob("validation_[0-9]*.json"))), len(history))
            for entry in history:
                saved = json.loads((out / entry["evaluation_file"]).read_text())
                self.assertEqual(saved["success_rate"], entry["success_rate"])
            # The initial model is evaluated before learning and participates in
            # selection; the saved winner must reproduce its recorded score.
            self.assertEqual(history[0]["timesteps"], 0)
            selected = [row for row in history if row["selected"]][-1]
            model = MaskablePPO.load(str(out / "best_model.zip"), device="cpu")
            evaluation = evaluate_policy(
                model, episodes=2, seed=config.validation_seed, opponent_league=str(league)
            )
            self.assertEqual(evaluation["mean_return"], selected["mean_return"])
            self.assertEqual(evaluation["success_rate"], selected["success_rate"])
            with self.assertRaises(FileExistsError):
                train(config)


if __name__ == "__main__":
    unittest.main()
