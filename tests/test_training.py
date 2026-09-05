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
