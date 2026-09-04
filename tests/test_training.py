import unittest

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


if __name__ == "__main__":
    unittest.main()
