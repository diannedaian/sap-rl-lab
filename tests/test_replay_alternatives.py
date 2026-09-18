"""Alternative-action probes must reconstruct the exact recorded prefix."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "inspect_replay_alternatives",
    Path(__file__).resolve().parents[1] / "scripts/inspect_replay_alternatives.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrefixEnv:
    def __init__(self, stop=False):
        self.position = 0
        self.actions = []
        self.stop = stop
        self.engine = SimpleNamespace(state=SimpleNamespace(to_dict=self.state))

    def state(self):
        return {"position": self.position}

    def reset(self, seed):
        self.seed = seed
        self.position = 0

    def step(self, action):
        self.actions.append(action)
        self.position += 1
        return None, 0, self.stop, False, {}


def record():
    return {
        "summary": {"seed": 12},
        "steps": [{"state": {"position": i}, "action_id": 10 + i} for i in range(3)],
    }


def test_restores_pre_action_state_without_applying_target_action():
    env = PrefixEnv()
    module.restore_prefix(env, record(), 12, 2)
    assert env.seed == 12
    assert env.position == 2
    assert env.actions == [10, 11]


@pytest.mark.parametrize("seed,index", [(13, 1), (12, -1), (12, 3)])
def test_rejects_wrong_seed_or_out_of_bounds_step(seed, index):
    with pytest.raises(ValueError, match="identify"):
        module.restore_prefix(PrefixEnv(), record(), seed, index)


def test_rejects_any_intermediate_state_mismatch():
    altered = record()
    altered["steps"][1]["state"]["position"] = 99
    with pytest.raises(ValueError, match="differs at step 1"):
        module.restore_prefix(PrefixEnv(), altered, 12, 2)


def test_rejects_premature_episode_end():
    with pytest.raises(ValueError, match="already ended"):
        module.restore_prefix(PrefixEnv(stop=True), record(), 12, 2)
