"""Replay representative old successes, losses and cutoffs before Round 3."""

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.opponents import SnapshotLeague
from sap_rl_lab.replay import ReplayRecorder, verify_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round2_directory")
    parser.add_argument("output")
    args = parser.parse_args()
    root, output = Path(args.round2_directory), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provider = SnapshotLeague.load(root / "data/test.json")
    audit = []
    for training_seed in (17, 23, 41):
        name = f"control-seed{training_seed}"
        report = json.loads((root / f"test-{name}.json").read_text())
        model = MaskablePPO.load(root / name / "best_model.zip", device="cpu")
        for reason in ("success", "lives_exhausted", "shop_action_limit"):
            expected = next(row for row in report["episode_results"] if row["reason"] == reason)
            seed = expected["seed"]
            env = SapAutoBattlerEnv(opponent_provider=provider)
            recorder = ReplayRecorder(env.engine, seed)
            while not (env.engine.state.terminated or env.engine.state.truncated):
                action, _ = model.predict(
                    env._observation(), action_masks=env.action_masks(), deterministic=True
                )
                assert env.action_masks()[int(action)]
                recorder.step(int(action))
                assert env.engine.state.gold >= 0 and len(env.engine.state.team) <= 5
            replay = recorder.finish()
            verify_replay(replay, AutoBattler(opponent_provider=provider))
            assert env.engine.state.wins == expected["wins"]
            assert len(replay.steps) == expected["actions"]
            assert sum(step.reward for step in replay.steps) == expected["return"]
            replay.save(output / f"{name}-{reason}.json")
            audit.append(
                {
                    "policy": name,
                    "seed": seed,
                    "reason": reason,
                    "steps": len(replay.steps),
                    "wins": expected["wins"],
                    "tail": [step.action for step in replay.steps[-12:]],
                    "exact_replay": True,
                    "matches_round2": True,
                }
            )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
