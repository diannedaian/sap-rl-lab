"""Bounded validation-state deviations using an existing paired rollout helper.

Trusted local checkpoints only. No training, model selection, or test-pool access.
"""

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev

from sap_rl_lab.evaluation import file_digest, policy_environment_options


def restore_prefix(env, record, seed, step_index):
    steps = record["steps"]
    if record["summary"]["seed"] != seed or not 0 <= step_index < len(steps):
        raise ValueError("Episode/step does not identify this recorded trajectory")
    env.reset(seed=seed)
    for index in range(step_index + 1):
        if env.engine.state.to_dict() != steps[index]["state"]:
            raise ValueError(f"Reconstructed pre-action state differs at step {index}")
        if index < step_index:
            _, _, terminated, truncated, _ = env.step(steps[index]["action_id"])
            if terminated or truncated:
                raise ValueError("Recorded prefix already ended")


def audit(directory, seed, step_index, action_names, repetitions, case_id, output):
    import sys

    import torch
    from sb3_contrib import MaskablePPO

    from sap_rl_lab.env import SapAutoBattlerEnv
    from sap_rl_lab.opponents import SnapshotLeague

    # Reuse the repository's bounded paired-deviation implementation rather than
    # inventing a second simulator or changing the frozen training modules.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import round3_exercises

    if not 2 <= repetitions <= 128 or not 1 <= case_id <= 100:
        raise ValueError("Requires 2–128 repetitions and a bounded development case ID")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    metadata = json.loads((directory / "summary.json").read_text())
    model_path, league = Path(metadata["model"]), Path(metadata["league"])
    if (
        file_digest(model_path) != metadata["model_sha256"]
        or file_digest(league) != metadata["league_sha256"]
    ):
        raise ValueError("Recorded model or opponent pool changed")
    record_path = directory / f"episode-{seed}.json"
    record = json.loads(record_path.read_text())
    if record["summary"] not in metadata["episodes"]:
        raise ValueError("Episode does not match the reconstruction manifest")
    config = json.loads((model_path.parent / "run_manifest.json").read_text())["config"]
    # The reused helper sums undiscounted objective rewards.
    if config["gamma"] != 1:
        raise ValueError("This undiscounted deviation audit requires gamma=1")
    shaping = {
        key: config[key]
        for key in (
            "action_cost",
            "swap_cost",
            "success_bonus_max",
            "success_action_cost",
            "observe_episode_actions",
        )
    }
    torch.set_num_threads(1)
    model = MaskablePPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)
    env = SapAutoBattlerEnv(
        opponent_provider=SnapshotLeague.load(league),
        **policy_environment_options(model),
        **shaping,
    )
    try:
        restore_prefix(env, record, seed, step_index)
        legal = {env.engine.codec.describe(a): a for a in env.engine.legal_action_ids()}
        if len(set(action_names)) != 2 or any(name not in legal for name in action_names):
            raise ValueError("Choose exactly two distinct legal actions in this state")
        value, probabilities = round3_exercises.inspect_policy(model, env)
        results = round3_exercises.alternatives(
            model, env, [legal[name] for name in action_names], case_id, repetitions
        )
    finally:
        env.close()
    differences = [b - a for a, b in zip(results[0]["returns"], results[1]["returns"])]
    report = {
        "state": record["steps"][step_index]["state"],
        "seed": seed,
        "step": step_index,
        "recorded_action": record["steps"][step_index]["action"],
        "value": value,
        "probabilities": probabilities,
        "actions": action_names,
        "results": results,
        "paired_second_minus_first": {
            "mean": fmean(differences),
            "standard_error": stdev(differences) / repetitions**0.5,
            "differences": differences,
        },
        "model_sha256": file_digest(model_path),
        "league_sha256": file_digest(league),
        "record_sha256": file_digest(record_path),
        "script_sha256": file_digest(__file__),
        "rollout_helper_sha256": file_digest(round3_exercises.__file__),
        "limits": "A development-state intervention, not optimal Q values or a global policy "
        "improvement. The recorded prefix is exact; future RNG is newly seeded and paired "
        "across actions, but different choices may consume draws differently. Mean differences "
        "and standard errors are descriptive, not a multiple-comparison-adjusted conclusion. "
        "No training, model selection, or unseen test-pool evaluation occurs.",
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("seed", "step", "actions", "value")}))
    print(
        json.dumps({k: report["paired_second_minus_first"][k] for k in ("mean", "standard_error")})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--actions", nargs=2, required=True)
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(
        args.directory,
        args.seed,
        args.step,
        args.actions,
        args.repetitions,
        args.case_id,
        args.output,
    )
