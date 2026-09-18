"""Recount BC/PPO validation and reconstruct representative actual-policy replays."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path

from audit_exploration_pilot import choose_cases
from audit_midgame_confirmation import exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read, save_new
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.imitation_experiment import checked


def run(root, output):
    p = checked(root)
    candidates = []
    for seed in p["seeds"]:
        warm = read(root / f"clone-seed{seed}-complete.json")
        candidates.append(
            {
                "name": f"bc-only-seed{seed}",
                "path": warm["selected"]["path"],
                "sha256": warm["selected"]["sha256"],
                "validation": str(root / f"clone-seed{seed}-bc-validation.json"),
                "critic_trained": False,
            }
        )
    if (root / "summary.json").exists():
        for name in p["arms"]:
            done = read(root / f"{name}_complete.json")
            history = read(root / name / "validation_history.json")
            candidates.append(
                {
                    "name": name,
                    "path": done["path"],
                    "sha256": done["sha256"],
                    "validation": str(root / name / history[-1]["evaluation_file"]),
                    "critic_trained": True,
                }
            )
    elif not (root / "stopped_at_bc_gate.json").exists():
        raise ValueError("Audit requires terminal BC gate or finished PPO comparisons")
    output.mkdir(parents=True, exist_ok=False)
    cases, recounts = [], {}
    for candidate in candidates:
        if file_digest(candidate["path"]) != candidate["sha256"]:
            raise ValueError("Audited checkpoint changed")
        data = read(candidate["validation"])
        recounts[candidate["name"]] = {
            f: exact_family_counts(d, 100, p["base_config"]["validation_seed"])
            for f, d in data["families"].items()
        }
        for case in choose_cases(data["families"]):
            cases.append({**candidate, **case})
    save_new(
        output / "plan.json",
        {
            "cases": cases,
            "recounts": recounts,
            "selection": "Earliest loss (prefer no purchase), forced/truncated case and success, "
            "deduplicated per model; at most 27 cases. Diagnostic, not population sampling.",
            "protocol_sha256": file_digest(root / "protocol.json"),
        },
    )
    records = []
    for case in cases:
        expected, family = case["expected"], case["family"]
        label = f"{case['name']}-{family}-{expected['seed']}"
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(
                case["path"], p["paths"]["validation"][family], output / label, 1, expected["seed"]
            )
        replay = read(output / label / f"episode-{expected['seed']}.json")
        compare_replay(replay["summary"], expected)
        for step in replay["steps"]:
            # For BC-only this is a counterfactual future PPO objective, not its BC loss.
            if abs(step["objective_reward"] - (step["info"]["game_reward"] - 0.005)) > 1e-10:
                raise ValueError("Actual rollout objective differs from fixed PPO reward")
        records.append(
            {
                "name": case["name"],
                "directory": label,
                "critic_trained": case["critic_trained"],
                "summary": replay["summary"],
                "behavior": summarize_episode(replay, p["base_config"]["gamma"]),
            }
        )
    checked(root)
    save_new(
        output / "summary.json",
        {
            "all_replays_exact": True,
            "replays": records,
            "recounted_final_or_bc_validation_rows": len(candidates) * 400,
            "replay_actions": sum(r["summary"]["actions"] for r in records),
            "objective_formula_verified": True,
            "training_updates": 0,
            "note": "BC-only critic values are untrained; its displayed PPO reward was NOT used "
            "during imitation. PPO values are reconstructed final weights, not historical rollout "
            "values. No test-driven model reselection or recipe confirmation.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
