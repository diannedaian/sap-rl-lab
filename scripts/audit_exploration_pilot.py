"""Recount pilot curves and replay <=3 final-weight cases per model; no training."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path

from audit_midgame_confirmation import check_environment_contract, exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded import verify_inputs
from sap_rl_lab.expanded_confirmation import read, save_new
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.training import TrainingConfig


def choose_cases(families):
    rows = sorted(
        ((family, row) for family, data in families.items() for row in data["episode_results"]),
        key=lambda item: (item[1]["seed"], item[0]),
    )
    empty = [
        item
        for item in rows
        if not item[1]["success"]
        and item[1]["wins"] == 0
        and not item[1]["action_counts"].get("buy_pet", 0)
    ]
    losses = empty or [item for item in rows if not item[1]["success"]]
    forcing = [item for item in rows if item[1]["forced_end_turns"] or item[1]["truncated"]]
    successes = [item for item in rows if item[1]["success"]]
    chosen = {}
    for group in (losses, forcing, successes):
        for family, row in group[:1]:
            chosen[(family, row["seed"])] = {"family": family, "expected": row}
    return list(chosen.values())


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    p = read(root / "protocol.json")
    verify_inputs(root, p)
    summary = read(root / "pilot_summary.json")
    if (root / "pipeline_failed.json").exists() or not all(
        summary[key]
        for key in (
            "all_six_budgets_complete",
            "all_selected_validation_reloads_exact",
            "paired_initial_weights_and_rows_exact",
        )
    ):
        raise ValueError("Requires completed, exactly reloaded pilot")
    if set(summary["results"]) != set(p["arms"]):
        raise ValueError("Missing pilot arm")
    cases, hashes, counts = [], {}, {}
    for name in p["arms"]:
        folder = root / name
        done = read(root / f"{name}_complete.json")
        if file_digest(done["final_model"]) != done["final_model_sha256"]:
            raise ValueError("Final model changed")
        manifest = read(folder / "run_manifest.json")
        config = TrainingConfig(**manifest["config"])
        history = read(folder / "validation_history.json")
        counts[name] = []
        for checkpoint in history:
            path = folder / checkpoint["evaluation_file"]
            data = read(path)
            hashes[str(path)] = file_digest(path)
            if set(data["families"]) != set(p["paths"]["validation"]):
                raise ValueError("Missing validation family")
            row_counts = {}
            for family, result in data["families"].items():
                if file_digest(p["paths"]["validation"][family]) != result["league_sha256"]:
                    raise ValueError("Opponent pool changed")
                if not result["deterministic"]:
                    raise ValueError("Unexpected stochastic evaluation")
                check_environment_contract(
                    result["environment_contract"], manifest["environment_contract"]
                )
                row_counts[family] = exact_family_counts(
                    result, config.validation_episodes, config.validation_seed
                )
            counts[name].append(
                {"checkpoint": checkpoint["evaluation_file"], "families": row_counts}
            )
        # The last validation is after the final PPO update, not the preceding callback.
        final = read(folder / history[-1]["evaluation_file"])
        for case in choose_cases(final["families"]):
            cases.append(
                {
                    "model": name,
                    "path": done["final_model"],
                    "model_sha256": done["final_model_sha256"],
                    "gamma": config.gamma,
                    **case,
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "curve_counts.json", counts)
    save_new(
        output / "plan.json",
        {
            "script_sha256": file_digest(__file__),
            "protocol_sha256": file_digest(root / "protocol.json"),
            "evaluation_sha256": hashes,
            "cases": cases,
            "selection": "Per final model: earliest no-purchase zero-win loss (otherwise ordinary "
            "loss), earliest forced/truncated episode, earliest success; deduplicate. At most "
            "three cases per model. Diagnostic selection is not a population-rate estimate.",
        },
    )
    records = []
    for case in cases:
        row, family = case["expected"], case["family"]
        label = f"{case['model']}-{family}-{row['seed']}"
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(case["path"], p["paths"]["validation"][family], output / label, 1, row["seed"])
        record = read(output / label / f"episode-{row['seed']}.json")
        compare_replay(record["summary"], row)
        if file_digest(case["path"]) != case["model_sha256"]:
            raise ValueError("Model changed during replay")
        records.append(
            {
                "directory": label,
                "model": case["model"],
                "family": family,
                "summary": record["summary"],
                "behavior": summarize_episode(record, case["gamma"]),
            }
        )
        print(f"Exact final-model replay: {label}", flush=True)
    verify_inputs(root, p)
    save_new(
        output / "summary.json",
        {
            "all_curves_recounted": True,
            "rows_recounted": sum(
                f["episodes"]
                for checkpoints in counts.values()
                for checkpoint in checkpoints
                for f in checkpoint["families"].values()
            ),
            "all_replays_exact": True,
            "replays": records,
            "training_updates": 0,
            "held_out_tests_opened": False,
            "model_reselection": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
