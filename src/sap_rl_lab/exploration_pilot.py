"""Paired, bounded entropy pilot; never a recipe-confirmation certificate."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean

from .evaluation import evaluate_suite, file_digest
from .expanded import run_arm, verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .midgame import ROOT
from .midgame_confirmation import check_protocol, new_config
from .round3 import source_archive

STEPS = 1_048_576
SEEDS = (17301, 17401, 17501)
ENTROPIES = {"entropy0": 0.0, "entropy001": 0.01}


def metrics(result):
    """No-purchase losses are observable; no inference from a low forcing rate."""
    families = {}
    for family, data in result["families"].items():
        rows = data["episode_results"]
        if len(rows) != data["episodes"] or not rows:
            raise ValueError("Missing evaluation rows")
        empty = sum(
            not r["success"] and r["wins"] == 0 and not r["action_counts"].get("buy_pet", 0)
            for r in rows
        )
        families[family] = {
            "episodes": len(rows),
            "no_purchase_zero_win_losses": empty,
            "no_purchase_zero_win_loss_rate": empty / len(rows),
            "mean_wins": fmean(r["wins"] for r in rows),
            "success_rate": fmean(r["success"] for r in rows),
            "forced_episode_rate": fmean(r["forced_end_turns"] > 0 for r in rows),
            "truncation_rate": fmean(r["truncated"] for r in rows),
            "mean_actions": fmean(r["actions"] for r in rows),
            "mean_pet_purchases_including_merges": fmean(
                r["action_counts"].get("buy_pet", 0) + r["action_counts"].get("merge", 0)
                for r in rows
            ),
        }
    if not families:
        raise ValueError("No evaluation families")
    return {
        "families": families,
        "mean_wins": fmean(f["mean_wins"] for f in families.values()),
        "success_rate": fmean(f["success_rate"] for f in families.values()),
        "no_purchase_zero_win_loss_rate": fmean(
            f["no_purchase_zero_win_loss_rate"] for f in families.values()
        ),
        "worst_family_no_purchase_loss_rate": max(
            f["no_purchase_zero_win_loss_rate"] for f in families.values()
        ),
        "worst_family_forced_episode_rate": max(
            f["forced_episode_rate"] for f in families.values()
        ),
        "worst_family_truncation_rate": max(f["truncation_rate"] for f in families.values()),
    }


def prepare(output):
    old_root = ROOT / "runs/midgame-confirmation-v1"
    old = check_protocol(old_root)
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split in ("train", "validation"):
        paths[split] = {}
        folder = output / "data" / split
        folder.mkdir(parents=True)
        for family, source in old["paths"][split].items():
            target = folder / f"{family}.json"
            shutil.copyfile(source, target)
            paths[split][family] = str(target)
    config = new_config(
        timesteps=STEPS,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=100,
        validation_seed=19_000_000,
        evaluation_interval=262_144,
    )
    save_new(
        output / "protocol.json",
        {
            "base_config": config,
            "arms": {
                f"{arm}-seed{seed}": {"seed": seed, "entropy_coefficient": coefficient}
                for seed in SEEDS
                for arm, coefficient in ENTROPIES.items()
            },
            "paths": paths,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for p in sorted((output / "data").rglob("*.json"))
            },
            "parent_protocol_sha256": file_digest(old_root / "protocol.json"),
            "training_steps_total": 6 * STEPS,
            "scope": "All arms use the same 50/50 mixed opponent distribution. Only entropy "
            "coefficient changes within each paired fresh initialization. Known failed seed "
            "17301 plus two new seeds. Fixed budget, no test set, "
            "no automatic rescue or extension.",
            "review": "Review full curves, final and validation-selected weights. Primary "
            "diagnostic is no-purchase zero-win loss frequency; also mean wins, ten wins, "
            "forcing and truncation. One-million-step results cannot confirm a long-run recipe.",
        },
    )


def batch(output):
    names = list(read(output / "protocol.json")["arms"])
    for start in range(0, len(names), 2):
        children = []
        try:
            for name in names[start : start + 2]:
                print(f"Starting entropy pilot: {name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "sap_rl_lab.exploration_pilot",
                            "arm",
                            "--output",
                            str(output),
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                    )
                )
            deadline = time.monotonic() + 3600
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Pilot worker failed; no automatic retry")
                if time.monotonic() > deadline:
                    raise TimeoutError("Pilot pair exceeded one hour")
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("Pilot worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)


def review(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = read(output / "protocol.json")
    verify_inputs(output, protocol)
    results = {}
    for name in protocol["arms"]:
        folder = output / name
        done = read(output / f"{name}_complete.json")
        if not done["training_complete"]:
            raise ValueError("Incomplete pilot arm")
        for field in ("best_model", "final_model"):
            if file_digest(done[field]) != done[field + "_sha256"]:
                raise ValueError("Pilot model changed")
        final = MaskablePPO.load(done["final_model"], device="cpu")
        if final.num_timesteps != STEPS:
            raise ValueError("Pilot budget mismatch")
        history = read(folder / "validation_history.json")
        selected = [r for r in history if r["selected"]][-1]
        model = MaskablePPO.load(done["best_model"], device="cpu")
        actual = evaluate_suite(
            model,
            protocol["paths"]["validation"],
            episodes=protocol["base_config"]["validation_episodes"],
            seed=protocol["base_config"]["validation_seed"],
        )
        compare_reload(actual, read(folder / selected["evaluation_file"]))
        save_new(output / f"reload-{name}.json", actual)
        results[name] = {
            "initial_policy_sha256": read(folder / "run_manifest.json")["initial_policy_sha256"],
            "actual_steps": final.num_timesteps,
            "selected_steps": selected["timesteps"],
            "selected_model_sha256": done["best_model_sha256"],
            "selected": metrics(actual),
            "final": metrics(read(folder / history[-1]["evaluation_file"])),
            "curves": [
                {
                    "timesteps": row["timesteps"],
                    "evaluation_file": row["evaluation_file"],
                    **metrics(read(folder / row["evaluation_file"])),
                }
                for row in history
            ],
            "elapsed_training_seconds": done["elapsed_seconds"],
        }
    for seed in SEEDS:
        a, b = [f"{arm}-seed{seed}" for arm in ENTROPIES]
        if results[a]["initial_policy_sha256"] != results[b]["initial_policy_sha256"]:
            raise ValueError("Paired initial weights differ")
        initial = []
        for name in (a, b):
            initial.append(read(output / name / results[name]["curves"][0]["evaluation_file"]))
        compare_reload(*initial)
    if len({r["initial_policy_sha256"] for r in results.values()}) != len(SEEDS):
        raise ValueError("Different seed pairs share initial weights")
    verify_inputs(output, protocol)
    save_new(
        output / "pilot_summary.json",
        {
            "results": results,
            "all_six_budgets_complete": True,
            "all_selected_validation_reloads_exact": True,
            "paired_initial_weights_and_rows_exact": True,
            "held_out_tests_opened": False,
            "recipe_confirmed": False,
            "next_step": "Human-readable diagnosis and bounded follow-up/independent confirmation "
            "remain required. Do not infer stable training from this pilot alone.",
        },
    )


def pipeline(output):
    prepare(output)
    try:
        batch(output)
        review(output)
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise
    print("Entropy pilot complete; recipe confirmation remains required", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "arm", "review", "pipeline"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "arm":
        if args.name not in read(output / "protocol.json")["arms"]:
            parser.error("Unknown arm")
        run_arm(output, args.name)
    else:
        {"prepare": prepare, "review": review, "pipeline": pipeline}[args.command](output)


if __name__ == "__main__":
    main()
