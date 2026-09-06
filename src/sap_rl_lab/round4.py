"""Two bounded, paired reward-objective continuations under corrected Fish rules."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from .catalog import load_catalog
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .experiments import save_json
from .opponents import build_scripted_league
from .round3 import FAMILIES, source_archive
from .training import TrainingConfig, train


def verify_inputs(output, protocol):
    root = Path(__file__).resolve().parents[2]
    for relative, expected in protocol["source_files_sha256"].items():
        if file_digest(str(root / relative)) != expected:
            raise ValueError(f"Source changed after protocol freeze: {relative}")
    for relative, expected in protocol["data_sha256"].items():
        if file_digest(str(output / relative)) != expected:
            raise ValueError(f"League changed after protocol freeze: {relative}")
    if file_digest(str(output / "initial_model.zip")) != protocol["initial_model_sha256"]:
        raise ValueError("Initial model changed")


def diagnostics(suite):
    rows = [row for family in suite["families"].values() for row in family["episode_results"]]
    wins = [row for row in rows if row["success"]]
    cutoffs = [row for row in rows if row["truncated"]]
    return {
        "mean_swaps_per_episode": fmean(
            row["action_counts"].get("swap_adjacent", 0) for row in rows
        ),
        "mean_actions_on_success": fmean(row["actions"] for row in wins) if wins else None,
        "successful_episodes": len(wins),
        "cutoff_episodes": len(cutoffs),
        "mean_unspent_gold_per_battle": fmean(
            family["mean_unspent_gold_per_battle"] for family in suite["families"].values()
        ),
    }


def comparison(a, b):
    """Stratified episode-paired intervals, conditional on these fixed policies."""
    import numpy as np

    rng = np.random.default_rng(409)
    result = {}
    for metric in ("success", "truncated", "wins", "return", "actions"):
        draws, means = [], []
        for family in FAMILIES:
            left, right = a["families"][family], b["families"][family]
            if left["league_sha256"] != right["league_sha256"]:
                raise ValueError("Comparison pools differ")
            left_rows, right_rows = left["episode_results"], right["episode_results"]
            if [r["seed"] for r in left_rows] != [r["seed"] for r in right_rows]:
                raise ValueError("Comparison episode seeds differ")
            delta = np.asarray(
                [float(x[metric]) - float(y[metric]) for x, y in zip(left_rows, right_rows)]
            )
            means.append(float(delta.mean()))
            draws.append(rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1))
        result[metric] = {
            "mean_difference": fmean(means),
            "episode_paired_95_percent": list(
                map(float, np.quantile(np.mean(draws, axis=0), [0.025, 0.975]))
            ),
        }
    return result


def run(args):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    started = time.monotonic()
    base = Path(args.base_run).resolve()
    selection = json.loads((base / "selection.json").read_text())
    initial = selection["mixed-seed307"]
    if file_digest(initial["path"]) != initial["sha256"]:
        raise ValueError("The selected historical checkpoint does not match its recorded hash")
    catalog = load_catalog()
    if catalog.rules_version != 2:
        raise ValueError("This experiment requires the corrected Fish rules")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(initial["path"], output / "initial_model.zip")

    paths = {}
    split_starts = {"train": 900000, "validation": 1000000, "test": 1100000}
    for split, start in split_starts.items():
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split] = {}
        for i, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            build_scripted_league(family, 100, start + i * 10000).save(path)
            paths[split][family] = str(path)
    common = TrainingConfig(
        timesteps=args.timesteps,
        seed=409,
        environments=8,
        rollout_steps=512,
        batch_size=256,
        vector_backend="dummy",
        device="cpu",
        torch_threads=1,
        initialize_from=str(output / "initial_model.zip"),
        observe_episode_actions=True,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=200,
        validation_seed=1200000,
        evaluation_interval=250000,
    )
    protocol = {
        "question": "Compare a swap penalty with a success-only action-efficiency bonus",
        "base_config": asdict(common),
        "arms": {
            "swap_cost": {"swap_cost": 0.005},
            "success_bonus": {"success_bonus_max": 1.0, "success_action_cost": 0.005},
        },
        "bonus_formula": "Only on true target-win termination: "
        "max(0, 1 - 0.005 * total decisions); all other endings get zero bonus",
        "decision_count": "All episode decisions, including end-turn; normalized by "
        "max_turns * max_actions_per_turn in one added observation input",
        "catalog_id": catalog.catalog_id,
        "initial_model": {**initial, "parent_run": str(base), "model_name": "mixed-seed307"},
        "initial_model_sha256": initial["sha256"],
        "initialization": "Both copy identical actor/critic weights; one zero-weight input "
        "column preserves the old policy; fresh identical optimizer states",
        "paths": paths,
        "pool_generation": {
            "starts": split_starts,
            "family_stride": 10000,
            "episodes_per_family": 100,
        },
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "source_files_sha256": source_archive(output),
        "test_episode_seed": 1300000,
        "test_episodes_per_family": 1000,
        "selection": "Maximum equal-family raw validation success, then raw return; "
        "earliest exact tie. Includes step-zero baseline. No test selection.",
        "fixed": "Both use rules-v2, identical masks, gamma=1, no entropy bonus, no general "
        "action cost, and unchanged truncation/forfeit semantics",
        "scope": "Exactly two full training runs, one paired continuation seed. Frozen parent "
        "is evaluated but not trained as a third arm. No unshaped continuation control, so "
        "improvements vs parent do not isolate shaping from extra training. No across-seed "
        "or unseen-family generalization claim.",
        "evaluation": "No shaping at evaluation. Primary: raw 10-win rate and cutoffs, then "
        "swaps, actions on successful episodes, and unspent gold. Fewer actions alone are "
        "not success.",
        "target": "Exploratory screening: below 1% macro cutoffs while losing no more than "
        "1 percentage point of raw success vs frozen parent. This is not a formal "
        "non-inferiority test.",
    }
    save_json(output / "protocol.json", protocol)
    verify_inputs(output, protocol)
    chosen = {}
    initial_policy_hash = ""
    for name, settings in protocol["arms"].items():
        verify_inputs(output, protocol)
        config = TrainingConfig(
            **{
                **asdict(common),
                **settings,
                "output_dir": str(output / name),
                "expected_initial_policy_sha256": initial_policy_hash,
            }
        )
        print(f"START {name}: {config.timesteps} requested steps", flush=True)
        arm_started = time.monotonic()
        with (output / f"{name}-training.log").open("x") as log:
            with redirect_stdout(log), redirect_stderr(log):
                train(config)
        manifest = json.loads((output / name / "run_manifest.json").read_text())
        initial_policy_hash = manifest["initial_policy_sha256"]
        history = json.loads((output / name / "validation_history.json").read_text())
        winner = [r for r in history if r["selected"]][-1]
        model = output / name / "best_model.zip"
        chosen[name] = {
            "path": str(model),
            "sha256": file_digest(str(model)),
            "selected_timesteps": winner["timesteps"],
            "trained_timesteps": history[-1]["timesteps"],
            "validation_success": winner["success_rate"],
            "validation_cutoffs": winner["truncation_rate"],
        }
        save_json(output / name / "runtime.json", {"seconds": time.monotonic() - arm_started})
        print(f"FINISHED {name}: {json.dumps(chosen[name])}", flush=True)
    # Freeze both selections before opening any test outcomes.
    save_json(output / "selection.json", chosen)
    verify_inputs(output, protocol)
    results = {}
    candidates = {
        "frozen_parent": str(output / "initial_model.zip"),
        **{k: v["path"] for k, v in chosen.items()},
    }
    for name, path in candidates.items():
        model = MaskablePPO.load(path, device="cpu")
        suite = evaluate_suite(model, paths["test"], episodes=1000, seed=1300000)
        suite["model_sha256"] = file_digest(path)
        suite["catalog_id"] = catalog.catalog_id
        suite["diagnostics"] = diagnostics(suite)
        save_json(output / f"test-{name}.json", suite)
        results[name] = suite
        print(
            f"TEST {name}: success={suite['success_rate']:.4f} "
            f"cutoffs={suite['truncation_rate']:.4f} actions={suite['mean_episode_actions']:.1f}",
            flush=True,
        )
        del model
    summary = {
        "results": {name: compact_evaluation(value) for name, value in results.items()},
        "comparisons": {
            "success_bonus_minus_swap_cost": comparison(
                results["success_bonus"], results["swap_cost"]
            ),
            **{
                f"{name}_minus_frozen_parent": comparison(results[name], results["frozen_parent"])
                for name in chosen
            },
        },
        "selection": chosen,
        "training_runs": 2,
        "actual_training_steps": sum(v["trained_timesteps"] for v in chosen.values()),
        "seconds": time.monotonic() - started,
        "limitations": protocol["scope"],
    }
    verify_inputs(output, protocol)
    save_json(output / "summary.json", summary)
    print(f"COMPLETE: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", default="runs/round3-full-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    run(args)


if __name__ == "__main__":
    main()
