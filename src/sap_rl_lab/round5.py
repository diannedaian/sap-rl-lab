"""Finite confirmation: three paired seeds, unshaped continuation vs swap cost."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev

from .catalog import load_catalog
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .experiments import save_json
from .opponents import build_scripted_league
from .round3 import FAMILIES, source_archive
from .round4 import diagnostics, verify_inputs
from .training import TrainingConfig, train

SEEDS = (503, 607, 709)
ARMS = {"unshaped": {"swap_cost": 0.0}, "swap_cost": {"swap_cost": 0.005}}


def select_delivery(chosen):
    """Select only within the prespecified swap-cost arm using validation, never test."""
    candidates = [name for name in chosen if name.startswith("swap_cost-seed")]
    return max(
        candidates,
        key=lambda name: (
            chosen[name]["validation_success"],
            chosen[name]["validation_return"],
            -chosen[name]["seed"],
        ),
    )


def aggregate(results):
    metrics = ("success_rate", "truncation_rate", "mean_episode_actions", "mean_return")
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for metric in metrics:
            values = [results[f"{arm}-seed{seed}"][metric] for seed in SEEDS]
            groups[arm][metric] = {
                "mean": fmean(values),
                "sample_sd": stdev(values),
                "min": min(values),
                "max": max(values),
                "per_seed": values,
            }
    deltas = {
        metric: [
            results[f"swap_cost-seed{seed}"][metric] - results[f"unshaped-seed{seed}"][metric]
            for seed in SEEDS
        ]
        for metric in metrics
    }
    checks = {
        "all_swap_seeds_cutoffs_below_one_percent": all(
            results[f"swap_cost-seed{s}"]["truncation_rate"] < 0.01 for s in SEEDS
        ),
        "mean_success_drop_at_most_one_percentage_point": fmean(deltas["success_rate"]) >= -0.01,
        "mean_cutoffs_lower_than_equal_budget_control": fmean(deltas["truncation_rate"]) < 0,
    }
    return {
        "arms": groups,
        "paired_swap_minus_unshaped": {
            k: {"per_seed": v, "mean": fmean(v), "sample_sd": stdev(v)} for k, v in deltas.items()
        },
        "predeclared_checks": checks,
        "closure_screen_passed": all(checks.values()),
    }


def run(args):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    started = time.monotonic()
    previous = Path(args.previous_run).resolve()
    prior_protocol = json.loads((previous / "protocol.json").read_text())
    initial = previous / "initial_model.zip"
    if file_digest(str(initial)) != prior_protocol["initial_model_sha256"]:
        raise ValueError("Historical parent hash mismatch")
    catalog = load_catalog()
    if catalog.rules_version != 2:
        raise ValueError("Requires corrected Fish rules")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(initial, output / "initial_model.zip")
    paths = {}
    for split in ("train", "validation", "test"):
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split] = {}
        for index, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            if split == "test":
                build_scripted_league(family, 100, 1400000 + index * 10000).save(path)
            else:
                original = previous / "data" / split / f"{family}.json"
                relative = str(original.relative_to(previous))
                if file_digest(str(original)) != prior_protocol["data_sha256"][relative]:
                    raise ValueError("Round 4 pool changed")
                shutil.copyfile(original, path)
            paths[split][family] = str(path)
    common = TrainingConfig(
        timesteps=args.timesteps,
        seed=SEEDS[0],
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
        "round": 5,
        "question": "Does fixed swap cost 0.005 outperform equally budgeted unshaped continuation?",
        "seeds": SEEDS,
        "arms": ARMS,
        "base_config": asdict(common),
        "catalog_id": catalog.catalog_id,
        "initial_model": {
            "path": str(initial),
            "parent_run": str(previous),
            "model_name": "Round 3 mixed-seed307 (not Round 4 winner)",
        },
        "initial_model_sha256": file_digest(str(initial)),
        "initialization": "Same parent weights across all six runs, zero-weight episode-count "
        "input extension, fresh optimizer; each seed paired across the two reward settings",
        "paths": paths,
        "pool_generation": "Train/validation reused hash-identically from Round 4. New test "
        "pools: 100 episodes/family, seeds 1400000 + family index * 10000.",
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "source_files_sha256": source_archive(output),
        "test_episode_seed": 1500000,
        "test_episodes_per_family": 1000,
        "selection": "Within each run: maximum equal-family raw validation success, then "
        "raw return, earliest exact tie, includes step zero. Delivery model: best validation "
        "success then return among the three swap-cost models, smallest seed breaks exact tie. "
        "All six choices and delivery choice frozen before test evaluation.",
        "evaluation": "Deterministic policy, raw game reward only, no shaping. All six models "
        "use identical fresh test pools and episode seeds. 18000 total test episodes.",
        "closure_screen": "All three swap-cost seeds below 1% macro cutoffs; mean raw success "
        "no more than 1 percentage point below equal-budget controls; mean cutoffs lower. "
        "Descriptive engineering checks, not formal statistical non-inferiority.",
        "scope": "Exactly six full continuations, 3 paired seeds, fixed coefficient 0.005. "
        "No hyperparameter search, no success-bonus retraining. One shared parent, eight pets "
        "and three scripted opponent families: no fresh-initialization, unseen-family or "
        "full-game generalization claim. Stop after this evaluation regardless of outcome.",
    }
    save_json(output / "protocol.json", protocol)
    chosen, initial_hash = {}, ""
    for seed in SEEDS:
        for arm, settings in ARMS.items():
            verify_inputs(output, protocol)
            name = f"{arm}-seed{seed}"
            config = TrainingConfig(
                **{
                    **asdict(common),
                    **settings,
                    "seed": seed,
                    "output_dir": str(output / name),
                    "expected_initial_policy_sha256": initial_hash,
                }
            )
            print(f"START {name}: {config.timesteps} requested decisions", flush=True)
            arm_started = time.monotonic()
            with (output / f"{name}-training.log").open("x") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    train(config)
            manifest = json.loads((output / name / "run_manifest.json").read_text())
            initial_hash = manifest["initial_policy_sha256"]
            history = json.loads((output / name / "validation_history.json").read_text())
            winner = [row for row in history if row["selected"]][-1]
            model = output / name / "best_model.zip"
            chosen[name] = {
                "path": str(model),
                "sha256": file_digest(str(model)),
                "seed": seed,
                "arm": arm,
                "selected_timesteps": winner["timesteps"],
                "trained_timesteps": history[-1]["timesteps"],
                "validation_success": winner["success_rate"],
                "validation_return": winner["mean_return"],
                "validation_cutoffs": winner["truncation_rate"],
            }
            save_json(output / name / "runtime.json", {"seconds": time.monotonic() - arm_started})
            print(f"FINISHED {name}: validation={winner['success_rate']:.4f}", flush=True)
    save_json(output / "selection.json", chosen)
    delivery = select_delivery(chosen)
    release = output / "release"
    release.mkdir()
    shutil.copyfile(chosen[delivery]["path"], release / "model.zip")
    save_json(
        release / "manifest.json",
        {
            "model": delivery,
            **chosen[delivery],
            "catalog_id": catalog.catalog_id,
            "protocol_sha256": file_digest(str(output / "protocol.json")),
            "source_archive_sha256": file_digest(str(output / "source.zip")),
            "selection_basis": protocol["selection"],
            "inference": "Use 9 global inputs, deterministic=True, legal-action masks. "
            "Shaping disabled for reported game results. Archived rules-v2 only.",
            "status": "Frozen educational candidate; reliability outcome is in summary.json",
        },
    )
    print(f"FROZEN delivery candidate: {delivery}; opening test evaluation now", flush=True)
    results = {}
    for name, selected in chosen.items():
        verify_inputs(output, protocol)
        model = MaskablePPO.load(selected["path"], device="cpu")
        suite = evaluate_suite(model, paths["test"], episodes=1000, seed=1500000)
        suite.update(
            model_sha256=selected["sha256"],
            catalog_id=catalog.catalog_id,
            diagnostics=diagnostics(suite),
        )
        save_json(output / f"test-{name}.json", suite)
        results[name] = compact_evaluation(suite)
        print(
            f"TEST {name}: win={suite['success_rate']:.4f} cutoff={suite['truncation_rate']:.4f}",
            flush=True,
        )
    summary = {
        "results": results,
        **aggregate(results),
        "selection": chosen,
        "delivery_model": delivery,
        "training_runs": 6,
        "actual_training_steps": sum(v["trained_timesteps"] for v in chosen.values()),
        "seconds": time.monotonic() - started,
        "limitations": protocol["scope"],
    }
    verify_inputs(output, protocol)
    save_json(output / "summary.json", summary)
    print(f"COMPLETE: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", default="runs/round4-two-objectives-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=1000000)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    run(args)


if __name__ == "__main__":
    main()
