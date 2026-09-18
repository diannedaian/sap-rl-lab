"""Approved bounded reward-cost ablation; preserves all frozen prior experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

from . import evaluation
from .expanded import verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .exploration_pilot import metrics
from .historical_confirmation import copy_checked, fallback_counts
from .midgame import ROOT
from .round3 import source_archive
from .stability_confirmation import verify as verify_parent
from .training import TrainingConfig, midgame_checkpoint_score, policy_digest, train

STEPS = 2_097_152
SEEDS = (17301, 203101, 203201)
ARMS = {
    "all_actions": {"action_cost": 0.005, "swap_cost": 0.0},
    "swap_only": {"action_cost": 0.0, "swap_cost": 0.005},
}
VAL_SEED, TEST_SEED = 26_000_000, 27_000_000
PLAN = ROOT / "docs/REWARD_COST_PLAN.md"
LEARNED = ("fresh_stats", "fresh_mix")


def checked(output):
    p = read(output / "protocol.json")
    if evaluation.file_digest(output / "protocol.json") != read(output / "seal.json")["sha256"]:
        raise ValueError("Reward-cost protocol changed")
    verify_inputs(output, p)
    if evaluation.file_digest(PLAN) != p["plan_sha256"]:
        raise ValueError("Reward-cost plan changed")
    for path, digest in p["audit_helpers_sha256"].items():
        if evaluation.file_digest(ROOT / path) != digest:
            raise ValueError("Frozen diagnostic helper changed")
    return p


def prepare(output):
    parent = ROOT / "runs/stability-confirmation-v1"
    prior = verify_parent(parent)
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split in ("train", "validation", "test"):
        paths[split] = {}
        for family, source in prior["paths"][split].items():
            dest = output / "data" / split / f"{family}.json"
            copy_checked(source, dest)
            paths[split][family] = str(dest)
    config = dict(prior["base_config"])
    config.update(
        timesteps=STEPS,
        entropy_coefficient=0.0,
        action_cost=0.005,
        swap_cost=0.0,
        success_bonus_max=0.0,
        success_action_cost=0.0,
        initialize_from="",
        expected_initial_policy_sha256="",
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=100,
        validation_seed=VAL_SEED,
        evaluation_interval=524_288,
    )
    TrainingConfig(**config).validate()
    fixed = {
        "environments": 8,
        "rollout_steps": 256,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "device": "cpu",
        "torch_threads": 1,
        "vector_backend": "dummy",
        "shop_action_limit_mode": "force_battle",
        "max_actions_per_turn": 30,
        "catalog_id": "turtle-v0.46-midgame-v5",
    }
    if any(config[k] != v for k, v in fixed.items()):
        raise ValueError("Inherited configuration differs from registered fixed settings")
    helpers = (
        "scripts/audit_exploration_pilot.py",
        "scripts/audit_midgame_confirmation.py",
        "scripts/inspect_confirmation_delivery.py",
        "scripts/summarize_policy_replays.py",
    )
    save_new(
        output / "protocol.json",
        {
            "base_config": config,
            "arms": {
                f"{arm}-seed{seed}": {"seed": seed, **costs}
                for seed in SEEDS
                for arm, costs in ARMS.items()
            },
            "paths": paths,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(x.relative_to(output)): evaluation.file_digest(x)
                for x in sorted((output / "data").rglob("*.json"))
            },
            "parent_protocol_sha256": evaluation.file_digest(parent / "protocol.json"),
            "plan_sha256": copy_checked(PLAN, output / "plan.md"),
            "audit_helpers_sha256": {x: evaluation.file_digest(ROOT / x) for x in helpers},
            "training_steps_total": 6 * STEPS,
            "validation_records_per_model": 6,
            "test_episodes_per_family": 200,
            "test_seed": TEST_SEED,
            "primary_endpoint": "fixed-budget final weights; validation best is secondary",
            "test_limitations": "Previously studied opponent pools, new episode seeds; diagnostic "
            "holdout, not independent confirmation. Known failure seed retained.",
            "automatic_recipe_confirmation": False,
        },
    )
    save_new(output / "seal.json", {"sha256": evaluation.file_digest(output / "protocol.json")})
    checked(output)


def checkpointing_evaluator(original, folder):
    """Persist the exact evaluated policy without altering the evaluator result or RNGs."""
    index = 0

    def evaluate(model, *args, **kwargs):
        nonlocal index
        import random

        import numpy as np
        import torch

        result = original(model, *args, **kwargs)
        weights = folder / "validation_weights"
        weights.mkdir(exist_ok=True)
        manifest = weights / "run_manifest.json"
        if not manifest.exists():
            copy_checked(folder / "run_manifest.json", manifest)
        elif evaluation.file_digest(manifest) != evaluation.file_digest(
            folder / "run_manifest.json"
        ):
            raise ValueError("Training manifest changed during validation")
        path = weights / f"eval{index:03d}.zip"
        if path.exists():
            raise FileExistsError(path)
        before = policy_digest(model.policy)
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        model.save(str(path))
        after_np = np.random.get_state()
        if (
            before != policy_digest(model.policy)
            or py_rng != random.getstate()
            or np_rng[0] != after_np[0]
            or not np.array_equal(np_rng[1], after_np[1])
            or np_rng[2:] != after_np[2:]
            or not torch.equal(torch_rng, torch.get_rng_state())
        ):
            raise ValueError("Checkpoint persistence changed policy or random state")
        save_new(
            weights / f"eval{index:03d}.json",
            {
                "path": str(path),
                "sha256": evaluation.file_digest(path),
                "policy_sha256": before,
                "timesteps": model.num_timesteps,
                "evaluation_file": f"validation_{model.num_timesteps}_eval{index:03d}.json",
            },
        )
        index += 1
        return result

    return evaluate


def arm(output, name):
    p = checked(output)
    folder = output / name
    config = TrainingConfig(**{**p["base_config"], **p["arms"][name], "output_dir": str(folder)})
    started = time.monotonic()
    wrapper = checkpointing_evaluator(evaluation.evaluate_suite, folder)
    with (output / f"{name}.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
        with patch.object(evaluation, "evaluate_suite", wrapper):
            final = train(config)
    checked(output)
    save_new(
        output / f"{name}_complete.json",
        {
            "training_complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "final_model": str(final),
            "final_model_sha256": evaluation.file_digest(final),
            "best_model": str(folder / "best_model.zip"),
            "best_model_sha256": evaluation.file_digest(folder / "best_model.zip"),
        },
    )
    print(f"Reward-cost training complete: {name}", flush=True)


def batch(output, stage, names):
    for start in range(0, len(names), 2):
        children = []
        try:
            for name in names[start : start + 2]:
                print(f"Starting {stage}: {name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "sap_rl_lab.reward_cost_pilot",
                            stage,
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
                    raise RuntimeError(f"{stage} worker failed; no automatic retry")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{stage} pair exceeded one hour")
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError(f"{stage} worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, models, initials = checked(output), {}, {}
    for name in p["arms"]:
        folder = output / name
        done, manifest = read(output / f"{name}_complete.json"), read(folder / "run_manifest.json")
        expected = {**p["base_config"], **p["arms"][name], "output_dir": str(folder)}
        if manifest["config"] != read_config(expected) or not done["training_complete"]:
            raise ValueError("Actual arm configuration or completion differs")
        history = read(folder / "validation_history.json")
        if len(history) != 6 or [h["timesteps"] for h in history] != [
            0,
            524288,
            1048576,
            1572864,
            STEPS,
            STEPS,
        ]:
            raise ValueError("Incomplete fixed validation schedule")
        best_index = max(
            range(len(history)),
            key=lambda i: midgame_checkpoint_score(read(folder / history[i]["evaluation_file"])),
        )
        selected_index = [i for i, h in enumerate(history) if h["selected"]][-1]
        if best_index != selected_index:
            raise ValueError("Selection differs from predeclared earliest-tie ranking")
        bindings = []
        for i, entry in enumerate(history):
            item = read(folder / "validation_weights" / f"eval{i:03d}.json")
            saved = MaskablePPO.load(item["path"], device="cpu")
            if (
                evaluation.file_digest(item["path"]) != item["sha256"]
                or saved.num_timesteps != entry["timesteps"]
                or item["evaluation_file"] != entry["evaluation_file"]
                or policy_digest(saved.policy) != item["policy_sha256"]
            ):
                raise ValueError("Exact validation checkpoint binding failed")
            bindings.append(item)
        initials[name] = manifest["initial_policy_sha256"]
        if bindings[0]["policy_sha256"] != initials[name]:
            raise ValueError("Initial validation policy differs from initialization")
        for kind, i, field in (("best", best_index, "best_model"), ("final", 5, "final_model")):
            model = MaskablePPO.load(done[field], device="cpu")
            if (
                evaluation.file_digest(done[field]) != done[field + "_sha256"]
                or policy_digest(model.policy) != bindings[i]["policy_sha256"]
                or model.num_timesteps != history[i]["timesteps"]
            ):
                raise ValueError("Selected/final weights differ from evaluated policy")
            if kind == "final" and model.num_timesteps != STEPS:
                raise ValueError("Training budget not exactly completed")
            models[f"{name}-{kind}"] = {
                "arm": name,
                "kind": kind,
                "path": done[field],
                "sha256": done[field + "_sha256"],
                "validation_path": str(folder / history[i]["evaluation_file"]),
                "validation_sha256": evaluation.file_digest(folder / history[i]["evaluation_file"]),
                "timesteps": model.num_timesteps,
            }
    for seed in SEEDS:
        names = [f"{a}-seed{seed}" for a in ARMS]
        if len({initials[n] for n in names}) != 1:
            raise ValueError("Paired initial policy tensors differ")
        compare_reload(*(read(output / n / "validation_0_eval000.json") for n in names))
    if len(set(initials.values())) != 3:
        raise ValueError("Different seeds share initial policies")
    checked(output)
    save_new(
        output / "selection.json",
        {
            "protocol_sha256": evaluation.file_digest(output / "protocol.json"),
            "models": models,
            "paired_initial_weights_and_rows_exact": True,
            "primary_endpoint": "final",
            "independent_confirmation": False,
        },
    )


def read_config(config):
    """Normalize tuples exactly as manifest JSON does."""
    import json

    return json.loads(json.dumps(config))


def frozen(output):
    p, selection = checked(output), read(output / "selection.json")
    if selection["protocol_sha256"] != evaluation.file_digest(output / "protocol.json"):
        raise ValueError("Selection protocol changed")
    expected = {f"{n}-{k}" for n in p["arms"] for k in ("best", "final")}
    if set(selection["models"]) != expected:
        raise ValueError("Incomplete all-model selection")
    for item in selection["models"].values():
        for path, digest in (
            (item["path"], item["sha256"]),
            (item["validation_path"], item["validation_sha256"]),
        ):
            if evaluation.file_digest(path) != digest:
                raise ValueError("Frozen model/evaluation changed")
    return p, selection


def require_reloads(output, selection):
    for name, item in selection["models"].items():
        actual = read(output / f"reload-{name}.json")
        if actual["model_sha256"] != item["sha256"] or actual[
            "selection_sha256"
        ] != evaluation.file_digest(output / "selection.json"):
            raise ValueError("Reload model/selection changed")
        compare_reload(actual, read(item["validation_path"]))


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selection = frozen(output)
    if stage == "test":
        require_reloads(output, selection)
    item = selection["models"][name]
    split, episodes, seed = (
        ("validation", 100, VAL_SEED) if stage == "reload" else ("test", 200, TEST_SEED)
    )
    binding = {
        "model_sha256": item["sha256"],
        "selection_sha256": evaluation.file_digest(output / "selection.json"),
    }
    save_new(output / f"{stage}-{name}-started.json", binding)
    model = MaskablePPO.load(item["path"], device="cpu")
    result = evaluation.evaluate_suite(model, p["paths"][split], episodes=episodes, seed=seed)
    if stage == "reload":
        compare_reload(result, read(item["validation_path"]))
    result.update(binding)
    result["opponent_fallback_coverage"] = {
        f: fallback_counts(
            result["families"][f]["episode_results"], {x["turn"] for x in read(path)["snapshots"]}
        )
        for f, path in p["paths"][split].items()
    }
    frozen(output)
    save_new(output / f"{stage}-{name}.json", result)
    print(f"Reward-cost {stage} complete: {name}", flush=True)


def summarize(output):
    from .experiments import paired_comparison

    p, selection = frozen(output)
    require_reloads(output, selection)
    results = {}
    for name in p["arms"]:
        folder = output / name
        history = read(folder / "validation_history.json")
        done = read(output / f"{name}_complete.json")
        results[name] = {
            "actual_steps": STEPS,
            "initial_policy_sha256": read(folder / "run_manifest.json")["initial_policy_sha256"],
            "selected": metrics(read(output / f"reload-{name}-best.json")),
            "final": metrics(read(output / f"reload-{name}-final.json")),
            "curves": [
                {
                    "timesteps": h["timesteps"],
                    "evaluation_file": h["evaluation_file"],
                    **metrics(read(folder / h["evaluation_file"])),
                }
                for h in history
            ],
            "elapsed_training_seconds": done["elapsed_seconds"],
        }
    pairs = {}
    for seed in SEEDS:
        pairs[str(seed)] = {}
        for kind in ("final", "best"):
            data = {a: read(output / f"test-{a}-seed{seed}-{kind}.json") for a in ARMS}
            row = {a: metrics(d) for a, d in data.items()}
            for a, d in data.items():
                item = selection["models"][f"{a}-seed{seed}-{kind}"]
                if d["model_sha256"] != item["sha256"] or d[
                    "selection_sha256"
                ] != evaluation.file_digest(output / "selection.json"):
                    raise ValueError("Test model/selection binding changed")
                row[a]["learned_success"] = fmean(d["families"][f]["success_rate"] for f in LEARNED)
            row["learned_difference"] = (
                row["swap_only"]["learned_success"] - row["all_actions"]["learned_success"]
            )
            row["family_comparisons"] = {
                f: paired_comparison(
                    data["swap_only"]["families"][f],
                    data["all_actions"]["families"][f],
                    resamples=2000,
                )
                for f in p["paths"]["test"]
            }
            pairs[str(seed)][kind] = row
    gates = candidate_gates(results, pairs)
    save_new(
        output / "pilot_summary.json",
        {
            "results": results,
            "pairs": pairs,
            "gates": gates,
            "all_six_budgets_complete": True,
            "all_selected_validation_reloads_exact": True,
            "all_final_validation_reloads_exact": True,
            "paired_initial_weights_and_rows_exact": True,
            "candidate_pilot_screen_passed": all(gates.values()),
            "recipe_confirmed": False,
            "training_steps_total": 6 * STEPS,
            "held_out_tests_opened": True,
            "limitations": "Diagnostic reused opponent holdout; known failure seed included. "
            "No independent long-budget recipe confirmation or automatic extension.",
        },
    )


def candidate_gates(results, pairs):
    candidates = [results[f"swap_only-seed{s}"]["final"] for s in SEEDS]
    candidates += [pairs[str(s)]["final"]["swap_only"] for s in SEEDS]
    return {
        "final_empty_losses_controlled": all(
            x["worst_family_no_purchase_loss_rate"] <= 0.01 for x in candidates
        ),
        "final_forcing_controlled": all(
            x["worst_family_forced_episode_rate"] <= 0.02 for x in candidates
        ),
        "final_zero_truncations": all(x["worst_family_truncation_rate"] == 0 for x in candidates),
        "all_three_learned_differences_nonnegative": all(
            pairs[str(s)]["final"]["learned_difference"] >= 0 for s in SEEDS
        ),
        "all_three_mean_wins_not_lower": all(
            pairs[str(s)]["final"]["swap_only"]["mean_wins"]
            >= pairs[str(s)]["final"]["all_actions"]["mean_wins"]
            for s in SEEDS
        ),
    }


def pipeline(output):
    started = time.monotonic()
    prepare(output)
    try:
        batch(output, "arm", list(checked(output)["arms"]))
        freeze(output)
        names = list(read(output / "selection.json")["models"])
        batch(output, "reload", names)
        batch(output, "test", names)
        summarize(output)
        save_new(
            output / "pipeline_complete.json",
            {
                "all_stages_complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "replay_audit_and_human_report_pending": True,
                "recipe_confirmed": False,
            },
        )
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise
    print("Reward-cost pilot complete; replay audit and evidence review still required", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "pipeline", "arm", "freeze", "reload", "test", "summarize")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command in ("reload", "test"):
        inference(output, args.name, args.command)
    elif args.command == "arm":
        arm(output, args.name)
    else:
        globals()[args.command](output)
