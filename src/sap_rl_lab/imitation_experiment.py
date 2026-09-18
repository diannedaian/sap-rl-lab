"""Bounded script BC -> PPO experiment, preserving all previous checkpoints.

Only new output directories, two CPU workers, no test-driven selection or retries.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

from . import evaluation, imitation
from .expanded import verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .exploration_pilot import metrics
from .historical_confirmation import copy_checked, fallback_counts
from .midgame import FAMILIES, ROOT
from .reward_cost_pilot import checked as check_parent
from .reward_cost_pilot import checkpointing_evaluator
from .round3 import source_archive
from .training import TrainingConfig, midgame_checkpoint_score, policy_digest, train

SEEDS = (17301, 204101, 204201)
STEPS = 8_388_608
VAL_SEED, TEST_SEED = 30_000_000, 31_000_000
PLAN = ROOT / "docs/IMITATION_PLAN.md"
OLD_BEST = ROOT / "runs/midgame-confirmation-v1/historical_mix-seed17101/best_model.zip"
OLD_SHA = "094f5f2ca802f98ae5ca7a9481d6c168d11e0c3853774cbd65e9f97c0866e644"


def checked(output):
    p = read(output / "protocol.json")
    if evaluation.file_digest(output / "protocol.json") != read(output / "seal.json")["sha256"]:
        raise ValueError("Imitation protocol changed")
    verify_inputs(output, p)
    if evaluation.file_digest(PLAN) != p["plan_sha256"]:
        raise ValueError("Imitation plan changed")
    if evaluation.file_digest(OLD_BEST) != OLD_SHA:
        raise ValueError("Previous best model changed")
    for path, digest in p["audit_helpers_sha256"].items():
        if evaluation.file_digest(ROOT / path) != digest:
            raise ValueError("Audit helper changed")
    return p


def prepare(output):
    prior = check_parent(ROOT / "runs/reward-cost-pilot-v1")
    if evaluation.file_digest(OLD_BEST) != OLD_SHA:
        raise ValueError("Previous best model hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split, items in prior["paths"].items():
        paths[split] = {}
        for family, source in items.items():
            target = output / "data" / split / f"{family}.json"
            copy_checked(source, target)
            paths[split][family] = str(target)
    copy_checked(OLD_BEST, output / "reference" / "best_model.zip")
    base = dict(prior["base_config"])
    base.update(
        timesteps=STEPS,
        action_cost=0.005,
        swap_cost=0.0,
        entropy_coefficient=0.0,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_seed=VAL_SEED,
        evaluation_interval=1_048_576,
        validation_episodes=100,
        initialize_from="",
        expected_initial_policy_sha256="",
    )
    TrainingConfig(**base).validate()
    save_new(
        output / "protocol.json",
        {
            "base_config": base,
            "seeds": SEEDS,
            "steps_per_model": STEPS,
            "arms": {f"{arm}-seed{s}": {"seed": s} for s in SEEDS for arm in ("scratch", "bc")},
            "paths": paths,
            "teachers": FAMILIES,
            "demo_train": {"episodes_per_teacher": 600, "seed": 28_000_000},
            "demo_validation": {"episodes_per_teacher": 100, "seed": 29_000_000},
            "bc": {"epochs": 20, "batch_size": 512, "lr": 3e-4},
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(x.relative_to(output)): evaluation.file_digest(x)
                for x in sorted((output / "data").rglob("*.json"))
            },
            "plan_sha256": copy_checked(PLAN, output / "plan.md"),
            "audit_helpers_sha256": {
                path: evaluation.file_digest(ROOT / path)
                for path in (
                    "scripts/audit_imitation.py",
                    "scripts/audit_exploration_pilot.py",
                    "scripts/audit_midgame_confirmation.py",
                    "scripts/inspect_confirmation_delivery.py",
                    "scripts/summarize_policy_replays.py",
                )
            },
            "parent_protocol_sha256": evaluation.file_digest(
                ROOT / "runs/reward-cost-pilot-v1/protocol.json"
            ),
            "old_best_sha256": OLD_SHA,
            "maximum_ppo_steps": 6 * STEPS,
            "maximum_worker_processes": 2,
            "maximum_pipeline_seconds": 6 * 3600,
            "primary_endpoint": "fixed-budget final; validation-selected best secondary",
            "test_seed": TEST_SEED,
            "test_episodes_per_family": 200,
            "validation_records_per_model": 10,
            "limitations": "Known failure seed 17301 and two fresh seeds. Previously studied "
            "opponent pools with fresh episode seeds: diagnostic, not an independent ecology. "
            "40 pets but normal shops stop at Tier 3; Tier 4 is level-up rewards only. "
            "Three scripted teachers never freeze and are not optimal experts.",
            "automatic_recipe_confirmation": False,
        },
    )
    save_new(output / "seal.json", {"sha256": evaluation.file_digest(output / "protocol.json")})
    checked(output)


def demonstrations(output):
    p = checked(output)
    config = TrainingConfig(**p["base_config"])
    records = {}
    for split in ("train", "validation"):
        records[split] = imitation.collect(
            config,
            p["teachers"],
            **p[f"demo_{split}"],
            destination=output / "demos" / f"{split}.npz",
        )
    imitation.validate_split(
        *(imitation.load_data(output / "demos" / f"{s}.npz") for s in ("train", "validation"))
    )
    save_new(output / "demonstrations.json", records)
    checked(output)


def demo_inputs(output):
    records = read(output / "demonstrations.json")
    for record in records.values():
        if evaluation.file_digest(record["path"]) != record["sha256"]:
            raise ValueError("Demonstration data changed")
    return records


def clone(output, seed):
    import torch
    from sb3_contrib import MaskablePPO

    p, records = checked(output), demo_inputs(output)
    torch.set_num_threads(1)
    config = TrainingConfig(**{**p["base_config"], "seed": seed})
    started = time.monotonic()
    with (
        (output / f"clone-seed{seed}.log").open("x") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        model = imitation.fresh_model(config)
        try:
            result = imitation.fit(
                model,
                imitation.load_data(records["train"]["path"]),
                imitation.load_data(records["validation"]["path"]),
                output / f"clone-seed{seed}",
                seed=seed,
                **p["bc"],
            )
        finally:
            model.get_env().close()
        save_new(
            output / f"clone-seed{seed}" / "run_manifest.json",
            {
                "config": asdict(config),
                "environment_contract": model.sap_environment_contract,
                "training": "actor-only behavior cloning; critic remains random",
            },
        )
        initial = MaskablePPO.load(result["history"][0]["path"], device="cpu")
        selected = MaskablePPO.load(result["selected"]["path"], device="cpu")
        validation = {}
        for kind, candidate in (("initial", initial), ("bc", selected)):
            validation[kind] = evaluation.evaluate_suite(
                candidate, p["paths"]["validation"], episodes=100, seed=VAL_SEED
            )
            save_new(output / f"clone-seed{seed}-{kind}-validation.json", validation[kind])
    checked(output)
    demo_inputs(output)
    save_new(
        output / f"clone-seed{seed}-complete.json",
        {
            "seed": seed,
            "selected": result["selected"],
            "initial": result["history"][0],
            "initial_metrics": metrics(validation["initial"]),
            "bc_metrics": metrics(validation["bc"]),
            "elapsed_seconds": time.monotonic() - started,
            "critic_unchanged": result["critic_unchanged"],
            "ppo_optimizer_untouched": result["ppo_optimizer_untouched"],
        },
    )


def bc_gate(rows):
    """A basic competence screen, deliberately NOT a delivery reliability target."""
    if len(rows) != 3:
        raise ValueError("All three BC seeds are required")
    return {
        "nonempty_play": all(
            r["bc_metrics"]["worst_family_no_purchase_loss_rate"] <= 0.05 for r in rows
        ),
        "not_mostly_forced": all(
            r["bc_metrics"]["worst_family_forced_episode_rate"] <= 0.20 for r in rows
        ),
        "no_truncation": all(r["bc_metrics"]["worst_family_truncation_rate"] == 0 for r in rows),
        "basic_strength": all(r["bc_metrics"]["mean_wins"] >= 1.0 for r in rows),
        "improved_from_initial": all(
            r["bc_metrics"]["mean_wins"] >= r["initial_metrics"]["mean_wins"] + 0.5 for r in rows
        ),
    }


def gate(output):
    p = checked(output)
    rows = [read(output / f"clone-seed{s}-complete.json") for s in p["seeds"]]
    criteria = bc_gate(rows)
    save_new(
        output / "bc_gate.json",
        {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "rows": rows,
            "recipe_confirmed": False,
        },
    )
    return all(criteria.values())


def arm(output, name):
    p = checked(output)
    if not read(output / "bc_gate.json")["passed"]:
        raise ValueError("BC competence gate did not pass; PPO not authorized by protocol")
    seed = p["arms"][name]["seed"]
    warm = read(output / f"clone-seed{seed}-complete.json")
    source = warm["selected"] if name.startswith("bc-") else warm["initial"]
    if evaluation.file_digest(source["path"]) != source["sha256"]:
        raise ValueError("Initialization changed")
    folder = output / name
    config = TrainingConfig(
        **{
            **p["base_config"],
            **p["arms"][name],
            "output_dir": str(folder),
            "initialize_from": source["path"],
            "expected_initial_policy_sha256": source["policy_sha256"],
        }
    )
    started = time.monotonic()
    with (output / f"{name}.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
        with patch.object(
            evaluation, "evaluate_suite", checkpointing_evaluator(evaluation.evaluate_suite, folder)
        ):
            path = train(config)
    checked(output)
    save_new(
        output / f"{name}_complete.json",
        {
            "path": str(path),
            "sha256": evaluation.file_digest(path),
            "elapsed_seconds": time.monotonic() - started,
            "initialization": source,
        },
    )


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, models = checked(output), {}
    for name in p["arms"]:
        folder = output / name
        history = read(folder / "validation_history.json")
        if len(history) != p["validation_records_per_model"]:
            raise ValueError("Incomplete validation curve")
        scores = [midgame_checkpoint_score(read(folder / h["evaluation_file"])) for h in history]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        manifest = read(folder / "run_manifest.json")
        if (
            manifest["initial_policy_sha256"]
            != read(output / f"{name}_complete.json")["initialization"]["policy_sha256"]
        ):
            raise ValueError("PPO warm start mismatch")
        for i, h in enumerate(history):
            binding = read(folder / "validation_weights" / f"eval{i:03d}.json")
            saved = MaskablePPO.load(binding["path"], device="cpu")
            if (
                evaluation.file_digest(binding["path"]) != binding["sha256"]
                or policy_digest(saved.policy) != binding["policy_sha256"]
                or saved.num_timesteps != h["timesteps"]
                or binding["evaluation_file"] != h["evaluation_file"]
            ):
                raise ValueError("Validation weights do not match binding")
        for kind, index in (("best", best_index), ("final", len(history) - 1)):
            path = folder / f"{kind}_model.zip"
            model = MaskablePPO.load(str(path), device="cpu")
            binding = read(folder / "validation_weights" / f"eval{index:03d}.json")
            if policy_digest(model.policy) != binding["policy_sha256"]:
                raise ValueError("Final/selected model not the evaluated weights")
            if kind == "final" and model.num_timesteps != p["steps_per_model"]:
                raise ValueError("PPO budget incomplete")
            models[f"{name}-{kind}"] = bind_model(path, folder / history[index]["evaluation_file"])
    for seed in p["seeds"]:
        row = read(output / f"clone-seed{seed}-complete.json")
        scratch = read(output / f"scratch-seed{seed}/validation_0_eval000.json")
        bc = read(output / f"bc-seed{seed}/validation_0_eval000.json")
        compare_reload(scratch, read(output / f"clone-seed{seed}-initial-validation.json"))
        compare_reload(bc, read(output / f"clone-seed{seed}-bc-validation.json"))
        models[f"bc-only-seed{seed}"] = bind_model(
            Path(row["selected"]["path"]), output / f"clone-seed{seed}-bc-validation.json"
        )
    old = output / "reference" / "best_model.zip"
    old_validation = output / "reference-validation.json"
    save_new(
        old_validation,
        evaluation.evaluate_suite(
            MaskablePPO.load(str(old), device="cpu"),
            p["paths"]["validation"],
            episodes=100,
            seed=VAL_SEED,
        ),
    )
    models["historical-best"] = bind_model(old, old_validation)
    save_new(
        output / "selection.json",
        {
            "models": models,
            "protocol_sha256": evaluation.file_digest(output / "protocol.json"),
            "primary_endpoint": "final",
            "test_results_used_for_selection": False,
        },
    )


def bind_model(path, validation):
    return {
        "path": str(path),
        "sha256": evaluation.file_digest(path),
        "validation_path": str(validation),
        "validation_sha256": evaluation.file_digest(validation),
    }


def frozen(output):
    p, selected = checked(output), read(output / "selection.json")
    if selected["protocol_sha256"] != evaluation.file_digest(output / "protocol.json"):
        raise ValueError("Selection protocol mismatch")
    if len(selected["models"]) != 16:
        raise ValueError("Missing comparisons")
    for item in selected["models"].values():
        if (
            evaluation.file_digest(item["path"]) != item["sha256"]
            or evaluation.file_digest(item["validation_path"]) != item["validation_sha256"]
        ):
            raise ValueError("Frozen comparison changed")
    return p, selected


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selected = frozen(output)
    if stage == "test":
        require_reloads(output, selected)
    item = selected["models"][name]
    split, count, seed = (
        ("validation", 100, VAL_SEED) if stage == "reload" else ("test", 200, TEST_SEED)
    )
    save_new(output / f"{stage}-{name}-started.json", {"sha256": item["sha256"]})
    result = evaluation.evaluate_suite(
        MaskablePPO.load(item["path"], device="cpu"), p["paths"][split], episodes=count, seed=seed
    )
    if stage == "reload":
        compare_reload(result, read(item["validation_path"]))
    result.update(
        model_sha256=item["sha256"],
        selection_sha256=evaluation.file_digest(output / "selection.json"),
    )
    result["opponent_fallback_coverage"] = {
        family: fallback_counts(
            result["families"][family]["episode_results"],
            {x["turn"] for x in read(path)["snapshots"]},
        )
        for family, path in p["paths"][split].items()
    }
    frozen(output)
    save_new(output / f"{stage}-{name}.json", result)


def require_reloads(output, selected):
    for name, item in selected["models"].items():
        result = read(output / f"reload-{name}.json")
        if result["model_sha256"] != item["sha256"] or result[
            "selection_sha256"
        ] != evaluation.file_digest(output / "selection.json"):
            raise ValueError("Reload binding mismatch")
        compare_reload(result, read(item["validation_path"]))


def summarize(output):
    p, selected = frozen(output)
    require_reloads(output, selected)
    results = {}
    for name, item in selected["models"].items():
        result = read(output / f"test-{name}.json")
        if result["model_sha256"] != item["sha256"] or result[
            "selection_sha256"
        ] != evaluation.file_digest(output / "selection.json"):
            raise ValueError("Test binding mismatch")
        results[name] = metrics(result)
        results[name]["learned_success"] = fmean(
            result["families"][f]["success_rate"] for f in ("fresh_stats", "fresh_mix")
        )
        results[name]["action_counts"] = {
            f: data["action_counts"] for f, data in result["families"].items()
        }
    pairs = {
        str(s): {
            "bc_minus_scratch_learned_success": results[f"bc-seed{s}-final"]["learned_success"]
            - results[f"scratch-seed{s}-final"]["learned_success"],
            "ppo_minus_bc_only_learned_success": results[f"bc-seed{s}-final"]["learned_success"]
            - results[f"bc-only-seed{s}"]["learned_success"],
        }
        for s in p["seeds"]
    }
    save_new(
        output / "summary.json",
        {
            "results": results,
            "pairs": pairs,
            "all_training_complete": True,
            "ppo_steps": 6 * p["steps_per_model"],
            "all_validation_reloads_exact": True,
            "reload_episodes": 6400,
            "test_episodes": 16000,
            "previous_best_preserved": True,
            "recipe_confirmed": False,
            "limitations": p["limitations"],
            "replay_audit_pending": True,
        },
    )


def batch(output, stage, names, deadline):
    for start in range(0, len(names), 2):
        children = []
        try:
            for name in names[start : start + 2]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Pipeline deadline reached before starting next worker")
                print(f"Starting {stage}: {name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "sap_rl_lab.imitation_experiment",
                            stage,
                            "--output",
                            str(output),
                            "--name",
                            str(name),
                        ],
                        cwd=ROOT,
                    )
                )
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError(f"{stage} worker failed; no automatic retry")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Six-hour pipeline limit reached; keep partial results")
                time.sleep(1)
            if any(c.returncode != 0 for c in children):
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


def pipeline(output):
    p = checked(output)
    save_new(output / "pipeline_started.json", {"time_unix": time.time()})
    started = time.monotonic()
    deadline = started + p["maximum_pipeline_seconds"]
    try:
        batch(output, "demos", ["all"], deadline)
        batch(output, "clone", p["seeds"], deadline)
        if not gate(output):
            save_new(
                output / "stopped_at_bc_gate.json",
                {
                    "reason": "Closed-loop BC competence screen failed; do not spend PPO budget",
                    "ppo_steps": 0,
                    "recipe_confirmed": False,
                },
            )
            batch(output, "audit", ["all"], deadline)
            return
        batch(output, "arm", list(p["arms"]), deadline)
        batch(output, "freeze", ["all"], deadline)
        names = list(read(output / "selection.json")["models"])
        batch(output, "reload", names, deadline)
        batch(output, "test", names, deadline)
        summarize(output)
        batch(output, "audit", ["all"], deadline)
        save_new(output / "pipeline_complete.json", {"elapsed_seconds": time.monotonic() - started})
    except BaseException as exc:
        save_new(
            output / "pipeline_failed.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "pipeline",
            "demos",
            "clone",
            "arm",
            "freeze",
            "reload",
            "test",
            "summarize",
            "audit",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in ("prepare", "pipeline", "freeze", "summarize"):
        globals()[args.stage](output)
    elif args.stage == "demos":
        demonstrations(output)
    elif args.stage == "clone":
        clone(output, int(args.name))
    elif args.stage == "arm":
        arm(output, args.name)
    elif args.stage == "audit":
        sys.path.insert(0, str(ROOT / "scripts"))
        from audit_imitation import run

        run(output, output / "audit")
    else:
        inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
