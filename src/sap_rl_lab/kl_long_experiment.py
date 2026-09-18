"""Fresh-sampling-seed, longer PPO/KL replication; historical runs stay immutable."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median

from . import evaluation
from .expanded import verify_inputs
from .expanded_confirmation import compare_reload, read
from .exploration_pilot import metrics
from .historical_confirmation import copy_checked, fallback_counts
from .imitation_experiment import OLD_BEST, OLD_SHA, bind_model, require_reloads
from .imitation_experiment import checked as check_parent
from .midgame import ROOT
from .ppo_guardrails import GuardrailConfig, ValidationGuard, train_guarded, write_new
from .round3 import source_archive
from .training import TrainingConfig, policy_digest

SEEDS = (17301, 204101, 204201)
STEPS, INTERVAL = 4_194_304, 524_288
TRAIN_SEEDS = (97301, 97302, 97303)
VAL_SEED, TEST_SEED = 35_000_000, 36_000_000
PLAN = ROOT / "docs/KL_LONG_PLAN.md"
PARENT = ROOT / "runs/imitation-v1"
HELPERS = (
    "scripts/audit_kl_long.py",
    "scripts/audit_exploration_pilot.py",
    "scripts/audit_midgame_confirmation.py",
    "scripts/inspect_confirmation_delivery.py",
    "scripts/summarize_policy_replays.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def checked(output):
    p = read(output / "protocol.json")
    if evaluation.file_digest(output / "protocol.json") != read(output / "seal.json")["sha256"]:
        raise ValueError("KL protocol changed")
    verify_inputs(output, p)
    if evaluation.file_digest(PLAN) != p["plan_sha256"]:
        raise ValueError("KL plan changed")
    if evaluation.file_digest(OLD_BEST) != OLD_SHA:
        raise ValueError("Historical best changed")
    for path, digest in p["external_sha256"].items():
        if evaluation.file_digest(path) != digest:
            raise ValueError(f"Frozen external input changed: {path}")
    return p


def prepare(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    prior = check_parent(PARENT)
    output.mkdir(parents=True, exist_ok=False)
    paths, starts, external = {}, {}, {}
    for split in ("train", "validation", "test"):
        paths[split] = {}
        for family, source in prior["paths"][split].items():
            target = output / "data" / split / f"{family}.json"
            copy_checked(source, target)
            paths[split][family] = str(target)
    for seed in SEEDS:
        binding_path = PARENT / f"bc-seed{seed}/validation_weights/eval003.json"
        item = read(binding_path)
        model = MaskablePPO.load(item["path"], device="cpu")
        if (
            evaluation.file_digest(item["path"]) != item["sha256"]
            or policy_digest(model.policy) != item["policy_sha256"]
            or model.num_timesteps != 3_145_728
            or item["timesteps"] != 3_145_728
        ):
            raise ValueError("Starting checkpoint binding differs")
        target = output / "reference" / f"initial-seed{seed}.zip"
        copy_checked(item["path"], target)
        starts[str(seed)] = {**item, "path": str(target), "source_path": item["path"]}
        external[str(binding_path)] = evaluation.file_digest(binding_path)
        external[item["path"]] = item["sha256"]
    copy_checked(OLD_BEST, output / "reference/best_model.zip")
    base = {
        **prior["base_config"],
        "timesteps": STEPS,
        "evaluation_interval": INTERVAL,
        "validation_seed": VAL_SEED,
        "validation_episodes": 100,
        "opponent_leagues": tuple(paths["train"].values()),
        "validation_leagues": paths["validation"],
    }
    arms = {}
    for index, seed in enumerate(SEEDS):
        for arm, kl in (("control", None), ("kl", 0.01)):
            name = f"{arm}-seed{seed}"
            config = TrainingConfig(
                **{
                    **base,
                    "seed": TRAIN_SEEDS[index],
                    "output_dir": str(output / name),
                    "initialize_from": starts[str(seed)]["path"],
                    "expected_initial_policy_sha256": starts[str(seed)]["policy_sha256"],
                }
            )
            config.validate()
            guard = GuardrailConfig(target_kl=kl, stop_on_regression=False, max_seconds=3600)
            guard.validate()
            arms[name] = {
                "source_seed": seed,
                "training": asdict(config),
                "guardrails": asdict(guard),
            }
    external.update({str(ROOT / path): evaluation.file_digest(ROOT / path) for path in HELPERS})
    external[str(PARENT / "protocol.json")] = evaluation.file_digest(PARENT / "protocol.json")
    write_new(
        output / "protocol.json",
        {
            "arms": arms,
            "seeds": SEEDS,
            "starts": starts,
            "paths": paths,
            "steps_per_model": STEPS,
            "maximum_ppo_steps": 6 * STEPS,
            "validation_records_per_model": STEPS // INTERVAL + 1,
            "training_sampling_seeds": TRAIN_SEEDS,
            "midpoint_steps": STEPS // 2,
            "maximum_comparison_models": 22,
            "maximum_reload_episodes": 8800,
            "maximum_test_episodes": 22000,
            "validation_seed": VAL_SEED,
            "validation_episodes_per_family": 100,
            "test_seed": TEST_SEED,
            "test_episodes_per_family": 200,
            "maximum_worker_processes": 2,
            "maximum_pipeline_seconds": 4 * 3600,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(x.relative_to(output)): evaluation.file_digest(x)
                for folder in ("data", "reference")
                for x in sorted((output / folder).rglob("*"))
                if x.is_file()
            },
            "external_sha256": external,
            "plan_sha256": copy_checked(PLAN, output / "plan.md"),
            "primary_endpoint": "fixed-budget final; midpoint and validation-selected secondary",
            "test_metrics_used_for_selection": False,
            "automatic_recipe_confirmation": False,
            "limitations": "Three reused, retrospectively selected healthy BC-PPO checkpoints; "
            "continuation sampling seeds, NOT new BC initializations or an independent ecology. "
            "Longer fixed decision budget, not matched gradient-update budget. Fresh "
            "optimizer and sampling RNG, not exact historical continuation. Previously studied "
            "opponent ecology with fresh evaluation seeds. 40 pets, normal shops Tier 1-3 and "
            "Tier 4 level-up rewards only. No automatic delivery or significance claim.",
        },
    )
    write_new(output / "seal.json", {"sha256": evaluation.file_digest(output / "protocol.json")})
    checked(output)


def arm(output, name):
    p = checked(output)
    payload = p["arms"][name]
    with (output / f"{name}.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
        result = train_guarded(
            TrainingConfig(**payload["training"]), GuardrailConfig(**payload["guardrails"])
        )
    folder = output / name
    manifest = read(folder / "manifest.json")
    # Adapter for the unchanged historical replay inspector, only in this new run.
    write_new(
        folder / "run_manifest.json",
        {
            "config": manifest["training"],
            "environment_contract": manifest["environment_contract"],
            "guardrails": manifest["guardrails"],
        },
    )
    checked(output)
    if result["stop_reason"] != "budget_complete" or result["actual_timesteps"] != STEPS:
        raise ValueError(f"{name} did not finish its fixed budget: {result['stop_reason']}")
    print(f"Completed {name}: {result['actual_timesteps']} decisions", flush=True)


def update_metrics(rows, target_kl, steps=STEPS, rollout=2048, maximum=80):
    if [r["timesteps"] for r in rows] != list(range(rollout, steps + 1, rollout)):
        raise ValueError("Missing or duplicated post-update telemetry")
    for row in rows:
        if (
            row["rollout_decisions"] != rollout
            or row["target_kl"] != target_kl
            or row["maximum_optimizer_steps"] != maximum
            or not 0 <= row["optimizer_steps"] <= maximum
            or (target_kl is None and row["optimizer_steps"] != maximum)
            or row["kl_early_stopped"] != (row["optimizer_steps"] < maximum)
        ):
            raise ValueError("Actual optimizer accounting differs from protocol")
    return {
        "updates": len(rows),
        "optimizer_steps": sum(r["optimizer_steps"] for r in rows),
        "maximum_optimizer_steps": maximum * len(rows),
        "early_stopped_rollouts": sum(r["kl_early_stopped"] for r in rows),
        "exact_rollout_kl_median": median(r["exact_rollout_kl_mean"] for r in rows),
        "exact_rollout_kl_max": max(r["exact_rollout_kl_mean"] for r in rows),
        "entropy_median": median(r["entropy_before_update"] for r in rows),
    }


def validate_pair(left, right):
    a, b = dict(left["training"]), dict(right["training"])
    a.pop("output_dir")
    b.pop("output_dir")
    ga, gb = dict(left["guardrails"]), dict(right["guardrails"])
    if ga.pop("target_kl") is not None or gb.pop("target_kl") != 0.01 or a != b or ga != gb:
        raise ValueError("Paired arms differ in more than KL/output directory")


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, models, telemetry, missing = checked(output), {}, {}, []
    for name, payload in p["arms"].items():
        folder = output / name
        manifest, done = read(folder / "manifest.json"), read(folder / "complete.json")
        if (
            manifest["training"] != payload["training"]
            or manifest["guardrails"] != payload["guardrails"]
            or manifest["initial_policy_sha256"]
            != payload["training"]["expected_initial_policy_sha256"]
            or manifest["initialization"] != "weights_only_fresh_optimizer"
            or done["stop_reason"] != "budget_complete"
            or done["actual_timesteps"] != STEPS
        ):
            raise ValueError("Incomplete or non-protocol model")
        rows = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        telemetry[name] = update_metrics(rows, payload["guardrails"]["target_kl"])
        history = done["evaluations"]
        if [h["checkpoint"]["timesteps"] for h in history] != list(range(0, STEPS + 1, INTERVAL)):
            raise ValueError("Incomplete validation curve")
        guard, records = ValidationGuard(GuardrailConfig(**payload["guardrails"])), {}
        for h in history:
            cp = h["checkpoint"]
            record = read(Path(cp["path"]).with_suffix(".json"))
            saved = MaskablePPO.load(cp["path"], device="cpu")
            if (
                evaluation.file_digest(cp["path"]) != cp["sha256"]
                or policy_digest(saved.policy) != cp["policy_sha256"]
                or saved.num_timesteps != cp["timesteps"]
                or record["checkpoint"] != cp
                or evaluation.compact_evaluation(record["evaluation"]) != h["evaluation"]
                or canonical(guard.consider(record["evaluation"], cp)) != canonical(h["decision"])
                or record["decision"] != h["decision"]
            ):
                raise ValueError("Validation weights, decision, or result binding differs")
            records[cp["path"]] = record["evaluation"]
        if canonical(guard.best) != canonical(done["selected"]):
            raise ValueError("Selection cannot be reproduced from validation")
        final = MaskablePPO.load(done["last_model"], device="cpu")
        if (
            evaluation.file_digest(done["last_model"]) != done["last_sha256"]
            or final.num_timesteps != STEPS
            or policy_digest(final.policy) != history[-1]["checkpoint"]["policy_sha256"]
        ):
            raise ValueError("Final weights are not the final evaluated weights")
        midpoint = next(
            h["checkpoint"] for h in history if h["checkpoint"]["timesteps"] == STEPS // 2
        )
        candidates = {
            "final": (done["last_model"], records[history[-1]["checkpoint"]["path"]]),
            "midpoint": (midpoint["path"], records[midpoint["path"]]),
        }
        if guard.best is None:
            missing.append(name)
        else:
            path = guard.best["checkpoint"]["path"]
            candidates["selected"] = (path, records[path])
        for kind, (path, result) in candidates.items():
            validation = output / f"validation-{name}-{kind}.json"
            write_new(validation, result)
            models[f"{name}-{kind}"] = bind_model(Path(path), validation)
    for seed in SEEDS:
        control, kl = f"control-seed{seed}", f"kl-seed{seed}"
        validate_pair(p["arms"][control], p["arms"][kl])
        initial = [read(output / n / "eval000-step0.json") for n in (control, kl)]
        compare_reload(*(r["evaluation"] for r in initial))
        expected = p["starts"][str(seed)]["policy_sha256"]
        if any(r["checkpoint"]["policy_sha256"] != expected for r in initial):
            raise ValueError("Pair initial tensors differ")
        validation = output / f"validation-initial-seed{seed}.json"
        write_new(validation, initial[0]["evaluation"])
        models[f"initial-seed{seed}"] = bind_model(Path(p["starts"][str(seed)]["path"]), validation)
    old = output / "reference/best_model.zip"
    validation = output / "validation-historical-best.json"
    write_new(
        validation,
        evaluation.evaluate_suite(
            MaskablePPO.load(str(old), device="cpu"),
            p["paths"]["validation"],
            episodes=100,
            seed=VAL_SEED,
        ),
    )
    models["historical-best"] = bind_model(old, validation)
    write_new(
        output / "selection.json",
        {
            "models": models,
            "missing_eligible": missing,
            "protocol_sha256": evaluation.file_digest(output / "protocol.json"),
            "telemetry": telemetry,
            "all_validation_bindings_exact": True,
            "paired_initial_weights_and_rows_exact": True,
            "test_results_used_for_selection": False,
        },
    )
    write_new(
        output / "selection_seal.json",
        {"sha256": evaluation.file_digest(output / "selection.json")},
    )


def frozen(output):
    p, selected = checked(output), read(output / "selection.json")
    if evaluation.file_digest(output / "selection.json") != read(output / "selection_seal.json")[
        "sha256"
    ] or selected["protocol_sha256"] != evaluation.file_digest(output / "protocol.json"):
        raise ValueError("Frozen selection changed")
    expected = {f"{name}-{kind}" for name in p["arms"] for kind in ("final", "midpoint")} | {
        f"initial-seed{s}" for s in SEEDS
    }
    expected |= {"historical-best"} | {
        f"{name}-selected" for name in p["arms"] if name not in selected["missing_eligible"]
    }
    if set(selected["models"]) != expected:
        raise ValueError("Frozen comparison is incomplete")
    for item in selected["models"].values():
        if (
            evaluation.file_digest(item["path"]) != item["sha256"]
            or evaluation.file_digest(item["validation_path"]) != item["validation_sha256"]
        ):
            raise ValueError("Frozen model/result changed")
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
    write_new(output / f"{stage}-{name}-started.json", {"sha256": item["sha256"]})
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
    write_new(output / f"{stage}-{name}.json", result)


def summarize(output):
    p, selected = frozen(output)
    require_reloads(output, selected)
    results = {}
    for name, item in selected["models"].items():
        result = read(output / f"test-{name}.json")
        if result["model_sha256"] != item["sha256"] or result[
            "selection_sha256"
        ] != evaluation.file_digest(output / "selection.json"):
            raise ValueError("Test binding differs")
        results[name] = {
            **metrics(result),
            "forced_episode_rate": result["forced_episode_rate"],
            "learned_success": fmean(
                result["families"][f]["success_rate"] for f in ("fresh_stats", "fresh_mix")
            ),
            "action_counts": {f: r["action_counts"] for f, r in result["families"].items()},
            "mean_unspent_gold_per_battle": fmean(
                r["mean_unspent_gold_per_battle"] for r in result["families"].values()
            ),
        }
    pairs = {}
    for seed in SEEDS:
        pairs[str(seed)] = {}
        for kind in ("final", "midpoint", "selected"):
            left, right = f"control-seed{seed}-{kind}", f"kl-seed{seed}-{kind}"
            pairs[str(seed)][kind] = (
                None
                if left not in results or right not in results
                else {
                    f"kl_minus_control_{key}": results[right][key] - results[left][key]
                    for key in (
                        "learned_success",
                        "forced_episode_rate",
                        "worst_family_forced_episode_rate",
                        "no_purchase_zero_win_loss_rate",
                    )
                }
            )
    late_changes = {
        name: {
            key: results[f"{name}-final"][key] - results[f"{name}-midpoint"][key]
            for key in ("learned_success", "forced_episode_rate")
        }
        for name in p["arms"]
    }
    write_new(
        output / "summary.json",
        {
            "results": results,
            "pairs": pairs,
            "final_minus_midpoint": late_changes,
            "telemetry": selected["telemetry"],
            "ppo_steps": 6 * STEPS,
            "all_training_complete": True,
            "all_validation_reloads_exact": True,
            "reload_episodes": len(results) * 400,
            "test_episodes": len(results) * 1000,
            "previous_best_preserved": True,
            "recipe_confirmed": False,
            "limitations": p["limitations"],
            "replay_audit_pending": True,
        },
    )


def batch(output, stage, names, deadline):
    for start in range(0, len(names), 2):
        children, logs = [], []
        try:
            for name in names[start : start + 2]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Long-horizon deadline reached")
                log = (output / f"worker-{stage}-{name}.log").open("x")
                logs.append(log)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        "-m",
                        "sap_rl_lab.kl_long_experiment",
                        stage,
                        "--output",
                        str(output),
                        "--name",
                        name,
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                children.append((name, child))
                write_new(
                    output / f"worker-{stage}-{name}.json",
                    {"pid": child.pid, "time_unix": time.time()},
                )
            while any(c.poll() is None for _, c in children):
                if any(c.poll() not in (None, 0) for _, c in children):
                    raise RuntimeError(f"Long-horizon {stage} failed; inspect durable worker logs")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Four-hour long-horizon limit reached")
                time.sleep(1)
            if any(c.returncode != 0 for _, c in children):
                raise RuntimeError(f"Long-horizon {stage} failed")
        finally:
            for name, child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                write_new(
                    output / f"worker-{stage}-{name}-exit.json", {"returncode": child.returncode}
                )
            for log in logs:
                log.close()


def pipeline(output):
    p = checked(output)
    write_new(output / "pipeline_started.json", {"time_unix": time.time()})
    started = time.monotonic()
    deadline = started + p["maximum_pipeline_seconds"]
    try:
        batch(output, "arm", list(p["arms"]), deadline)
        batch(output, "freeze", ["all"], deadline)
        names = list(read(output / "selection.json")["models"])
        batch(output, "reload", names, deadline)
        batch(output, "test", names, deadline)
        batch(output, "summarize", ["all"], deadline)
        batch(output, "audit", ["all"], deadline)
        checked(output)
        check_parent(PARENT)
        write_new(
            output / "pipeline_complete.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "training_evaluation_and_audit_complete": True,
            },
        )
    except BaseException as exc:
        write_new(
            output / "pipeline_failed.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "pipeline", "arm", "freeze", "reload", "test", "summarize", "audit"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "pipeline":
        with (output / "pipeline.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            pipeline(output)
    elif args.stage in ("prepare", "freeze", "summarize"):
        globals()[args.stage](output)
    elif args.stage == "arm":
        arm(output, args.name)
    elif args.stage == "audit":
        sys.path.insert(0, str(ROOT / "scripts"))
        from audit_kl_long import run

        run(output, output / "audit")
    else:
        inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
