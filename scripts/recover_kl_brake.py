"""Evaluation-only amendment for the interrupted KL run; never resumes training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from statistics import fmean

from audit_exploration_pilot import choose_cases
from audit_midgame_confirmation import check_environment_contract, exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab import evaluation
from sap_rl_lab.expanded import verify_inputs
from sap_rl_lab.expanded_confirmation import compare_reload, read
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.exploration_pilot import metrics
from sap_rl_lab.historical_confirmation import copy_checked, fallback_counts
from sap_rl_lab.imitation_experiment import bind_model, require_reloads
from sap_rl_lab.kl_brake_experiment import (
    INTERVAL,
    ROOT,
    SEEDS,
    TEST_SEED,
    VAL_SEED,
    canonical,
    update_metrics,
    validate_pair,
)
from sap_rl_lab.kl_brake_experiment import checked as check_parent
from sap_rl_lab.ppo_guardrails import GuardrailConfig, ValidationGuard, write_new
from sap_rl_lab.round3 import source_archive
from sap_rl_lab.training import policy_digest

PARENT = ROOT / "runs/kl-brake-v1"
PLAN = ROOT / "docs/KL_BRAKE_RECOVERY_PLAN.md"


def common_checkpoint_step(histories):
    if len(histories) != 6:
        raise ValueError("All six checkpoint histories are required")
    for steps in histories:
        if not steps or steps != list(range(0, max(steps) + 1, INTERVAL)):
            raise ValueError("Non-contiguous checkpoint history")
    common = set.intersection(*(set(steps) for steps in histories))
    if not common or max(common) <= 0:
        raise ValueError("No positive common checkpoint")
    return max(common)


def checked(output):
    p = read(output / "protocol.json")
    if evaluation.file_digest(output / "protocol.json") != read(output / "seal.json")["sha256"]:
        raise ValueError("Recovery protocol changed")
    verify_inputs(output, p)
    check_parent(PARENT)
    for path, sha in p["external_sha256"].items():
        if evaluation.file_digest(path) != sha:
            raise ValueError(f"Recovery input changed: {path}")
    return p


def prepare(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    prior = check_parent(PARENT)
    if not (PARENT / "pipeline_failed.json").exists():
        raise ValueError("Recovery requires the recorded interrupted original")
    if list(PARENT.glob("test-*-started.json")) or (PARENT / "selection.json").exists():
        raise ValueError("Original selection/test already started; amendment not applicable")
    histories = {
        n: [read(x) for x in sorted((PARENT / n).glob("eval*-step*.json"))] for n in prior["arms"]
    }
    step = common_checkpoint_step(
        [[r["checkpoint"]["timesteps"] for r in h] for h in histories.values()]
    )
    if step != 1_835_008:
        raise ValueError("Available common checkpoint differs from disclosed amendment")
    output.mkdir(parents=True, exist_ok=False)
    external = {
        str(PLAN): evaluation.file_digest(PLAN),
        str(Path(__file__).resolve()): evaluation.file_digest(__file__),
        str(PARENT / "pipeline_failed.json"): evaluation.file_digest(
            PARENT / "pipeline_failed.json"
        ),
    }
    paths = {}
    for split in ("validation", "test"):
        paths[split] = {}
        for family, source in prior["paths"][split].items():
            target = output / "data" / split / f"{family}.json"
            copy_checked(source, target)
            paths[split][family] = str(target)
    models, telemetry, curves, missing = {}, {}, {}, []
    for name, payload in prior["arms"].items():
        source_folder, folder = PARENT / name, output / name
        manifest_path = source_folder / "manifest.json"
        manifest = read(manifest_path)
        if (
            manifest["training"] != payload["training"]
            or manifest["guardrails"] != payload["guardrails"]
            or manifest["initialization"] != "weights_only_fresh_optimizer"
            or manifest["initial_policy_sha256"]
            != payload["training"]["expected_initial_policy_sha256"]
        ):
            raise ValueError("Initial recipe differs")
        update_path = source_folder / "updates.jsonl"
        updates = [json.loads(line) for line in update_path.read_text().splitlines()]
        telemetry[name] = update_metrics(
            [r for r in updates if r["timesteps"] <= step],
            payload["guardrails"]["target_kl"],
            steps=step,
        )
        external[str(manifest_path)] = evaluation.file_digest(manifest_path)
        external[str(update_path)] = evaluation.file_digest(update_path)
        guard = ValidationGuard(GuardrailConfig(**payload["guardrails"]))
        records, curves[name] = {}, []
        for record in histories[name]:
            cp = record["checkpoint"]
            if cp["timesteps"] > step:
                continue
            path = Path(cp["path"])
            model = MaskablePPO.load(str(path), device="cpu")
            if (
                evaluation.file_digest(path) != cp["sha256"]
                or model.num_timesteps != cp["timesteps"]
                or policy_digest(model.policy) != cp["policy_sha256"]
                or canonical(guard.consider(record["evaluation"], cp))
                != canonical(record["decision"])
            ):
                raise ValueError("Prefix weight/decision binding differs")
            if set(record["evaluation"]["families"]) != set(paths["validation"]):
                raise ValueError("Prefix validation families differ")
            for family, data in record["evaluation"]["families"].items():
                if (not data["deterministic"] or data["league_sha256"]
                        != evaluation.file_digest(paths["validation"][family])):
                    raise ValueError("Prefix validation pool differs")
                check_environment_contract(data["environment_contract"],
                                           manifest["environment_contract"])
                exact_family_counts(data, 100, VAL_SEED)
            external[str(path)] = cp["sha256"]
            external[str(path.with_suffix(".json"))] = evaluation.file_digest(
                path.with_suffix(".json")
            )
            records[str(path)] = record["evaluation"]
            curves[name].append(
                {
                    "checkpoint": cp,
                    "decision": record["decision"],
                    "evaluation": evaluation.compact_evaluation(record["evaluation"]),
                }
            )
        common = curves[name][-1]["checkpoint"]
        candidates = {"common": common}
        if guard.best is not None:
            candidates["selected"] = guard.best["checkpoint"]
        else:
            missing.append(name)
        folder.mkdir(parents=True)
        write_new(
            folder / "run_manifest.json",
            {
                "config": payload["training"],
                "environment_contract": manifest["environment_contract"],
                "role": "evaluation-only copied checkpoint; no training in this directory",
            },
        )
        for kind, cp in candidates.items():
            path = folder / f"{kind}.zip"
            copy_checked(cp["path"], path)
            validation = folder / f"validation-{kind}.json"
            write_new(validation, records[cp["path"]])
            models[f"{name}-{kind}"] = {
                **bind_model(path, validation),
                "checkpoint_timesteps": cp["timesteps"],
            }
    for seed in SEEDS:
        names = [f"{arm}-seed{seed}" for arm in ("control", "kl")]
        validate_pair(*(prior["arms"][n] for n in names))
        compare_reload(*(histories[n][0]["evaluation"] for n in names))
        expected = prior["starts"][str(seed)]["policy_sha256"]
        if any(histories[n][0]["checkpoint"]["policy_sha256"] != expected for n in names):
            raise ValueError("Paired initial weights differ")
        path = output / "reference" / f"initial-seed{seed}.zip"
        copy_checked(prior["starts"][str(seed)]["path"], path)
        validation = output / "reference" / f"initial-seed{seed}-validation.json"
        write_new(validation, histories[names[0]][0]["evaluation"])
        models[f"initial-seed{seed}"] = bind_model(path, validation)
    copy_checked(PARENT / "reference/best_model.zip", output / "reference/best_model.zip")
    copy_checked(PLAN, output / "plan.md")
    write_new(output / "curves.json", curves)
    write_new(
        output / "protocol.json",
        {
            "paths": paths,
            "arms": prior["arms"],
            "models_without_historical_reference": models,
            "missing_eligible": missing,
            "telemetry_common_prefix": telemetry,
            "common_steps": step,
            "new_training_steps": 0,
            "original_fixed_budget_complete": False,
            "maximum_pipeline_seconds": 7200,
            "maximum_workers": 2,
            "source_files_sha256": source_archive(output),
            "external_sha256": external,
            "data_sha256": {
                str(x.relative_to(output)): evaluation.file_digest(x)
                for x in output.rglob("*")
                if x.is_file() and x.name != "source.zip"
            },
            "limitations": "Post-interruption amendment after validation inspection: largest "
            "common saved prefix, not completed original preregistration. Known opponent ecology, "
            "fresh episode seeds, three retrospective starts; no automatic recipe confirmation.",
        },
    )
    write_new(output / "seal.json", {"sha256": evaluation.file_digest(output / "protocol.json")})
    checked(output)


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = checked(output)
    models = dict(p["models_without_historical_reference"])
    path = output / "reference/best_model.zip"
    validation = output / "reference/historical-validation.json"
    write_new(
        validation,
        evaluation.evaluate_suite(
            MaskablePPO.load(str(path), device="cpu"),
            p["paths"]["validation"],
            episodes=100,
            seed=VAL_SEED,
        ),
    )
    models["historical-best"] = bind_model(path, validation)
    write_new(
        output / "selection.json",
        {
            "models": models,
            "protocol_sha256": evaluation.file_digest(output / "protocol.json"),
            "test_results_used_for_selection": False,
            "primary": "common",
            "common_steps": p["common_steps"],
            "post_interruption_amendment": True,
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
        raise ValueError("Selection changed")
    expected = set(p["models_without_historical_reference"]) | {"historical-best"}
    if set(selected["models"]) != expected:
        raise ValueError("Missing comparison")
    for item in selected["models"].values():
        if (
            evaluation.file_digest(item["path"]) != item["sha256"]
            or evaluation.file_digest(item["validation_path"]) != item["validation_sha256"]
        ):
            raise ValueError("Frozen weights or validation changed")
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
    write_new(
        output / f"{stage}-{name}-started.json",
        {
            "model_sha256": item["sha256"],
            "selection_sha256": evaluation.file_digest(output / "selection.json"),
        },
    )
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
        f: fallback_counts(
            result["families"][f]["episode_results"], {r["turn"] for r in read(path)["snapshots"]}
        )
        for f, path in p["paths"][split].items()
    }
    frozen(output)
    write_new(output / f"{stage}-{name}.json", result)


def audit(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selected = frozen(output)
    require_reloads(output, selected)
    folder = output / "audit"
    folder.mkdir(exist_ok=False)
    counts, cases, results = {}, [], {}

    def recount(result, split, contract):
        if set(result["families"]) != set(p["paths"][split]):
            raise ValueError("Missing families")
        count, seed = (100, VAL_SEED) if split == "validation" else (200, TEST_SEED)
        values, fallback = {}, {}
        for f, data in result["families"].items():
            pool = p["paths"][split][f]
            if not data["deterministic"] or data["league_sha256"] != evaluation.file_digest(pool):
                raise ValueError("Incorrect evaluation pool/semantics")
            check_environment_contract(data["environment_contract"], contract)
            values[f] = exact_family_counts(data, count, seed)
            fallback[f] = fallback_counts(
                data["episode_results"], {r["turn"] for r in read(pool)["snapshots"]}
            )
        if (
            "opponent_fallback_coverage" in result
            and result["opponent_fallback_coverage"] != fallback
        ):
            raise ValueError("Fallback recount differs")
        return values

    for name, history in read(output / "curves.json").items():
        contract = read(output / name / "run_manifest.json")["environment_contract"]
        for i, row in enumerate(history):
            result = read(Path(row["checkpoint"]["path"]).with_suffix(".json"))["evaluation"]
            counts[f"curve/{name}/{i}"] = recount(result, "validation", contract)
    for name, item in selected["models"].items():
        contract = MaskablePPO.load(item["path"], device="cpu").sap_environment_contract
        for stage, split in (("reload", "validation"), ("test", "test")):
            result = read(output / f"{stage}-{name}.json")
            if result["model_sha256"] != item["sha256"] or result[
                "selection_sha256"
            ] != evaluation.file_digest(output / "selection.json"):
                raise ValueError("Result binding differs")
            counts[f"{stage}/{name}"] = recount(result, split, contract)
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
            "fallback_coverage": result["opponent_fallback_coverage"],
        }
        if name.endswith("-common"):
            for case in choose_cases(read(item["validation_path"])["families"]):
                cases.append({"name": name, "path": item["path"], "sha256": item["sha256"], **case})
    write_new(folder / "counts.json", counts)
    write_new(
        folder / "replay_plan.json",
        {
            "cases": cases,
            "selection": "Earliest loss "
            "(prefer no purchase), forced/truncated, success per common checkpoint; deduplicate.",
        },
    )
    replays = []
    for case in cases:
        row, family = case["expected"], case["family"]
        label = f"{case['name']}-{family}-{row['seed']}"
        with (folder / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(case["path"], p["paths"]["validation"][family], folder / label, 1, row["seed"])
        replay = read(folder / label / f"episode-{row['seed']}.json")
        compare_replay(replay["summary"], row)
        for s in replay["steps"]:
            if abs(s["objective_reward"] - (s["info"]["game_reward"] - 0.005)) > 1e-10:
                raise ValueError("Replay reward differs")
        replays.append(
            {
                "name": case["name"],
                "directory": label,
                "summary": replay["summary"],
                "behavior": summarize_episode(replay, 1.0),
                "values_are_checkpoint_reconstructions": True,
            }
        )
    frozen(output)
    write_new(
        folder / "summary.json",
        {
            "all_replays_exact": True,
            "all_counts_recomputed": True,
            "objective_formula_verified": True,
            "replays": replays,
            "rows_recounted": sum(f["episodes"] for r in counts.values() for f in r.values()),
        },
    )
    write_new(
        output / "summary.json",
        {
            "results": results,
            "common_steps": p["common_steps"],
            "new_training_steps": 0,
            "original_fixed_budget_complete": False,
            "amended_evaluation_complete": True,
            "telemetry_common_prefix": p["telemetry_common_prefix"],
            "reload_episodes": len(results) * 400,
            "test_episodes": len(results) * 1000,
            "recipe_confirmed": False,
            "limitations": p["limitations"],
        },
    )


def batch(output, stage, names, deadline):
    for start in range(0, len(names), 2):
        children, logs = [], []
        try:
            for name in names[start : start + 2]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Recovery deadline reached")
                log = (output / f"worker-{stage}-{name}.log").open("x")
                logs.append(log)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(Path(__file__).resolve()),
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
                    raise RuntimeError(f"Recovery {stage} failed; inspect durable worker logs")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Two-hour recovery limit reached")
                time.sleep(1)
            if any(c.returncode != 0 for _, c in children):
                raise RuntimeError(f"Recovery {stage} failed")
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
    write_new(output / "pipeline_started.json", {"time_unix": time.time(), "training_steps": 0})
    started = time.monotonic()
    deadline = started + p["maximum_pipeline_seconds"]
    try:
        batch(output, "freeze", ["all"], deadline)
        names = list(read(output / "selection.json")["models"])
        batch(output, "reload", names, deadline)
        batch(output, "test", names, deadline)
        batch(output, "audit", ["all"], deadline)
        checked(output)
        write_new(
            output / "pipeline_complete.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "amended_evaluation_complete": True,
                "original_fixed_budget_complete": False,
                "new_training_steps": 0,
            },
        )
    except BaseException as exc:
        write_new(
            output / "pipeline_failed.json",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "pipeline", "freeze", "reload", "test", "audit")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "pipeline":
        with (output / "pipeline.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            pipeline(output)
    elif args.stage in ("prepare", "freeze", "audit"):
        globals()[args.stage](output)
    else:
        inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
