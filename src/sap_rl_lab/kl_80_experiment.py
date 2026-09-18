"""Bounded three-seed KL extension with a predeclared 80% acceptance endpoint."""

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
from statistics import fmean

from . import evaluation
from .expanded import verify_inputs
from .expanded_confirmation import compare_reload, read
from .historical_confirmation import copy_checked, fallback_counts
from .imitation_experiment import OLD_BEST, OLD_SHA, bind_model, require_reloads
from .kl_long_experiment import canonical, update_metrics
from .kl_long_experiment import frozen as parent_frozen
from .midgame import ROOT
from .ppo_guardrails import GuardrailConfig, ValidationGuard, train_guarded, write_new
from .round3 import source_archive
from .training import TrainingConfig, policy_digest

SEEDS = (17301, 204101, 204201)
TRAIN_SEEDS = (98301, 98302, 98303)
STEPS, INTERVAL = 2_097_152, 262_144
VAL_SEED, TEST_SEED = 37_000_000, 38_000_000
VAL_EPISODES, TEST_EPISODES = 100, 500
PLAN, PARENT = ROOT / "docs/KL_80_PLAN.md", ROOT / "runs/kl-long-v1"
HELPERS = (
    "scripts/audit_kl_80.py",
    "scripts/audit_exploration_pilot.py",
    "scripts/audit_midgame_confirmation.py",
    "scripts/inspect_confirmation_delivery.py",
    "scripts/summarize_policy_replays.py",
)


def checked(output):
    p = read(output / "protocol.json")
    if evaluation.file_digest(output / "protocol.json") != read(output / "seal.json")["sha256"]:
        raise ValueError("80% protocol changed")
    verify_inputs(output, p)
    if evaluation.file_digest(PLAN) != p["plan_sha256"]:
        raise ValueError("80% plan changed")
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
    prior, selected = parent_frozen(PARENT)
    if not read(PARENT / "pipeline_complete.json")["training_evaluation_and_audit_complete"]:
        raise ValueError("Parent has not completed")
    output.mkdir(parents=True, exist_ok=False)
    paths, starts, arms, external = {}, {}, {}, {}
    for split, families in prior["paths"].items():
        paths[split] = {}
        for family, source in families.items():
            target = output / "data" / split / f"{family}.json"
            copy_checked(source, target)
            paths[split][family] = str(target)
    for index, seed in enumerate(SEEDS):
        name = f"kl-seed{seed}"
        item = selected["models"][f"{name}-final"]
        model = MaskablePPO.load(item["path"], device="cpu")
        if model.num_timesteps != 4_194_304 or model.target_kl != 0.01:
            raise ValueError("Parent is not the fixed KL endpoint")
        target = output / "reference" / f"initial-seed{seed}.zip"
        copy_checked(item["path"], target)
        starts[str(seed)] = {
            "path": str(target),
            "source_path": item["path"],
            "sha256": item["sha256"],
            "policy_sha256": policy_digest(model.policy),
            "timesteps": model.num_timesteps,
        }
        external[item["path"]] = item["sha256"]
        config = TrainingConfig(
            **{
                **prior["arms"][name]["training"],
                "seed": TRAIN_SEEDS[index],
                "timesteps": STEPS,
                "evaluation_interval": INTERVAL,
                "validation_seed": VAL_SEED,
                "validation_episodes": VAL_EPISODES,
                "opponent_leagues": tuple(paths["train"].values()),
                "validation_leagues": paths["validation"],
                "output_dir": str(output / name),
                "initialize_from": str(target),
                "expected_initial_policy_sha256": starts[str(seed)]["policy_sha256"],
            }
        )
        guard = GuardrailConfig(**prior["arms"][name]["guardrails"])
        config.validate()
        guard.validate()
        arms[name] = {"source_seed": seed, "training": asdict(config), "guardrails": asdict(guard)}
    external.update({str(ROOT / x): evaluation.file_digest(ROOT / x) for x in HELPERS})
    for filename in ("protocol.json", "selection.json", "pipeline_complete.json"):
        external[str(PARENT / filename)] = evaluation.file_digest(PARENT / filename)
    write_new(
        output / "protocol.json",
        {
            "arms": arms,
            "seeds": SEEDS,
            "starts": starts,
            "paths": paths,
            "steps_per_model": STEPS,
            "maximum_ppo_steps": 3 * STEPS,
            "validation_seed": VAL_SEED,
            "test_seed": TEST_SEED,
            "validation_episodes_per_family": VAL_EPISODES,
            "test_episodes_per_family": TEST_EPISODES,
            "maximum_worker_processes": 2,
            "maximum_pipeline_seconds": 7200,
            "maximum_comparison_models": 9,
            "maximum_test_episodes": 22_500,
            "maximum_reload_episodes": 3600,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(x.relative_to(output)): evaluation.file_digest(x)
                for folder in ("data", "reference")
                for x in sorted((output / folder).rglob("*"))
                if x.is_file()
            },
            "external_sha256": external,
            "plan_sha256": copy_checked(PLAN, output / "plan.md"),
            "primary_endpoint": "mean learned ten-win success across three validation selections",
            "test_results_used_for_selection": False,
            "automatic_delivery": False,
            "limits": "Reused weights, new optimizer/RNG; studied opponent ecology, fresh episode "
            "seeds. 80% is an observed mean, not a confidence lower bound. Swap/freeze per battle "
            "is a proxy, not a measured loop rate. No new BC, rewards, opponents, or simulator.",
        },
    )
    write_new(output / "seal.json", {"sha256": evaluation.file_digest(output / "protocol.json")})
    checked(output)


def arm(output, name):
    payload = checked(output)["arms"][name]
    with (output / f"{name}.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
        result = train_guarded(
            TrainingConfig(**payload["training"]), GuardrailConfig(**payload["guardrails"])
        )
    manifest = read(output / name / "manifest.json")
    write_new(
        output / name / "run_manifest.json",
        {
            "config": manifest["training"],
            "environment_contract": manifest["environment_contract"],
            "guardrails": manifest["guardrails"],
        },
    )
    checked(output)
    if result["stop_reason"] != "budget_complete" or result["actual_timesteps"] != STEPS:
        raise ValueError(f"{name} incomplete: {result['stop_reason']}")


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
            or manifest["initialization"] != "weights_only_fresh_optimizer"
            or manifest["initial_policy_sha256"]
            != payload["training"]["expected_initial_policy_sha256"]
            or done["stop_reason"] != "budget_complete"
            or done["actual_timesteps"] != STEPS
        ):
            raise ValueError("Incomplete or non-protocol training")
        rows = [json.loads(x) for x in (folder / "updates.jsonl").read_text().splitlines()]
        telemetry[name] = update_metrics(rows, 0.01, steps=STEPS)
        history = done["evaluations"]
        if [h["checkpoint"]["timesteps"] for h in history] != list(range(0, STEPS + 1, INTERVAL)):
            raise ValueError("Incomplete validation curve")
        guard, records = ValidationGuard(GuardrailConfig(**payload["guardrails"])), {}
        for h in history:
            cp = h["checkpoint"]
            record = read(Path(cp["path"]).with_suffix(".json"))
            model = MaskablePPO.load(cp["path"], device="cpu")
            if (
                evaluation.file_digest(cp["path"]) != cp["sha256"]
                or policy_digest(model.policy) != cp["policy_sha256"]
                or model.num_timesteps != cp["timesteps"]
                or record["checkpoint"] != cp
                or evaluation.compact_evaluation(record["evaluation"]) != h["evaluation"]
                or canonical(guard.consider(record["evaluation"], cp)) != canonical(h["decision"])
                or record["decision"] != h["decision"]
            ):
                raise ValueError("Validation weight, result, or decision binding differs")
            records[cp["path"]] = record["evaluation"]
        if canonical(guard.best) != canonical(done["selected"]):
            raise ValueError("Cannot reproduce validation selection")
        final = MaskablePPO.load(done["last_model"], device="cpu")
        if (
            evaluation.file_digest(done["last_model"]) != done["last_sha256"]
            or final.num_timesteps != STEPS
            or policy_digest(final.policy) != history[-1]["checkpoint"]["policy_sha256"]
        ):
            raise ValueError("Final model differs from its evaluated weights")
        start = p["starts"][str(payload["source_seed"])]
        if history[0]["checkpoint"]["policy_sha256"] != start["policy_sha256"]:
            raise ValueError("Initial policy differs")
        candidates = {
            "initial": (start["path"], records[history[0]["checkpoint"]["path"]]),
            "final": (done["last_model"], records[history[-1]["checkpoint"]["path"]]),
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
    write_new(
        output / "selection.json",
        {
            "models": models,
            "missing_eligible": missing,
            "telemetry": telemetry,
            "protocol_sha256": evaluation.file_digest(output / "protocol.json"),
            "all_validation_bindings_exact": True,
            "test_results_used_for_selection": False,
        },
    )
    write_new(
        output / "selection_seal.json",
        {
            "sha256": evaluation.file_digest(output / "selection.json"),
        },
    )


def frozen(output):
    p, selected = checked(output), read(output / "selection.json")
    if evaluation.file_digest(output / "selection.json") != read(output / "selection_seal.json")[
        "sha256"
    ] or selected["protocol_sha256"] != evaluation.file_digest(output / "protocol.json"):
        raise ValueError("Frozen selection changed")
    expected = {f"{n}-{k}" for n in p["arms"] for k in ("initial", "final")}
    expected |= {f"{n}-selected" for n in p["arms"] if n not in selected["missing_eligible"]}
    if set(selected["models"]) != expected:
        raise ValueError("Frozen comparison incomplete")
    for item in selected["models"].values():
        if (
            evaluation.file_digest(item["path"]) != item["sha256"]
            or evaluation.file_digest(item["validation_path"]) != item["validation_sha256"]
        ):
            raise ValueError("Frozen weights/results changed")
    return p, selected


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selected = frozen(output)
    if stage == "test":
        require_reloads(output, selected)
    elif stage != "reload":
        raise ValueError("Unknown inference stage")
    item = selected["models"][name]
    split, count, seed = (
        ("validation", VAL_EPISODES, VAL_SEED)
        if stage == "reload"
        else ("test", TEST_EPISODES, TEST_SEED)
    )
    write_new(
        output / f"{stage}-{name}-started.json",
        {
            "sha256": item["sha256"],
            "time_unix": time.time(),
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
            result["families"][f]["episode_results"], {x["turn"] for x in read(path)["snapshots"]}
        )
        for f, path in p["paths"][split].items()
    }
    frozen(output)
    write_new(output / f"{stage}-{name}.json", result)


def result_metrics(result):
    families = result["families"]
    all_rows = [r for f in families.values() for r in f["episode_results"]]
    learned = [r for n in ("fresh_stats", "fresh_mix") for r in families[n]["episode_results"]]
    battles = sum(r["battles"] for r in all_rows)
    manipulation = sum(
        r["action_counts"].get(k, 0) for r in all_rows for k in ("swap_adjacent", "freeze")
    )
    return {
        "learned_successes": sum(r["success"] for r in learned),
        "learned_episodes": len(learned),
        "learned_success": fmean(r["success"] for r in learned),
        "episodes": len(all_rows),
        "forced_episodes": sum(r["forced_end_turns"] > 0 for r in all_rows),
        "forced_episode_rate": fmean(r["forced_end_turns"] > 0 for r in all_rows),
        "worst_family_forced_episode_rate": max(
            fmean(r["forced_end_turns"] > 0 for r in f["episode_results"])
            for f in families.values()
        ),
        "truncations": sum(r["truncated"] for r in all_rows),
        "no_purchase_zero_win_losses": sum(
            not r["success"] and r["wins"] == 0 and r["action_counts"].get("buy_pet", 0) == 0
            for r in all_rows
        ),
        "swap_freeze_actions_per_battle": manipulation / max(1, battles),
        "mean_episode_actions": fmean(r["actions"] for r in all_rows),
        "family_success": {
            n: fmean(r["success"] for r in f["episode_results"]) for n, f in families.items()
        },
    }


def acceptance(results):
    """No rounding or post-test checkpoint/seed selection; absent means fail closed."""
    names = [f"kl-seed{s}" for s in SEEDS]
    missing = [n for n in names if f"{n}-selected" not in results]
    selected = [results[f"{n}-selected"] for n in names if n not in missing]
    successes = sum(r["learned_successes"] for r in selected)
    episodes = sum(r["learned_episodes"] for r in selected)
    target = not missing and episodes == 3 * 2 * TEST_EPISODES and successes * 5 >= episodes * 4
    checks = {}
    for name in names:
        if name in missing:
            checks[name] = {"missing_selection": True, "passed": False}
            continue
        r, initial = results[f"{name}-selected"], results[f"{name}-initial"]
        limit = initial["swap_freeze_actions_per_battle"] + max(
            1.0, 0.2 * initial["swap_freeze_actions_per_battle"]
        )
        checks[name] = {
            "forcing_ok": r["forced_episode_rate"] <= 0.01,
            "worst_family_forcing_ok": r["worst_family_forced_episode_rate"] <= 0.05,
            "no_truncations": r["truncations"] == 0,
            "no_empty_losses": r["no_purchase_zero_win_losses"] == 0,
            "behavior_proxy_ok": r["swap_freeze_actions_per_battle"] <= limit,
            "behavior_proxy_limit": limit,
        }
        checks[name]["passed"] = all(
            v for k, v in checks[name].items() if k != "behavior_proxy_limit"
        )
    return {
        "selected_learned_successes": successes,
        "selected_learned_episodes": episodes,
        "selected_mean_learned_success": successes / episodes if episodes else None,
        "score_target_met": bool(target),
        "checks": checks,
        "reliability_and_behavior_passed": all(c["passed"] for c in checks.values()),
        "missing_eligible": missing,
        "eligible_after_evidence_audit": bool(target and all(c["passed"] for c in checks.values())),
        "interpretation": "Observed fixed-benchmark target, not an 80% true-rate lower bound. "
        "Behavior proxy is not actual loop frequency. "
        "Human replay review precedes Tier preparation.",
    }


def summarize(output):
    p, selected = frozen(output)
    require_reloads(output, selected)
    results = {}
    for name, item in selected["models"].items():
        result = read(output / f"test-{name}.json")
        if result["model_sha256"] != item["sha256"] or result[
            "selection_sha256"
        ] != evaluation.file_digest(output / "selection.json"):
            raise ValueError("Test weight/selection binding differs")
        results[name] = result_metrics(result)
    write_new(
        output / "summary.json",
        {
            "results": results,
            "acceptance": acceptance(results),
            "telemetry": selected["telemetry"],
            "all_training_complete": True,
            "ppo_steps": 3 * STEPS,
            "all_validation_reloads_exact": True,
            "previous_best_preserved": True,
            "reload_episodes": len(results) * 4 * VAL_EPISODES,
            "test_episodes": len(results) * 5 * TEST_EPISODES,
            "limits": p["limits"],
            "evidence_audit_pending": True,
            "automatic_delivery": False,
        },
    )


def batch(output, stage, names, deadline):
    for start in range(0, len(names), 2):
        children, logs = [], []
        try:
            for name in names[start : start + 2]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("KL 80 deadline reached")
                log = (output / f"worker-{stage}-{name}.log").open("x")
                logs.append(log)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        "-m",
                        "sap_rl_lab.kl_80_experiment",
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
                    raise RuntimeError(f"KL 80 {stage} failed; inspect durable worker logs")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Two-hour KL 80 limit reached")
                time.sleep(1)
            if any(c.returncode != 0 for _, c in children):
                raise RuntimeError(f"KL 80 {stage} failed")
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
        parent_frozen(PARENT)
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
        from audit_kl_80 import run

        run(output, output / "audit")
    else:
        inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
