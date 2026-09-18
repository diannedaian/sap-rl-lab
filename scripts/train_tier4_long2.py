"""Second bounded Tier4 continuation, preserving every earlier frozen experiment."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean

import finish_tier4_evaluation as evidence
import train_tier4_long as prior_run
from audit_exploration_pilot import choose_cases

from sap_rl_lab import tier4_experiment as parent
from sap_rl_lab.evaluation import evaluate_suite, file_digest
from sap_rl_lab.historical_confirmation import copy_checked
from sap_rl_lab.ppo_guardrails import GuardrailConfig, train_guarded, write_new
from sap_rl_lab.training import TrainingConfig, policy_digest

ROOT = parent.ROOT
PARENT = ROOT / "runs/tier4-long-v1"
PRIOR_EVAL = PARENT / "evaluation"
PLAN = ROOT / "docs/TIER4_LONG2_PLAN.md"
STEPS, INTERVAL = 1_048_576, 262_144
TRAIN_SEEDS = (56101, 56201, 56301)
SOURCE_SEGMENT_STEPS, SOURCE_LINEAGE_STEPS = 1_048_576, 3_145_728
TEST_SEED, MAX_SECONDS = 69_000_000, 7200


def checked(output):
    prior_run.checked(PARENT)
    evidence.checked(PRIOR_EVAL)
    path = output / "protocol.json"
    if file_digest(path) != parent.read(output / "seal.json")["sha256"]:
        raise ValueError("Continuation protocol changed")
    p = parent.read(path)
    for path, digest in p["external_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError(f"Continuation input changed: {path}")
    return p


def make_config(original, output, seed, training_seed, binding):
    return replace(
        TrainingConfig(**original),
        seed=training_seed,
        timesteps=STEPS,
        output_dir=str(output / f"candidate-{seed}"),
        initialize_from=binding["path"],
        expected_initial_policy_sha256=binding["policy_sha256"],
        evaluation_interval=INTERVAL,
    )


def prepare(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    prior = evidence.checked(PRIOR_EVAL)
    if not parent.read(PARENT / "pipeline_complete.json")["training_and_evaluation_complete"]:
        raise ValueError("Prior evaluation incomplete")
    output.mkdir(parents=True, exist_ok=False)
    external = {}

    def protect(path):
        path = Path(path).resolve()
        external[str(path)] = file_digest(path)

    for path in (
        PLAN,
        Path(__file__),
        ROOT / "tests/test_tier4_long2.py",
        PARENT / "pipeline_complete.json",
        PARENT / "summary.json",
        PRIOR_EVAL / "summary.json",
    ):
        protect(path)
    arms = {}
    for source_seed, training_seed in zip(parent.SEEDS, TRAIN_SEEDS):
        original = prior["manifests"][str(source_seed)]
        old_binding = prior["models"][f"{source_seed}-final"]
        model = MaskablePPO.load(old_binding["path"], device="cpu")
        if (
            model.num_timesteps != SOURCE_SEGMENT_STEPS
            or model.target_kl != 0.01
            or policy_digest(model.policy) != old_binding["policy_sha256"]
        ):
            raise ValueError("Source is not the fixed endpoint")
        target = output / "reference" / f"seed{source_seed}.zip"
        if copy_checked(old_binding["path"], target) != old_binding["sha256"]:
            raise ValueError("Copy differs from source")
        binding = {**old_binding, "path": str(target), "source_path": old_binding["path"]}
        config = make_config(original["training"], output, source_seed, training_seed, binding)
        config.validate()
        guard = GuardrailConfig(**original["guardrails"])
        guard.validate()
        arms[str(source_seed)] = {
            "source_seed": source_seed,
            "training": asdict(config),
            "guardrails": asdict(guard),
            "initial": binding,
        }
        protect(target)
        protect(Path(old_binding["path"]).with_suffix(".json"))
    write_new(
        output / "protocol.json",
        {
            "created_utc": evidence.now(),
            "arms": arms,
            "paths": prior["paths"],
            "external_sha256": external,
            "maximum_new_ppo_decisions": len(arms) * STEPS,
            "source_timesteps": SOURCE_LINEAGE_STEPS,
            "source_checkpoint_counter": SOURCE_SEGMENT_STEPS,
            "new_timesteps_per_model": STEPS,
            "lineage_timesteps_per_model": SOURCE_LINEAGE_STEPS + STEPS,
            "maximum_seconds": MAX_SECONDS,
            "maximum_workers": 2,
            "test_seed": TEST_SEED,
            "test_episodes_per_family": parent.TEST_EPISODES,
            "weight_only_initialization": True,
            "exact_optimizer_rng_resume": False,
            "old_test_used_for_checkpoint_selection": False,
            "automatic_delivery": False,
        },
    )
    write_new(output / "seal.json", {"sha256": file_digest(output / "protocol.json")})
    checked(output)


def train_arm(output, name):
    p = checked(output)
    arm = p["arms"][name]
    config = TrainingConfig(**arm["training"])
    original = parent.read(Path(arm["initial"]["source_path"]).with_suffix(".json"))
    initial_seen = False

    def evaluator(*args, **kwargs):
        nonlocal initial_seen
        result = evaluate_suite(*args, **kwargs)
        if not initial_seen:
            for family in original["evaluation"]["families"]:
                if (
                    result["families"][family]["episode_results"]
                    != original["evaluation"]["families"][family]["episode_results"]
                ):
                    raise ValueError("Initial validation differs before PPO; stop without updates")
            initial_seen = True
        return result

    result = train_guarded(
        config,
        GuardrailConfig(**arm["guardrails"]),
        guard_factory=parent.LearnedFirstGuard,
        evaluator=evaluator,
    )
    if result["actual_timesteps"] != STEPS or result["stop_reason"] != "budget_complete":
        raise ValueError(f"Incomplete continuation: {name}/{result['stop_reason']}")
    initial = parent.read(Path(result["evaluations"][0]["checkpoint"]["path"]).with_suffix(".json"))
    for family in original["evaluation"]["families"]:
        if (
            initial["evaluation"]["families"][family]["episode_results"]
            != original["evaluation"]["families"][family]["episode_results"]
        ):
            raise ValueError("Initial validation differs from old endpoint")
    write_new(
        output / f"train-{name}-complete.json",
        {
            "actual_timesteps": result["actual_timesteps"],
            "parent_initial_reload_exact": True,
            "selected_missing": result["selected"] is None,
        },
    )
    checked(output)


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = checked(output)
    models, selected, curves, manifests, external = {}, {}, {}, {}, dict(p["external_sha256"])

    def protect(path):
        external[str(path)] = file_digest(path)

    for filename in ("protocol.json", "seal.json"):
        protect(output / filename)
    for name, arm in p["arms"].items():
        folder = output / f"candidate-{name}"
        done = parent.read(folder / "complete.json")
        manifest = parent.read(folder / "manifest.json")
        if (
            manifest["training"] != arm["training"]
            or manifest["guardrails"] != arm["guardrails"]
            or manifest["initial_policy_sha256"] != arm["initial"]["policy_sha256"]
            or done["actual_timesteps"] != STEPS
            or done["stop_reason"] != "budget_complete"
            or not parent.read(output / f"train-{name}-complete.json")[
                "parent_initial_reload_exact"
            ]
        ):
            raise ValueError("Training recipe, initialization or completion differs")
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        if [r["timesteps"] for r in updates] != list(range(2048, STEPS + 1, 2048)):
            raise ValueError("Missing PPO updates")
        if [e["checkpoint"]["timesteps"] for e in done["evaluations"]] != list(
            range(0, STEPS + 1, INTERVAL)
        ):
            raise ValueError("Missing validation history")
        guard = parent.LearnedFirstGuard(GuardrailConfig(**arm["guardrails"]))
        records, curves[name] = [], []
        for entry in done["evaluations"]:
            item = entry["checkpoint"]
            path = Path(item["path"])
            record = parent.read(path.with_suffix(".json"))
            model = MaskablePPO.load(path, device="cpu")
            if (
                file_digest(path) != item["sha256"]
                or model.num_timesteps != item["timesteps"]
                or policy_digest(model.policy) != item["policy_sha256"]
                or record["checkpoint"] != item
            ):
                raise ValueError("Checkpoint binding differs")
            evidence.count_suite(
                record["evaluation"],
                p["paths"]["validation"],
                parent.VAL_EPISODES,
                parent.VAL_SEED,
                model.sap_environment_contract,
            )
            decision = guard.consider(record["evaluation"], item)
            if json.loads(json.dumps(decision)) != record["decision"]:
                raise ValueError("Selection cannot be reproduced")
            records.append(item)
            curves[name].append(
                {
                    "checkpoint": item,
                    "decision": decision,
                    "lineage_timesteps": SOURCE_LINEAGE_STEPS + item["timesteps"],
                }
            )
            protect(path)
            protect(path.with_suffix(".json"))
        if json.loads(json.dumps(guard.best)) != done["selected"]:
            raise ValueError("Selected checkpoint differs")
        selected[name] = evidence.bind_selection(name, records, done["selected"], models)
        manifests[name] = manifest
        for file in ("complete.json", "manifest.json", "updates.jsonl"):
            protect(folder / file)
    cases = {
        name: choose_cases(
            parent.read(Path(item["path"]).with_suffix(".json"))["evaluation"]["families"]
        )
        for name, item in models.items()
    }
    destination = output / "evaluation"
    destination.mkdir(exist_ok=False)
    write_new(
        destination / "protocol.json",
        {
            "created_utc": evidence.now(),
            "models": models,
            "selected": selected,
            "curves": curves,
            "manifests": manifests,
            "cases": cases,
            "paths": p["paths"],
            "external_sha256": external,
            "test_seed": TEST_SEED,
            "original_selection_incomplete": any(v is None for v in selected.values()),
            "new_training_decisions": 0,
            "note": "Evaluation substage only; training budget is in parent protocol.",
        },
    )
    write_new(destination / "seal.json", {"sha256": file_digest(destination / "protocol.json")})
    evidence.checked(destination)


def test_model(output, name):
    destination = output / "evaluation"
    p, item, model = evidence.model_for(destination, name)
    evidence.require_reloads(destination, p)
    write_new(
        destination / f"test-start-{name}.json", {"started_utc": evidence.now(), "checkpoint": item}
    )
    result = evaluate_suite(
        model, p["paths"]["test"], episodes=parent.TEST_EPISODES, seed=p["test_seed"]
    )
    evidence.count_suite(
        result,
        p["paths"]["test"],
        parent.TEST_EPISODES,
        p["test_seed"],
        model.sap_environment_contract,
    )
    write_new(
        destination / f"test-{name}.json",
        {
            "checkpoint": item,
            "evaluation": result,
            "completed_utc": evidence.now(),
        },
    )


def worker(output, action, name):
    checked(output)
    if action == "train":
        train_arm(output, name)
    elif action == "test":
        test_model(output, name)
    else:
        evidence.worker(output / "evaluation", action, name)
    checked(output)


def summarize(output):
    evidence.summarize(output / "evaluation")
    result = parent.read(output / "evaluation/summary.json")
    changes = {
        str(seed): {
            "initial": result["results"][f"{seed}-initial"]["learned_success"],
            "final": result["results"][f"{seed}-final"]["learned_success"],
            "change": result["results"][f"{seed}-final"]["learned_success"]
            - result["results"][f"{seed}-initial"]["learned_success"],
        }
        for seed in parent.SEEDS
    }
    write_new(
        output / "summary.json",
        {
            "new_ppo_decisions": STEPS * len(parent.SEEDS),
            "changes": changes,
            "mean_initial": fmean(x["initial"] for x in changes.values()),
            "mean_final": fmean(x["final"] for x in changes.values()),
            "selected": result["selected"],
            "selection_incomplete": result["original_selection_incomplete"],
            "results": result["results"],
            "test_used_for_selection": False,
            "same_opponent_ecology_fresh_episode_seeds": True,
            "automatic_delivery": False,
            "evaluation_summary_sha256": file_digest(output / "evaluation/summary.json"),
        },
    )
    checked(output)


def batch(output, action, names, deadline):
    pending, running = list(names), {}
    try:
        while pending or running:
            if time.monotonic() >= deadline:
                raise TimeoutError("Continuation deadline")
            while pending and len(running) < 2:
                name = str(pending.pop(0))
                log = (output / f"{action}-{name}.log").open("x")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            str(Path(__file__).resolve()),
                            "worker",
                            "--output",
                            str(output),
                            "--action",
                            action,
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log.close()
                    raise
                running[name] = process, log
                print(f"Started {action}/{name} pid={process.pid}", flush=True)
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                if code:
                    raise RuntimeError(f"Worker {action}/{name} exit={code}")
                print(f"Completed {action}/{name}", flush=True)
            if running:
                time.sleep(1)
    finally:
        for process, log in running.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def pipeline(output):
    deadline = time.monotonic() + MAX_SECONDS
    prepare(output)
    batch(output, "train", parent.SEEDS, deadline)
    freeze(output)
    names = parent.read(output / "evaluation/protocol.json")["models"]
    for action in ("reload", "test", "replay"):
        batch(output, action, names, deadline)
    summarize(output)
    write_new(
        output / "pipeline_complete.json",
        {"utc": evidence.now(), "training_and_evaluation_complete": True},
    )


def timeout(signum, frame):
    raise TimeoutError("Tier4 continuation two-hour wall-clock limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pipeline", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=("train", "reload", "test", "replay"))
    parser.add_argument("--name")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.command == "pipeline":
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(MAX_SECONDS)
            try:
                pipeline(output)
            finally:
                signal.alarm(0)
        else:
            worker(output, args.action, args.name)
    except Exception:
        if output.exists():
            path = output / f"failure-{args.action or 'pipeline'}-{args.name or 'root'}.json"
            if not path.exists():
                write_new(path, {"traceback": traceback.format_exc(), "utc": evidence.now()})
        raise


if __name__ == "__main__":
    main()
