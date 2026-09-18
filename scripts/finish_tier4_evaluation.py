"""Finish frozen Tier4 evaluation, retaining missing selections; never train."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from audit_exploration_pilot import choose_cases
from audit_midgame_confirmation import check_environment_contract, exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab import tier4_experiment as parent
from sap_rl_lab.evaluation import evaluate_suite, file_digest
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.historical_confirmation import copy_checked
from sap_rl_lab.ppo_guardrails import GuardrailConfig, write_new
from sap_rl_lab.training import policy_digest

ROOT = parent.ROOT
SOURCE = ROOT / "runs/tier4-bc-kl-v1"
PLAN = ROOT / "docs/TIER4_EVALUATION_CONTINUATION.md"
MAX_SECONDS = 3600


def now():
    return datetime.now(timezone.utc).isoformat()


def checked(output):
    parent.checked(SOURCE)
    protocol = output / "protocol.json"
    if file_digest(protocol) != parent.read(output / "seal.json")["sha256"]:
        raise ValueError("Continuation protocol changed")
    p = parent.read(protocol)
    for path, sha in p["external_sha256"].items():
        if file_digest(path) != sha:
            raise ValueError(f"Frozen continuation input changed: {path}")
    return p


def bind_selection(seed, records, selected, models):
    """Keep a failed selector null; fixed endpoint is a separate diagnostic role."""
    for kind, item in (("initial", records[0]), ("final", records[-1])):
        models[f"{seed}-{kind}"] = item
    if selected is None:
        return None
    item = selected["checkpoint"]
    if item not in records:
        raise ValueError("Selected checkpoint missing from frozen history")
    for name, bound in models.items():
        if name.startswith(f"{seed}-") and item == bound:
            return name
    name = f"{seed}-selected"
    models[name] = item
    return name


def count_suite(result, paths, episodes, seed, contract):
    if set(result["families"]) != set(paths):
        raise ValueError("Missing or unexpected family")
    counts = {}
    for name, family in result["families"].items():
        if not family["deterministic"] or family["league_sha256"] != file_digest(paths[name]):
            raise ValueError("Opponent or inference mode changed")
        check_environment_contract(family["environment_contract"], contract)
        counts[name] = exact_family_counts(family, episodes, seed)
    return counts


def prepare(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = parent.checked(SOURCE)
    if not (SOURCE / "failure-candidate-54301.json").exists():
        raise ValueError("Expected recorded selection failure")
    if (SOURCE / "selection.json").exists() or list(SOURCE.glob("test-*.json")):
        raise ValueError("Original selection or tests already opened")
    external = {}

    def protect(path):
        path = Path(path).resolve()
        external[str(path)] = file_digest(path)

    for path in (PLAN, Path(__file__), ROOT / "tests/test_tier4_continuation.py"):
        protect(path)
    for name in (
        "audit_exploration_pilot.py",
        "audit_midgame_confirmation.py",
        "inspect_confirmation_delivery.py",
        "summarize_policy_replays.py",
    ):
        protect(ROOT / "scripts" / name)
    for path in SOURCE.glob("failure-*.json"):
        protect(path)
    models, selected, curves, manifests = {}, {}, {}, {}
    total = 0
    for name in parent.GENERATORS:
        path = SOURCE / "generators" / name / "complete.json"
        done = parent.read(path)
        if (
            done["stop_reason"] != "budget_complete"
            or done["actual_timesteps"] != parent.GENERATOR_STEPS
        ):
            raise ValueError("Generator budget incomplete")
        total += done["actual_timesteps"]
        protect(path)
    for seed in parent.SEEDS:
        _, cp, cfg = parent.candidate_config(SOURCE, seed)
        folder = SOURCE / f"candidate-{seed}"
        done = parent.read(folder / "complete.json")
        manifest = parent.read(folder / "manifest.json")
        if (
            done["stop_reason"] != "budget_complete"
            or done["actual_timesteps"] != parent.CANDIDATE_STEPS
        ):
            raise ValueError("Candidate budget incomplete")
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        if [r["timesteps"] for r in updates] != list(range(2048, parent.CANDIDATE_STEPS + 1, 2048)):
            raise ValueError("Missing PPO update history")
        total += done["actual_timesteps"]
        if [e["checkpoint"]["timesteps"] for e in done["evaluations"]] != list(
            range(0, parent.CANDIDATE_STEPS + 1, 262144)
        ):
            raise ValueError("Missing validation checkpoint")
        guard = parent.LearnedFirstGuard(GuardrailConfig(**p["guardrails"]))
        records, curves[str(seed)] = [], []
        for entry in done["evaluations"]:
            item = entry["checkpoint"]
            path = Path(item["path"])
            raw = parent.read(path.with_suffix(".json"))
            if raw["checkpoint"] != item or file_digest(path) != item["sha256"]:
                raise ValueError("Saved checkpoint binding differs")
            model = MaskablePPO.load(path, device="cpu")
            if (
                model.num_timesteps != item["timesteps"]
                or policy_digest(model.policy) != item["policy_sha256"]
            ):
                raise ValueError("Saved checkpoint tensors differ")
            count_suite(
                raw["evaluation"],
                cp["paths"]["validation"],
                parent.VAL_EPISODES,
                parent.VAL_SEED,
                model.sap_environment_contract,
            )
            decision = guard.consider(raw["evaluation"], item)
            if json.loads(json.dumps(decision)) != raw["decision"]:
                raise ValueError("Selector decision differs")
            records.append(item)
            curves[str(seed)].append({"checkpoint": item, "decision": decision})
            protect(path)
            protect(path.with_suffix(".json"))
        if json.loads(json.dumps(guard.best)) != done["selected"]:
            raise ValueError("Recomputed selection differs")
        selected[str(seed)] = bind_selection(seed, records, done["selected"], models)
        manifests[str(seed)] = manifest
        for path in (folder / "complete.json", folder / "manifest.json", folder / "updates.jsonl"):
            protect(path)
    if total != p["maximum_ppo_decisions"]:
        raise ValueError("Original total budget mismatch")
    for path in (SOURCE / "candidate_protocol.json", SOURCE / "candidate_protocol.seal.json"):
        protect(path)
    for paths in cp["paths"].values():
        for path in paths.values():
            protect(path)
    cases = {}
    for name, item in models.items():
        raw = parent.read(Path(item["path"]).with_suffix(".json"))
        cases[name] = choose_cases(raw["evaluation"]["families"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "protocol.json",
        {
            "created_utc": now(),
            "models": models,
            "selected": selected,
            "curves": curves,
            "manifests": manifests,
            "cases": cases,
            "paths": cp["paths"],
            "external_sha256": external,
            "original_training_decisions": total,
            "new_training_decisions": 0,
            "original_selection_incomplete": any(v is None for v in selected.values()),
            "test_used_for_selection": False,
            "maximum_seconds": MAX_SECONDS,
        },
    )
    write_new(output / "seal.json", {"sha256": file_digest(output / "protocol.json")})
    checked(output)
    print(f"Sealed {len(models)} distinct checkpoints; missing selection retained", flush=True)


def model_for(output, name):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = checked(output)
    item = p["models"][name]
    model = MaskablePPO.load(item["path"], device="cpu")
    if policy_digest(model.policy) != item["policy_sha256"]:
        raise ValueError("Reload policy digest differs")
    return p, item, model


def require_reloads(output, p):
    for name, item in p["models"].items():
        r = parent.read(output / f"reload-{name}.json")
        if r["checkpoint"] != item or r["exact_rows"] != 5 * parent.VAL_EPISODES:
            raise ValueError("Missing exact reload")


def worker(output, action, name):
    p, item, model = model_for(output, name)
    if action == "reload":
        actual = evaluate_suite(
            model, p["paths"]["validation"], episodes=parent.VAL_EPISODES, seed=parent.VAL_SEED
        )
        expected = parent.read(Path(item["path"]).with_suffix(".json"))["evaluation"]
        for family in expected["families"]:
            if (
                actual["families"][family]["episode_results"]
                != expected["families"][family]["episode_results"]
            ):
                raise ValueError(f"Exact validation reload differs: {name}/{family}")
        write_new(
            output / f"reload-{name}.json",
            {
                "checkpoint": item,
                "exact_rows": 5 * parent.VAL_EPISODES,
                "completed_utc": now(),
            },
        )
    elif action == "test":
        require_reloads(output, p)
        write_new(output / f"test-start-{name}.json", {"started_utc": now(), "checkpoint": item})
        result = evaluate_suite(
            model, p["paths"]["test"], episodes=parent.TEST_EPISODES, seed=parent.TEST_SEED
        )
        count_suite(
            result,
            p["paths"]["test"],
            parent.TEST_EPISODES,
            parent.TEST_SEED,
            model.sap_environment_contract,
        )
        write_new(
            output / f"test-{name}.json",
            {"checkpoint": item, "evaluation": result, "completed_utc": now()},
        )
    else:
        folder = output / "replays" / name
        copied = folder / "model.zip"
        copy_checked(item["path"], copied)
        config = p["manifests"][name.split("-")[0]]["training"]
        write_new(folder / "run_manifest.json", {"config": config})
        rows = []
        for index, case in enumerate(p["cases"][name]):
            destination = folder / f"case{index}"
            with (folder / f"case{index}.log").open("x") as log, redirect_stdout(log):
                inspect(
                    str(copied),
                    p["paths"]["validation"][case["family"]],
                    destination,
                    episodes=1,
                    seed=case["expected"]["seed"],
                )
            record = parent.read(destination / f"episode-{case['expected']['seed']}.json")
            compare_replay(record["summary"], case["expected"])
            for s in record["steps"]:
                if abs(s["objective_reward"] - (s["info"]["game_reward"] - 0.005)) > 1e-9:
                    raise ValueError("Action reward objective differs")
            rows.append(
                {
                    "family": case["family"],
                    "path": str(destination),
                    "sha256": file_digest(destination / f"episode-{case['expected']['seed']}.json"),
                    **summarize_episode(record, config["gamma"]),
                }
            )
        write_new(output / f"replay-{name}.json", {"checkpoint": item, "exact_replays": rows})
    checked(output)


def summarize(output):
    p = checked(output)
    require_reloads(output, p)
    last_reload = max(
        parent.read(output / f"reload-{n}.json")["completed_utc"] for n in p["models"]
    )
    metrics = {}
    for name, item in p["models"].items():
        if parent.read(output / f"test-start-{name}.json")["started_utc"] <= last_reload:
            raise ValueError("Test opened before reload barrier")
        data = parent.read(output / f"test-{name}.json")
        if data["checkpoint"] != item:
            raise ValueError("Test binding changed")
        families = data["evaluation"]["families"]
        rows = [r for f in families.values() for r in f["episode_results"]]
        learned = [
            r for n, f in families.items() if n.startswith("test_") for r in f["episode_results"]
        ]
        metrics[name] = {
            "learned_successes": sum(r["success"] for r in learned),
            "learned_episodes": len(learned),
            "learned_success": fmean(r["success"] for r in learned),
            "learned_mean_wins": fmean(r["wins"] for r in learned),
            "forced_episodes": sum(r["forced_end_turns"] > 0 for r in rows),
            "episodes": len(rows),
            "forcing": fmean(r["forced_end_turns"] > 0 for r in rows),
            "truncations": sum(r["truncated"] for r in rows),
            "no_purchase_zero_win_losses": sum(
                r["wins"] == 0 and not r["action_counts"].get("buy_pet", 0) for r in rows
            ),
            "family_success": {
                n: fmean(r["success"] for r in f["episode_results"]) for n, f in families.items()
            },
            "exact_replays": len(parent.read(output / f"replay-{name}.json")["exact_replays"]),
        }
    evidence = {str(path): file_digest(path) for path in sorted(output.glob("*.json"))}
    write_new(
        output / "summary.json",
        {
            "results": metrics,
            "selected": p["selected"],
            "all_training_was_already_complete": True,
            "new_training_decisions": 0,
            "original_selection_incomplete": p["original_selection_incomplete"],
            "selected_three_seed_mean": None
            if p["original_selection_incomplete"]
            else fmean(metrics[name]["learned_success"] for name in p["selected"].values()),
            "all_reloads_before_tests": True,
            "all_replays_exact": True,
            "evidence_sha256": evidence,
            "automatic_delivery": False,
            "completed_utc": now(),
        },
    )
    checked(output)


def batch(output, action, names, deadline):
    pending, running = list(names), {}
    try:
        while pending or running:
            if time.monotonic() >= deadline:
                raise TimeoutError("Evaluation continuation deadline")
            while pending and len(running) < 2:
                name = pending.pop(0)
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
                    raise RuntimeError(f"{action}/{name} exit={code}")
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
    p = checked(output)
    for action in ("reload", "test", "replay"):
        batch(output, action, p["models"], deadline)
    summarize(output)
    write_new(
        output / "evaluation_complete.json", {"completed_utc": now(), "new_training_decisions": 0}
    )


def timeout(signum, frame):
    raise TimeoutError("Evaluation-only continuation hard one-hour limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pipeline", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=("reload", "test", "replay"))
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
                write_new(path, {"traceback": traceback.format_exc(), "utc": now()})
        raise


if __name__ == "__main__":
    main()
