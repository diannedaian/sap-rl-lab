"""Full Tier1-6 campaign: curriculum transfer, independent pools, repeated KL-PPO segments.

Stops on a reliable 80% independent-test screen, a genuine safety/technical issue,
or low disk space. The 80% screen still requires the final confirmation/audit.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from sap_rl_lab.fullpack import experiment as exp
from sap_rl_lab.fullpack.evaluation import evaluate_suite, file_digest
from sap_rl_lab.fullpack.horizon40 import HorizonTrainingConfig, configuration
from sap_rl_lab.fullpack.opponents import build_scripted_league
from sap_rl_lab.fullpack.ppo_guardrails import (
    GuardrailConfig,
    family_metrics,
    train_guarded,
    write_new,
)
from sap_rl_lab.fullpack.recipe import LearnedFirstGuard
from sap_rl_lab.fullpack.training import policy_digest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/FULLPACK_TIER6_CAMPAIGN.md"
SEEDS = (85101, 85201, 85301)
STEP = 1_048_576


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


continuation = load_script("fullpack_continuation", "continue_fullpack40.py")
transfer = load_script("fullpack_transfer40", "transfer_fullpack_policy40.py")


def read(path):
    return json.loads(Path(path).read_text())


def setup():
    """Explicit dependency injection into the frozen pool/generator helpers.

    Their implementations and old artifacts remain unchanged. Every new game,
    demonstration, generator and saved environment contract uses forty turns.
    """
    exp.configure_tier(6)
    exp.configuration = lambda **kwargs: configuration(tier=6, **kwargs)

    def build40(*args, **kwargs):
        kwargs.setdefault("config", configuration(6).environment()[1])
        return build_scripted_league(*args, **kwargs)

    exp.build_scripted_league = build40


def seal(path, data):
    write_new(path, data)
    write_new(path.with_suffix(".seal.json"), {"sha256": file_digest(path)})


def sealed(path):
    if file_digest(path) != read(path.with_suffix(".seal.json"))["sha256"]:
        raise ValueError(f"Sealed record changed: {path}")
    return read(path)


def checked(root):
    p = sealed(root / "campaign.json")
    for filename, digest in p["external_sha256"].items():
        if file_digest(filename) != digest:
            raise ValueError(f"Campaign input changed: {filename}")
    exp.checked(root)
    return p


def prepare(root, curriculum):
    previous = continuation.recovery.checked(curriculum)
    terminal = read(curriculum / "pipeline_complete.json")
    if any(v["stop_reason"] != "budget_complete" for v in terminal["outcomes"].values()):
        raise ValueError("Curriculum has a safety/technical stop; inspect before advancing")
    gate_path = ROOT / "runs/fullpack-tier6-h40-campaign-gate-v1/summary.json"
    gate = read(gate_path)
    if gate["exit_code"] != 0:
        raise ValueError("40-turn test gate failed")
    for relative, digest in {**gate["source_files_sha256"], **gate["test_files_sha256"]}.items():
        if file_digest(ROOT / relative) != digest:
            raise ValueError(f"Gate input changed: {relative}")
    smoke_path = ROOT / "runs/fullpack-tier6-h40-transfer-smoke-v1/transfer.json"
    smoke = read(smoke_path)
    if smoke["script_sha256"] != file_digest(ROOT / "scripts/transfer_fullpack_policy40.py"):
        raise ValueError("Transfer implementation changed after smoke verification")
    if (
        smoke["exact_inference_states"] != 64
        or smoke["new_contract"]["game_config"]["max_turns"] != 40
    ):
        raise ValueError("Tier6/40-turn transfer smoke incomplete")
    exp.prepare(root)
    external = dict(previous["external_sha256"])
    for path in (
        SCRIPT,
        PLAN,
        ROOT / "scripts/continue_fullpack40.py",
        ROOT / "scripts/transfer_fullpack_policy40.py",
        gate_path,
        smoke_path,
        curriculum / "protocol.json",
        curriculum / "pipeline_complete.json",
    ):
        external[str(path)] = file_digest(path)
    sources = {}
    for target_seed, old_seed in zip(SEEDS, continuation.SEEDS):
        path = curriculum / f"candidate-{old_seed}/complete.json"
        done = read(path)
        # Selection is validation-only. No qualified checkpoint is required for
        # TRAINING initialization; fixed-final fallback is never a delivery claim.
        item = (
            done["selected"]["checkpoint"]
            if done["selected"] is not None
            else done["evaluations"][-1]["checkpoint"]
        )
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Curriculum checkpoint changed")
        sources[str(target_seed)] = {
            "checkpoint": item,
            "selection": "validation-qualified" if done["selected"] else "fixed-final-init-only",
        }
        external[str(path)] = file_digest(path)
        external[item["path"]] = item["sha256"]
    seal(
        root / "campaign.json",
        {
            "curriculum": str(curriculum),
            "sources": sources,
            "external_sha256": external,
            "max_turns": 40,
            "ordinary_shop_tier": 6,
            "candidate_initialization": "Tier5 weights; same feature/action meanings; new Adam",
            "candidate_guardrails": asdict(GuardrailConfig(stop_on_regression=False)),
            "candidate_guard_class": "ContinuingGuard; strict delivery gates remain unchanged",
            "segment_steps": STEP,
            "first_segment_steps": 2 * STEP,
            "generator_steps_each": STEP,
            "max_workers": 2,
            "batch_max_seconds": 14400,
            "test_screen": "3 validation-selected models; 500 episodes x 2 learned families each",
            "test_opening_gate": "All three selected; mean learned validation success >=80%",
            "test_used_for_checkpoint_selection": False,
            "test_used_for_campaign_stopping": True,
            "confirmation_required_after_screen": True,
            "overrides_to_base_protocol": {
                "no_automatic_expansion": False,
                "no_80_percent_acceptance_claim": "80% screening only; final confirmation pending",
                "scope": "Warm candidate transfer; fresh BC for all opponent generators",
                "candidate_stop_on_regression": "severe 3-check rule, not 2 moderate gate failures",
            },
        },
    )
    checked(root)


def batch(root, action, names, segment=0):
    pending, running = list(names), {}
    deadline = time.monotonic() + 14400
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    try:
        while pending or running:
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded four-hour batch deadline")
            while pending and len(running) < 2:
                name = str(pending.pop(0))
                log = (log_dir / f"{action}-{segment:03d}-{name}.log").open("x")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            str(SCRIPT),
                            "worker",
                            "--output",
                            str(root),
                            "--action",
                            action,
                            "--name",
                            name,
                            "--segment",
                            str(segment),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log.close()
                    raise
                running[name] = process, log
                print(f"Started {action}/{segment}/{name} pid={process.pid}", flush=True)
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                if code:
                    raise RuntimeError(f"Worker failed: {action}/{segment}/{name} exit={code}")
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


def segment_dir(root, index):
    return root / "segments" / f"segment{index:03d}"


def prepare_segment(root, index):
    p = checked(root)
    cp = sealed(root / "candidate_protocol.json")
    folder = segment_dir(root, index)
    folder.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for seed in SEEDS:
        if index == 1:
            item = p["sources"][str(seed)]["checkpoint"]
            warm = transfer.transfer(Path(item["path"]), root / f"warm-{seed}", seed)
            binding, expected = warm, None
        else:
            done = read(segment_dir(root, index - 1) / f"candidate-{seed}/complete.json")
            binding = done["evaluations"][-1]["checkpoint"]
            expected = str(Path(binding["path"]).with_suffix(".json"))
        args = dict(cp["configs"][str(seed)])
        args.update(
            seed=seed + (index - 1) * 1000,
            timesteps=STEP * (2 if index == 1 else 1),
            output_dir=str(folder / f"candidate-{seed}"),
            initialize_from=binding["path"],
            expected_initial_policy_sha256=binding["policy_sha256"],
        )
        cfg = configuration(6, **args)
        cfg.validate()
        jobs[str(seed)] = {
            "config": asdict(cfg),
            "source": binding,
            "expected_initial_validation": expected,
            "expected_initial_validation_sha256": file_digest(expected) if expected else None,
        }
    seal(folder / "jobs.json", {"jobs": jobs, "segment": index})


def candidate(root, seed, index):
    p = checked(root)
    jobs = sealed(segment_dir(root, index) / "jobs.json")
    job = jobs["jobs"][str(seed)]
    if file_digest(job["source"]["path"]) != job["source"]["sha256"]:
        raise ValueError("Initial weights changed")
    cfg = HorizonTrainingConfig(**job["config"])
    first = True

    def evaluator(model, leagues, **kwargs):
        nonlocal first
        actual = evaluate_suite(model, leagues, **kwargs)
        if first and job["expected_initial_validation"]:
            path = job["expected_initial_validation"]
            if file_digest(path) != job["expected_initial_validation_sha256"]:
                raise ValueError("Expected validation changed")
            expected = read(path)["evaluation"]
            for name in leagues:
                if (
                    actual["families"][name]["episode_results"]
                    != expected["families"][name]["episode_results"]
                ):
                    raise ValueError("Parent validation reload differs")
        first = False
        return actual

    train_guarded(
        cfg,
        GuardrailConfig(**p["candidate_guardrails"]),
        evaluator=evaluator,
        guard_factory=continuation.ContinuingGuard,
    )
    checked(root)


def selections(root, index):
    selections = {}
    for seed in SEEDS:
        guard = LearnedFirstGuard(GuardrailConfig(stop_on_regression=False))
        for previous in range(1, index + 1):
            done = read(segment_dir(root, previous) / f"candidate-{seed}/complete.json")
            for entry in done["evaluations"]:
                binding = entry["checkpoint"]
                if file_digest(binding["path"]) != binding["sha256"]:
                    raise ValueError("Validation checkpoint changed")
                record = read(Path(binding["path"]).with_suffix(".json"))
                if record["checkpoint"] != binding:
                    raise ValueError("Validation record binding differs")
                guard.consider(record["evaluation"], binding)
        selections[str(seed)] = guard.best
    seal(
        segment_dir(root, index) / "selection.json",
        {
            "selected": selections,
            "test_scores_opened_for_these_selections": False,
            "selection_uses_validation_only": True,
        },
    )
    return selections


def test_candidate(root, seed, index):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    checked(root)
    item = sealed(segment_dir(root, index) / "selection.json")["selected"][str(seed)]["checkpoint"]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Test model changed after selection")
    model = MaskablePPO.load(item["path"], device="cpu")
    if policy_digest(model.policy) != item["policy_sha256"]:
        raise ValueError("Reloaded model tensors differ")
    cp = sealed(root / "candidate_protocol.json")
    # Exact validation reload precedes the first test query for this model.
    expected = read(Path(item["path"]).with_suffix(".json"))["evaluation"]
    actual = evaluate_suite(model, cp["paths"]["validation"], episodes=200, seed=119000000)
    for name in cp["paths"]["validation"]:
        if (
            expected["families"][name]["episode_results"]
            != actual["families"][name]["episode_results"]
        ):
            raise ValueError("Selected model validation did not reload exactly")
    result = evaluate_suite(model, cp["paths"]["test"], episodes=500, seed=120000000)
    write_new(
        segment_dir(root, index) / f"test-{seed}.json",
        {"checkpoint": item, "validation_reload_exact": True, "evaluation": result},
    )


def screen(root, index):
    successes, count, reliable = 0, 0, True
    for seed in SEEDS:
        result = read(segment_dir(root, index) / f"test-{seed}.json")["evaluation"]
        for f in family_metrics(result).values():
            reliable &= (
                f["forcing"] <= 0.05 and f["truncation"] == 0 and f["no_purchase_loss"] <= 0.05
            )
        for name, f in result["families"].items():
            if name.startswith("test_"):
                count += len(f["episode_results"])
                successes += sum(r["success"] for r in f["episode_results"])
    if count != 3000:
        raise ValueError("Expected three models x two learned test families x 500 episodes")
    result = {
        "learned_successes": successes,
        "learned_episodes": count,
        "learned_ten_win_rate": successes / count,
        "all_families_reliable": bool(reliable),
        "screen_passed": bool(reliable and successes / count >= 0.8),
        "ordinary_shop_tier": 6,
        "pets": 60,
        "max_turns": 40,
        "confirmation_pending": True,
        "goal_complete": False,
    }
    write_new(segment_dir(root, index) / "screen.json", result)
    return result


def campaign(root, curriculum):
    setup()
    prepare(root, curriculum)
    batch(root, "generator", exp.GENERATORS)
    exp.prepare_candidates(root)
    tested = set()
    index = 1
    while True:
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise RuntimeError("Less than20GiB free; retain artifacts and request storage review")
        prepare_segment(root, index)
        batch(root, "candidate", SEEDS, index)
        reasons = {
            str(s): read(segment_dir(root, index) / f"candidate-{s}/complete.json")["stop_reason"]
            for s in SEEDS
        }
        chosen = selections(root, index)
        scores = {s: v["score"][0] if v else None for s, v in chosen.items()}
        write_new(
            segment_dir(root, index) / "segment_complete.json",
            {
                "stop_reasons": reasons,
                "selected_learned_validation": scores,
                "goal_complete": False,
            },
        )
        print(f"Segment {index}: selected learned validation {scores}", flush=True)
        if any(reason != "budget_complete" for reason in reasons.values()):
            write_new(root / "training_stopped.json", {"segment": index, "reasons": reasons})
            return
        if all(v is not None for v in chosen.values()) and fmean(scores.values()) >= 0.8:
            signature = tuple(chosen[str(s)]["checkpoint"]["policy_sha256"] for s in SEEDS)
            if signature not in tested:
                batch(root, "test", SEEDS, index)
                tested.add(signature)
                result = screen(root, index)
                if result["screen_passed"]:
                    write_new(
                        root / "reached80_screen.json",
                        {
                            **result,
                            "segment": index,
                            "selection": str(segment_dir(root, index) / "selection.json"),
                        },
                    )
                    return
        index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "campaign", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--action", choices=("generator", "candidate", "test"))
    parser.add_argument("--name")
    parser.add_argument("--segment", type=int, default=0)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "launch":
        if root.exists() or args.curriculum is None:
            raise ValueError("Need a fresh output and completed curriculum")
        if not (args.curriculum / "pipeline_complete.json").exists():
            raise ValueError("Curriculum still running; do not overlap training jobs")
        with root.with_suffix(".launch.log").open("x") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "campaign",
                    "--output",
                    str(root),
                    "--curriculum",
                    str(args.curriculum.resolve()),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_new(
            root.with_suffix(".launch.json"),
            {
                "pid": proc.pid,
                "max_workers": 2,
                "training_continues_across_budgets": True,
                "confirmation_required": True,
            },
        )
        print(f"Started full Tier6 campaign pid={proc.pid}", flush=True)
        return
    try:
        setup()
        if args.command == "campaign":
            campaign(root, args.curriculum.resolve())
        else:
            checked(root)
            if args.action == "generator":
                exp.generator(root, args.name)
            elif args.action == "candidate":
                candidate(root, int(args.name), args.segment)
            else:
                test_candidate(root, int(args.name), args.segment)
            checked(root)
    except Exception:
        if root.exists():
            path = (
                root
                / f"failure-{args.action or 'campaign'}-{args.segment}-{args.name or 'root'}.json"
            )
            write_new(path, {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
