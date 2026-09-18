"""Continue all three 40-turn lineages; budget boundaries are not goal completion."""

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.horizon40 import HorizonTrainingConfig
from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig, train_guarded, write_new
from sap_rl_lab.fullpack.recipe import LearnedFirstGuard
from sap_rl_lab.round3 import source_archive

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SEEDS = (84101, 84201, 84301)
STEPS = 1_048_576
spec = importlib.util.spec_from_file_location(
    "recovery_runner", ROOT / "scripts/resume_fullpack40.py"
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
# Reuse only the checked-file reader and bounded two-worker process supervisor.
# Point its child command at THIS script; no frozen source files are changed.
recovery.SCRIPT = SCRIPT


def read(path):
    return json.loads(Path(path).read_text())


class ContinuingGuard(LearnedFirstGuard):
    """Keep strict delivery selection, but stop learning only on sustained severe deterioration."""

    def __init__(self, config):
        super().__init__(config)
        self.peak_learned = 0.0
        self.severe_streak = 0

    def consider(self, result, checkpoint):
        decision = super().consider(result, checkpoint)
        f = decision["families"]
        learned = sum(v["success"] for k, v in f.items() if k.startswith("validation_")) / 2
        severe = (
            max(v["forcing"] for v in f.values()) > 0.50
            or max(v["no_purchase_loss"] for v in f.values()) > 0.25
            or (self.peak_learned >= 0.30 and learned < self.peak_learned - 0.20)
        )
        self.severe_streak = self.severe_streak + 1 if severe else 0
        self.peak_learned = max(self.peak_learned, learned)
        decision["severe_streak"] = self.severe_streak
        if self.severe_streak >= 3:
            decision.update(stop=True, stop_reason="sustained_severe_deterioration")
        return decision


def prepare(output, parent):
    previous = recovery.checked(parent)
    if not (parent / "pipeline_complete.json").exists():
        raise ValueError("Parent pipeline must be terminal before continuing")
    output.mkdir(parents=True, exist_ok=False)
    external = dict(previous["external_sha256"])
    for path in (
        SCRIPT,
        parent / "protocol.json",
        parent / "pipeline_complete.json",
        ROOT / "docs/FULLPACK_CONTINUE.md",
    ):
        external[str(path)] = file_digest(path)
    configs, sources = {}, {}
    for seed in SEEDS:
        done_path = parent / f"candidate-{seed}/complete.json"
        done = read(done_path)
        binding = done["evaluations"][-1]["checkpoint"]
        if file_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("Latest saved checkpoint changed")
        external[str(done_path)] = file_digest(done_path)
        external[binding["path"]] = binding["sha256"]
        external[str(Path(binding["path"]).with_suffix(".json"))] = file_digest(
            Path(binding["path"]).with_suffix(".json")
        )
        cfg = dict(previous["configs"][str(seed)])
        cfg.update(
            seed=cfg["seed"] + 1000,
            timesteps=STEPS,
            output_dir=str(output / f"candidate-{seed}"),
            initialize_from=binding["path"],
            expected_initial_policy_sha256=binding["policy_sha256"],
        )
        HorizonTrainingConfig(**cfg).validate()
        configs[str(seed)], sources[str(seed)] = cfg, binding
    guards = asdict(GuardrailConfig(stop_on_regression=False))
    write_new(
        output / "protocol.json",
        {
            "parent": str(parent),
            "configs": configs,
            "sources": sources,
            "guardrails": guards,
            "additional_ppo_budget": 3 * STEPS,
            "external_sha256": external,
            "source_files_sha256": source_archive(output),
            "maximum_workers": 2,
            "maximum_seconds": 14400,
            "test_scores_opened": False,
            "delivery_gate_unchanged": "forced <=5% per family; no cuts; strength gates kept",
            "learning_stop": "3 consecutive validations: forcing >50%, no-buy loss >25%, "
            "or learned success >20pp below peak >=30%; any truncation still stops",
            "scope": "Tier5 continuation only; cannot establish the Tier6 80% goal",
            "older_eligible_models_retained": True,
        },
    )
    write_new(output / "seal.json", {"sha256": file_digest(output / "protocol.json")})
    recovery.checked(output)


def worker(output, seed):
    p = recovery.checked(output)
    cfg = HorizonTrainingConfig(**p["configs"][str(seed)])
    source_record = read(Path(p["sources"][str(seed)]["path"]).with_suffix(".json"))
    from sap_rl_lab.fullpack.evaluation import evaluate_suite

    initial_checked = False

    def evaluator(model, leagues, **kwargs):
        nonlocal initial_checked
        actual = evaluate_suite(model, leagues, **kwargs)
        if not initial_checked:
            for name in leagues:
                if (
                    actual["families"][name]["episode_results"]
                    != source_record["evaluation"]["families"][name]["episode_results"]
                ):
                    raise ValueError("Parent policy did not reproduce its exact validation rows")
            initial_checked = True
        return actual

    result = train_guarded(
        cfg, GuardrailConfig(**p["guardrails"]), guard_factory=ContinuingGuard, evaluator=evaluator
    )
    recovery.checked(output)
    write_new(
        output / f"worker-{seed}-complete.json",
        {
            "stop_reason": result["stop_reason"],
            "actual_timesteps": result["actual_timesteps"],
            "selected": result["selected"],
            "parent_validation_exact": initial_checked,
        },
    )


def pipeline(output, parent):
    started = time.monotonic()
    prepare(output, parent)
    recovery.batch(output, started + 14400)
    outcomes = {str(s): read(output / f"worker-{s}-complete.json") for s in SEEDS}
    write_new(
        output / "pipeline_complete.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "outcomes": outcomes,
            "test_scores_opened": False,
            "fullpack_80_target_met": False,
            "next": "Inspect validation then advance to Tier6; no new user approval required.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "pipeline", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    output = args.output.resolve()
    parent = args.parent.resolve() if args.parent else None
    if args.command == "launch":
        if output.exists() or parent is None:
            raise ValueError("A new output and completed parent are required")
        with output.with_suffix(".launch.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    "pipeline",
                    "--output",
                    str(output),
                    "--parent",
                    str(parent),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_new(
            output.with_suffix(".launch.json"),
            {"pid": process.pid, "maximum_workers": 2, "maximum_seconds": 14400},
        )
        print(f"Started continuation pid={process.pid}", flush=True)
        return
    try:
        if args.command == "worker":
            worker(output, args.seed)
        else:
            signal.signal(signal.SIGALRM, recovery.experiment.wall_timeout)
            signal.alarm(14400)
            try:
                pipeline(output, parent)
            finally:
                signal.alarm(0)
    except Exception:
        if output.exists():
            write_new(
                output / f"failure-{args.seed or 'pipeline'}.json",
                {"traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
