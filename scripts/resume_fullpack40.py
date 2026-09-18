"""Bounded recovery of the three Tier5 candidates under the authorized 40-turn limit."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from sap_rl_lab.fullpack import experiment, imitation
from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.horizon40 import HorizonTrainingConfig, configuration, transfer_model
from sap_rl_lab.fullpack.ppo_guardrails import GuardrailConfig, train_guarded, write_new
from sap_rl_lab.fullpack.recipe import LearnedFirstGuard
from sap_rl_lab.round3 import source_archive

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/fullpack-tier5-v1"
SCRIPT = Path(__file__).resolve()
SEEDS = (84101, 84201, 84301)
SOURCE_STEPS = {84101: 1_310_720, 84201: 1_048_576, 84301: 0}
TARGET_STEPS = 2_097_152


def read(path):
    return json.loads(Path(path).read_text())


def checked(output):
    path = output / "protocol.json"
    if file_digest(path) != read(output / "seal.json")["sha256"]:
        raise ValueError("Recovery protocol changed")
    p = read(path)
    for filename, digest in p["external_sha256"].items():
        if file_digest(filename) != digest:
            raise ValueError(f"Frozen recovery input changed: {filename}")
    for relative, digest in p["source_files_sha256"].items():
        if file_digest(ROOT / relative) != digest:
            raise ValueError(f"Frozen recovery code changed: {relative}")
    return p


def prepare(output):
    experiment.checked(PARENT)
    cp_path = PARENT / "candidate_protocol.json"
    if file_digest(cp_path) != read(cp_path.with_suffix(".seal.json"))["sha256"]:
        raise ValueError("Parent candidate protocol changed")
    cp, parent = read(cp_path), read(PARENT / "protocol.json")
    gate_path = ROOT / "runs/fullpack-h40-rules-gate-v2/summary.json"
    gate = read(gate_path)
    if gate["exit_code"] != 0:
        raise ValueError("40-turn regression gate failed")
    for rel, digest in {**gate["source_files_sha256"], **gate["test_files_sha256"]}.items():
        if file_digest(ROOT / rel) != digest:
            raise ValueError(f"Code/test changed after gate: {rel}")
    audit_path = ROOT / "runs/fullpack-horizon40-audit-v1/summary.json"
    audit = read(audit_path)
    if not audit["original_200_rows_exact"] or audit["new_truncations"]:
        raise ValueError("Stopped validation family did not pass the horizon audit")
    output.mkdir(parents=True, exist_ok=False)
    external = dict(cp["data_sha256"])
    for path in (
        SCRIPT,
        cp_path,
        PARENT / "protocol.json",
        gate_path,
        audit_path,
        ROOT / "docs/FULLPACK_HORIZON40.md",
    ):
        external[str(path)] = file_digest(path)
    for item in parent["demonstrations"].values():
        external[item["path"]] = item["sha256"]
    configs, sources = {}, {}
    for seed in SEEDS:
        args = dict(cp["configs"][str(seed)])
        args.update(
            timesteps=TARGET_STEPS - SOURCE_STEPS[seed],
            output_dir=str(output / f"candidate-{seed}"),
        )
        configs[str(seed)] = asdict(configuration(5, **args))
        if SOURCE_STEPS[seed]:
            files = list(
                (PARENT / f"candidate-{seed}").glob(f"eval*-step{SOURCE_STEPS[seed]}.json")
            )
            if len(files) != 1:
                raise ValueError("Need exactly one saved continuation checkpoint")
            binding = read(files[0])["checkpoint"]
            if file_digest(binding["path"]) != binding["sha256"]:
                raise ValueError("Continuation checkpoint changed")
            sources[str(seed)] = binding
            external[binding["path"]] = binding["sha256"]
            external[str(files[0])] = file_digest(files[0])
    protocol = {
        "configs": configs,
        "sources": sources,
        "source_steps": SOURCE_STEPS,
        "target_cumulative_steps": TARGET_STEPS,
        "additional_ppo_budget": sum(TARGET_STEPS - n for n in SOURCE_STEPS.values()),
        "guardrails": parent["guardrails"],
        "bc": parent["bc"],
        "demonstrations": parent["demonstrations"],
        "bc_opponent_leagues": list(parent["paths"]["train"].values()),
        "external_sha256": external,
        "source_files_sha256": source_archive(output),
        "test_scores_opened": False,
        "maximum_workers": 2,
        "maximum_seconds": 14400,
        "opponent_pools": "unchanged frozen pools; nearest earlier saved turn after pool horizon",
        "initialization": "latest reproducible checkpoint, not validation/test-best; fresh Adam",
        "discarded_unsaved_B_updates": "B reached 1310720 but last saved weights are 1048576",
        "scope": "Tier5 recovery only; Tier6 and the 80% goal remain incomplete",
    }
    write_new(output / "protocol.json", protocol)
    write_new(output / "seal.json", {"sha256": file_digest(output / "protocol.json")})
    checked(output)


def worker(output, seed):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = checked(output)
    cfg = HorizonTrainingConfig(**p["configs"][str(seed)])
    source = p["sources"].get(str(seed))
    if source is None:
        # C never started; reproduce the existing BC recipe/data in a NEW directory.
        old_config = experiment.configuration(
            seed=seed, opponent_leagues=tuple(p["bc_opponent_leagues"])
        )
        teacher = imitation.fresh_model(old_config)
        try:
            bc = imitation.fit(
                teacher,
                imitation.load_data(Path(p["demonstrations"]["train"]["path"])),
                imitation.load_data(Path(p["demonstrations"]["validation"]["path"])),
                output / f"bc-{seed}",
                seed=seed,
                **p["bc"],
            )
            source = bc["selected"]
        finally:
            teacher.get_env().close()
    old = MaskablePPO.load(source["path"], device="cpu")
    initial, binding = transfer_model(old, cfg)
    initial_path = output / f"initial-{seed}.zip"
    try:
        initial.save(initial_path)
    finally:
        initial.get_env().close()
    write_new(
        output / f"transfer-{seed}.json",
        {
            **binding,
            "source": source,
            "path": str(initial_path),
            "sha256": file_digest(initial_path),
        },
    )
    cfg = replace(
        cfg,
        initialize_from=str(initial_path),
        expected_initial_policy_sha256=binding["policy_sha256"],
    )
    print(f"Starting 40-turn candidate {seed}: {cfg.timesteps} new PPO steps", flush=True)
    result = train_guarded(cfg, GuardrailConfig(**p["guardrails"]), guard_factory=LearnedFirstGuard)
    checked(output)
    # A deliberate guard stop is a recorded outcome, not a process crash that
    # kills a different healthy candidate. Technical exceptions still halt all.
    write_new(
        output / f"worker-{seed}-complete.json",
        {
            "stop_reason": result["stop_reason"],
            "actual_timesteps": result["actual_timesteps"],
            "selected": result["selected"],
            "target_achieved": False,
        },
    )
    print(f"Completed {seed}: {result['stop_reason']}", flush=True)


def batch(output, deadline):
    pending, active = list(SEEDS), {}
    try:
        while pending or active:
            if time.monotonic() >= deadline:
                raise TimeoutError("Four-hour recovery limit")
            while pending and len(active) < 2:
                seed = pending.pop(0)
                log = (output / f"candidate-{seed}.log").open("x")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            str(SCRIPT),
                            "worker",
                            "--output",
                            str(output),
                            "--seed",
                            str(seed),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log.close()
                    raise
                active[seed] = process, log
                print(f"Started candidate {seed}, pid={process.pid}", flush=True)
            for seed, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[seed]
                if code:
                    raise RuntimeError(f"Technical failure: candidate {seed}, exit={code}")
            if active:
                time.sleep(1)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


def pipeline(output):
    started = time.monotonic()
    prepare(output)
    batch(output, started + 14400)
    checked(output)
    outcomes = {str(s): read(output / f"worker-{s}-complete.json") for s in SEEDS}
    write_new(
        output / "pipeline_complete.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "outcomes": outcomes,
            "all_budgets_completed": all(
                v["stop_reason"] == "budget_complete" for v in outcomes.values()
            ),
            "test_scores_opened": False,
            "fullpack_80_target_met": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "pipeline", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "launch":
        if output.exists():
            raise FileExistsError(output)
        with output.with_suffix(".launch.log").open("x") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", str(SCRIPT), "pipeline", "--output", str(output)],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_new(
            output.with_suffix(".launch.json"),
            {
                "pid": process.pid,
                "output": str(output),
                "maximum_workers": 2,
                "maximum_seconds": 14400,
            },
        )
        print(f"Started recovery pipeline pid={process.pid}", flush=True)
        return
    try:
        if args.command == "worker":
            worker(output, args.seed)
        else:
            signal.signal(signal.SIGALRM, experiment.wall_timeout)
            signal.alarm(14400)
            try:
                pipeline(output)
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
