"""One bounded, write-once post-training pipeline; never starts or extends training.

Wait for all six completion artifacts, then use the existing verified stages.
No automatic retry, candidate replacement, gate changes or stable-release claim.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded import FAMILIES, verify_inputs


def stage_plan(protocol):
    candidate = protocol["candidate_arm"]
    seeds = protocol["seeds"]
    if candidate not in {"action_cost", "swap_cost"} or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Requires one prespecified treatment and three distinct seeds")
    names = [f"{arm}-seed{seed}" for seed in seeds for arm in ("control", candidate)]
    if set(protocol["arms"]) != set(names):
        raise ValueError("Protocol does not contain exactly the six paired runs")
    return [
        [("freeze", None)],
        [("reload", name) for name in names],
        [("evaluate", name) for name in names + list(FAMILIES)],
        [("summarize", None)],
    ]


def training_ready(root, names):
    if (root / "aborted.json").exists():
        raise ValueError("Training batch has been aborted")
    ready = True
    for name in names:
        path = root / f"{name}_complete.json"
        if not path.exists():
            ready = False
        elif json.loads(path.read_text()).get("training_complete") is not True:
            raise ValueError(f"Invalid training completion artifact: {name}")
    return ready


def save_new(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def finish(root, wait_seconds):
    if not 0 <= wait_seconds <= 14400:
        raise ValueError("Wait budget must be between zero and four hours")
    root = Path(root).resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    plan = stage_plan(protocol)
    verify_inputs(root, protocol)
    if (root / "selection.json").exists() or list(root.glob("evaluation-*-started.json")):
        raise ValueError("This fresh pipeline cannot resume existing selection/evaluation")
    logs = root / "post-training"
    logs.mkdir(exist_ok=False)
    protocol_hash = file_digest(protocol_path)
    save_new(
        logs / "started.json",
        {
            "protocol_sha256": protocol_hash,
            "runner_sha256": file_digest(__file__),
            "wait_seconds": wait_seconds,
            "plan": plan,
            "max_parallel_inference_processes": 2,
            "training_started_by_runner": False,
        },
    )

    def execute(task):
        stage, name = task
        if file_digest(protocol_path) != protocol_hash:
            raise ValueError("Protocol changed after pipeline start")
        verify_inputs(root, protocol)
        command = [
            sys.executable,
            "-m",
            "sap_rl_lab.expanded_confirmation",
            stage,
            "--output",
            str(root),
        ]
        if name is not None:
            command += ["--name", name]
        label = stage + (f"-{name}" if name else "")
        print(f"Starting {label}", flush=True)
        with (logs / f"{label}.log").open("x") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {label}", flush=True)

    try:
        deadline = time.monotonic() + wait_seconds
        names = [name for _, name in plan[1]]
        while not training_ready(root, names):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Training completion wait expired; no training is restarted")
            time.sleep(min(30, remaining))
        # Each batch is a barrier: ALL six exact reloads finish before ANY test.
        # Existing stages separately verify model/full-budget/source provenance.
        for tasks in plan:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(execute, tasks))
        save_new(logs / "complete.json", {"stages_complete": True, "goal_complete": False})
        print(
            "Post-training stages complete; rule/replay/delivery audit still required", flush=True
        )
    except Exception as error:
        save_new(logs / "failed.json", {"error": repr(error), "goal_complete": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    finish(args.output, args.wait_seconds)
