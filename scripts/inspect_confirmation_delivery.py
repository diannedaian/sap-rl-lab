"""Reconstruct a bounded diagnostic sample AFTER frozen confirmation evaluation.

Select the earliest five failures, three successes and five forced/truncated
episodes per family, deduplicated. This outcome-stratified sample is NOT a rate
estimate, model selection or training. Reuse the existing replay implementation.
"""

import argparse
import json
from pathlib import Path

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded import verify_inputs
from sap_rl_lab.expanded_confirmation import read, save_new


def choose_rows(rows):
    ordered = sorted(rows, key=lambda row: row["seed"])
    if len({row["seed"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate episode seeds within one family")
    groups = [
        [row for row in ordered if not row["success"]][:5],
        [row for row in ordered if row["success"]][:3],
        [row for row in ordered if row["forced_end_turns"] or row["truncated"]][:5],
    ]
    chosen = {row["seed"]: row for group in groups for row in group}
    return [chosen[seed] for seed in sorted(chosen)]


def compare_replay(actual, expected):
    mapping = {
        "seed": "seed",
        "success": "success",
        "wins": "wins",
        "raw_return": "return",
        "actions": "actions",
        "action_counts": "action_counts",
        "forced_turns": "forced_end_turns",
        "truncated": "truncated",
    }
    for replay_key, evaluation_key in mapping.items():
        if actual[replay_key] != expected[evaluation_key]:
            raise ValueError(f"Diagnostic replay differs from held-out row: {replay_key}")


def run(root, output):
    from sap_rl_lab.expanded_diagnostics import inspect

    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = read(root / "protocol.json")
    verify_inputs(root, protocol)
    if (root / "aborted.json").exists():
        raise ValueError("Cannot inspect an aborted batch as current confirmation")
    if not read(root / "post-training/complete.json")["stages_complete"]:
        raise ValueError("All frozen evaluations must complete before diagnostic selection")
    selection_path = root / "selection.json"
    selection = read(selection_path)
    selection_hash = file_digest(selection_path)
    if selection["protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("Protocol changed after model selection")
    summary = read(root / "summary.json")
    if summary["selection_sha256"] != selection_hash:
        raise ValueError("Confirmation summary belongs to another model selection")
    name = selection["delivery"]
    chosen = selection["models"][name]
    if file_digest(chosen["path"]) != chosen["sha256"]:
        raise ValueError("Frozen delivery checkpoint changed")
    started = read(root / f"evaluation-{name}-started.json")
    if started != {"selection_sha256": selection_hash, "model_sha256": chosen["sha256"]}:
        raise ValueError("Evaluation does not belong to this frozen model")
    evaluation_path = root / f"evaluation-{name}.json"
    evaluation = read(evaluation_path)
    cases = []
    for split in ("test", "challenge"):
        for family, data in evaluation[split]["families"].items():
            league = protocol["paths"][split][family]
            if file_digest(league) != data["league_sha256"] or not data["deterministic"]:
                raise ValueError("Requires unchanged pools and deterministic original evaluation")
            for row in choose_rows(data["episode_results"]):
                cases.append({"split": split, "family": family, "expected": row})
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "delivery": name,
        "model_sha256": chosen["sha256"],
        "selection_sha256": selection_hash,
        "evaluation_sha256": file_digest(evaluation_path),
        "script_sha256": file_digest(__file__),
        "selection_rule": __doc__,
        "cases": cases,
    }
    save_new(output / "plan.json", metadata)
    completed = []
    try:
        for case in cases:
            split, family, row = case["split"], case["family"], case["expected"]
            label = f"{split}-{family}-{row['seed']}"
            directory = output / label
            inspect(chosen["path"], protocol["paths"][split][family], directory, 1, row["seed"])
            actual = read(directory / "summary.json")["episodes"][0]
            compare_replay(actual, row)
            completed.append({"directory": label, "reproduced": True, "summary": actual})
        verify_inputs(root, protocol)
        if (
            file_digest(chosen["path"]) != chosen["sha256"]
            or file_digest(selection_path) != selection_hash
            or file_digest(evaluation_path) != metadata["evaluation_sha256"]
        ):
            raise ValueError("Inputs changed during reconstruction")
        save_new(output / "complete.json", {"cases": completed, "manual_inspection_done": False})
    except Exception as error:
        save_new(output / "failed.json", {"error": repr(error), "completed_cases": completed})
        raise
    print(json.dumps({"delivery": name, "reproduced_episodes": len(completed)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.confirmation, args.output)
