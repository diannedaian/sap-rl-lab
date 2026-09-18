"""One bounded post-test diagnosis; never train, replace models, or edit frozen inputs."""

import argparse
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.historical_confirmation import check_tests, checked


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2)


def select_cases(root, selection, tests):
    target = "historical_mix-seed2027"
    evaluation = read(root / f"test-{target}.json")
    cases = []
    for family, result in evaluation["families"].items():
        rows = sorted(result["episode_results"], key=lambda r: r["seed"])
        groups = {
            "forced": [r for r in rows if r["forced_end_turns"]][:2],
            "success": [r for r in rows if r["success"] and not r["forced_end_turns"]][:1],
            "ordinary_loss": [r for r in rows if not r["success"] and not r["forced_end_turns"]][
                :1
            ],
        }
        for group, chosen in groups.items():
            for row in chosen:
                cases.append(
                    {
                        "model": target,
                        "family": family,
                        "group": group,
                        "expected": row,
                        "league": tests["paths"][family],
                    }
                )
    # Matched game seeds and opponent pools, not a claim of matched later states.
    for case in [c for c in cases if c["group"] == "forced"][::2][:3]:
        for name in ("scripted-seed2027", "historical_mix-seed1907"):
            rows = read(root / f"test-{name}.json")["families"][case["family"]]["episode_results"]
            expected = next(r for r in rows if r["seed"] == case["expected"]["seed"])
            cases.append({**case, "model": name, "group": "matched_control", "expected": expected})
    for case in cases:
        case["model_path"] = selection["models"][case["model"]]["path"]
    return cases


def turn_summary(record):
    output = []
    for index, step in enumerate(record["steps"]):
        if not step["info"].get("forced_end_turn"):
            continue
        turn = step["state"]["turn"]
        segment = [
            (i, s) for i, s in enumerate(record["steps"][: index + 1]) if s["state"]["turn"] == turn
        ]
        actions = Counter(s["action"].split(":")[0] for _, s in segment)
        repeats, seen = [], set()
        for i, s in segment:
            key = json.dumps(
                {k: v for k, v in s["state"].items() if k != "actions_this_turn"}, sort_keys=True
            )
            if key in seen:
                repeats.append(i)
            seen.add(key)
        output.append(
            {
                "turn": turn,
                "last_step": index,
                "actions": dict(actions),
                "repeated_state_steps": repeats,
                "gold_at_forcing": step["state"]["gold"],
                "tail": [
                    {
                        "step": i,
                        "action": s["action"],
                        "value": s["value"],
                        "gold": s["state"]["gold"],
                        "top_actions": s["top_actions"],
                    }
                    for i, s in segment[-12:]
                ],
                "terminated": step["terminated"],
                "truncated": step["truncated"],
            }
        )
    return output


def run(root, output):
    protocol, tests = checked(root), check_tests(root)
    if not (root / "pipeline_complete.json").exists():
        raise ValueError("Requires completed frozen test, not ongoing model selection")
    selection = read(root / "selection.json")
    for item in selection["models"].values():
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen selected model changed")
    output.mkdir(parents=True, exist_ok=False)
    cases = select_cases(root, selection, tests)
    save_new(
        output / "plan.json",
        {
            "cases": cases,
            "script_sha256": file_digest(__file__),
            "selection_sha256": file_digest(root / "selection.json"),
            "scope": "First two forced, first unforced success and ordinary loss per "
            "test family for mixed2027, plus six matched-seed controls. Outcome-selected "
            "diagnostics, not another generalization test or checkpoint selection.",
        },
    )
    curves = {}
    for name in protocol["arms"]:
        history = read(root / name / "validation_history.json")
        rows = []
        for item in history:
            rows.append(
                {
                    "step": item["timesteps"],
                    "file": item["evaluation_file"],
                    "unassisted_success": item["success_without_forcing_rate"],
                    "success": item["success_rate"],
                    "return": item["mean_return"],
                    "forced_events": item["forced_end_turns"],
                    "max_family_forced_episode_rate": max(
                        f["forced_episode_rate"] for f in item["families"].values()
                    ),
                    "selected": item["selected"],
                }
            )
        curves[name] = rows
    save_new(output / "checkpoint_curves.json", curves)
    completed = []
    for case in cases:
        seed = case["expected"]["seed"]
        label = f"{case['model']}-{case['family']}-{seed}"
        directory = output / label
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(case["model_path"], case["league"], directory, 1, seed)
        record = read(directory / f"episode-{seed}.json")
        compare_replay(record["summary"], case["expected"])
        gamma = read(Path(case["model_path"]).parent / "run_manifest.json")["config"]["gamma"]
        audit = summarize_episode(record, gamma)
        completed.append(
            {
                "directory": label,
                "model": case["model"],
                "family": case["family"],
                "group": case["group"],
                "summary": record["summary"],
                "behavior": audit,
                "forced_turns": turn_summary(record),
                "critic_above_remaining_win_bound": [
                    i
                    for i, s in enumerate(record["steps"])
                    if s["value"] > 10 - s["state"]["wins"] + 1e-6
                ],
            }
        )
        print(f"Reproduced {label}", flush=True)
    checked(root)
    check_tests(root)
    save_new(
        output / "replay_audit.json",
        {"cases": completed, "reconstruction_exact": True, "training_performed": False},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.root).resolve(), Path(args.output).resolve())
