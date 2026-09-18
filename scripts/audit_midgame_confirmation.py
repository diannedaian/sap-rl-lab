"""Audit completed v5 evidence and reconstruct a bounded post-test replay sample.

This is read-only with respect to the experiment. It never trains, replaces a
model, changes a reward, or uses test results to choose a checkpoint. Output must
be a new directory. The diagnostic sample is outcome-selected, not a rate estimate.
"""

import argparse
from contextlib import redirect_stdout
from math import isclose
from pathlib import Path
from statistics import fmean

from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read, save_new
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.midgame_confirmation import ARMS, SEEDS, check_protocol


def select_rows(rows):
    ordered = sorted(rows, key=lambda row: row["seed"])
    if len({r["seed"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate evaluation seeds")
    groups = (
        [r for r in ordered if not r["success"]][:4],
        [r for r in ordered if r["success"]][:2],
        [r for r in ordered if r["forced_end_turns"] or r["truncated"]][:1],
    )
    selected = {r["seed"]: r for group in groups for r in group}
    return [selected[seed] for seed in sorted(selected)]


def exact_family_counts(result, episodes, seed):
    rows = result["episode_results"]
    if len(rows) != episodes or [r["seed"] for r in rows] != list(range(seed, seed + episodes)):
        raise ValueError("Missing, reordered or duplicated test rows")
    counts = {
        "successes": sum(r["success"] for r in rows),
        "forced_episodes": sum(r["forced_end_turns"] > 0 for r in rows),
        "forced_events": sum(r["forced_end_turns"] for r in rows),
        "truncations": sum(r["truncated"] for r in rows),
        "unassisted_successes": sum(r["success"] and not r["forced_end_turns"] for r in rows),
        "episodes": episodes,
        "battles": sum(r["battles"] for r in rows),
    }
    rates = {
        "success_rate": counts["successes"] / episodes,
        "forced_episode_rate": counts["forced_episodes"] / episodes,
        "forced_end_turn_rate": counts["forced_events"] / max(1, counts["battles"]),
        "truncation_rate": counts["truncations"] / episodes,
        "success_without_forcing_rate": counts["unassisted_successes"] / episodes,
        "mean_wins": fmean(r["wins"] for r in rows),
        "mean_return": fmean(r["return"] for r in rows),
        "mean_episode_actions": fmean(r["actions"] for r in rows),
    }
    for key, value in rates.items():
        if not isclose(result[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Reported metric disagrees with integer rows: {key}")
    for row in rows:
        if sum(row["action_counts"].values()) != row["actions"]:
            raise ValueError("Action counts do not add up")
        if bool(row["success"]) != (row["wins"] >= 10):
            raise ValueError("Success flag does not mean ten wins")
        if row["forced_end_turns"] > row["battles"]:
            raise ValueError("More forced transitions than battles")
    return {**counts, **rates}


def check_environment_contract(actual, saved):
    # Evaluation records transition semantics, while model metadata additionally
    # records permission opt-in and the policy implementation revision.
    required = {"schema_version", "catalog_id", "catalog_sha256", "game_config"}
    if set(actual) != required or any(actual[k] != saved[k] for k in required):
        raise ValueError("Test used a different environment")


def control_to_inspect(counts):
    """Diagnostic target only; never changes delivery or its checkpoint selection."""
    controls = sorted(name for name in counts if name.startswith("scripted-seed"))
    if not controls:
        return None

    def forced(name):
        return sum(f["forced_episodes"] for f in counts[name].values())

    name = max(controls, key=forced)
    return name if forced(name) else None


def zero_success_models(counts):
    """Include complete failures in diagnostics even if they never force combat."""
    return sorted(
        name
        for name, families in counts.items()
        if families and all(f["successes"] == 0 for f in families.values())
    )


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = check_protocol(root)
    complete = read(root / "pipeline_complete.json")
    if not complete["training_and_evaluation_complete"] or (root / "pipeline_failed.json").exists():
        raise ValueError("Only completed, non-failed confirmation can be audited")
    design = read(root / "design.json")
    selection = read(root / "selection.json")
    expected_names = {f"{a}-seed{s}" for s in SEEDS for a in ARMS}
    if set(selection["models"]) != expected_names:
        raise ValueError("Incomplete model selection")
    if selection["protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("Selection belongs to a different protocol")
    counts, evidence_hashes = {}, {}
    for name in sorted(expected_names):
        item = selection["models"][name]
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Selected model changed")
        reload = read(root / f"reload-{name}.json")
        if not reload["all_validation_rows_exact"] or reload["model_sha256"] != item["sha256"]:
            raise ValueError("Missing exact model reload")
        if reload["selection_sha256"] != file_digest(root / "selection.json"):
            raise ValueError("Reload belongs to a different selection")
        path = root / f"test-{name}.json"
        result = read(path)
        evidence_hashes[name] = file_digest(path)
        if set(result["families"]) != set(protocol["paths"]["test"]):
            raise ValueError("Incomplete held-out family coverage")
        counts[name] = {}
        for family, data in result["families"].items():
            if data["league_sha256"] != file_digest(protocol["paths"]["test"][family]):
                raise ValueError("Test used a different opponent pool")
            check_environment_contract(
                data["environment_contract"], protocol["environment_contract"]
            )
            if not data["deterministic"]:
                raise ValueError("Test used a different action-selection mode")
            counts[name][family] = exact_family_counts(
                data, design["test_episodes"], design["test_seed"]
            )
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "exact_counts.json", counts)
    name = selection["delivery"]
    item = selection["models"][name]
    result = read(root / f"test-{name}.json")
    cases = [
        {"model": name, "group": "delivery", "family": family, "expected": row}
        for family, data in result["families"].items()
        for row in select_rows(data["episode_results"])
    ]
    control = control_to_inspect(counts)
    if control:
        control_result = read(root / f"test-{control}.json")
        for family, data in control_result["families"].items():
            row = next(
                (
                    r
                    for r in sorted(data["episode_results"], key=lambda r: r["seed"])
                    if r["forced_end_turns"]
                ),
                None,
            )
            if row is not None:
                cases.append(
                    {"model": control, "group": "control_forced", "family": family, "expected": row}
                )
    zero_success = zero_success_models(counts)
    seen = {(c["model"], c["family"], c["expected"]["seed"]) for c in cases}
    for candidate in zero_success:
        candidate_result = read(root / f"test-{candidate}.json")
        family = sorted(candidate_result["families"])[0]
        row = min(candidate_result["families"][family]["episode_results"], key=lambda r: r["seed"])
        key = (candidate, family, row["seed"])
        if key not in seen:
            cases.append(
                {"model": candidate, "group": "zero_success", "family": family, "expected": row}
            )
            seen.add(key)
    save_new(
        output / "replay_plan.json",
        {
            "selection_sha256": file_digest(root / "selection.json"),
            "model": name,
            "model_sha256": item["sha256"],
            "diagnostic_control": control,
            "diagnostic_zero_success_models": zero_success,
            "cases": cases,
            "script_sha256": file_digest(__file__),
            "test_file_sha256": evidence_hashes,
            "sampling": "Earliest four losses, two successes and one forced/truncated episode "
            "per family, deduplicated. Additionally, first forced episode per family from "
            "the scripted control with most forced test episodes (lower seed on ties). "
            "At most five control cases. Also the earliest episode of the alphabetically "
            "first family for each model with zero successes across all test families, "
            "deduplicated (at most six additional cases). "
            "All checkpoints remain validation-selected; "
            "no delivery change and no rate inference from this diagnostic sample.",
        },
    )
    replays = []
    for case in cases:
        family, row = case["family"], case["expected"]
        inspected = selection["models"][case["model"]]
        gamma = read(Path(inspected["path"]).parent / "run_manifest.json")["config"]["gamma"]
        label = f"{case['model']}-{family}-{row['seed']}"
        directory = output / label
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(inspected["path"], protocol["paths"]["test"][family], directory, 1, row["seed"])
        record = read(directory / f"episode-{row['seed']}.json")
        compare_replay(record["summary"], row)
        replays.append(
            {
                "directory": label,
                "model": case["model"],
                "group": case["group"],
                "family": family,
                "summary": record["summary"],
                "behavior": summarize_episode(record, gamma),
                "forced_steps": [
                    {
                        "index": i,
                        "state": s["state"],
                        "action": s["action"],
                        "value": s["value"],
                        "top_actions": s["top_actions"],
                        "terminated": s["terminated"],
                        "truncated": s["truncated"],
                    }
                    for i, s in enumerate(record["steps"])
                    if s["info"].get("forced_end_turn")
                ],
            }
        )
        print(f"Exact replay: {label}", flush=True)
    check_protocol(root)
    save_new(
        output / "replay_audit.json",
        {
            "cases": replays,
            "reconstruction_exact": True,
            "audited_test_rows": sum(
                c["episodes"] for families in counts.values() for c in families.values()
            ),
            "training_updates": 0,
            "model_reselection": False,
            "limitation": "Values are reconstructed with the selected checkpoint, not historical "
            "training-time critic values. Repeated states ignore only the shop-action counter.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
