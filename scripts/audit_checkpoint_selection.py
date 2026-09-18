"""Audit the completed selector comparison, then reconstruct <=30 selected replays.

No training, selection changes, primary test reads, or automatic recipe approval.
Uses raw episode counts to check the comparison, including missing seed failures.
"""

import argparse
from contextlib import redirect_stdout
from pathlib import Path
from statistics import fmean

import compare_checkpoint_selection as comparison
from audit_exploration_pilot import choose_cases
from audit_stability_confirmation import recount_suite
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import compare_reload, read, save_new
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.experiments import paired_comparison
from sap_rl_lab.exploration_pilot import metrics
from sap_rl_lab.historical_confirmation import copy_checked
from sap_rl_lab.training import midgame_checkpoint_score


def result_binding(result, started, item, selection_sha):
    expected = {"model_sha256": item["sha256"], "selection_sha256": selection_sha}
    if started != expected or any(result.get(k) != value for k, value in expected.items()):
        raise ValueError("Evaluation model or selection binding differs")


def comparison_from_rows(results, counts, final_empty):
    """Use audited integer ten-win counts, not a copied mean in the final report."""
    import numpy as np

    pairs = {}
    for seed in comparison.SEEDS:
        seed = str(seed)
        raw, row = results[seed], {}
        for selector in comparison.SELECTORS:
            if raw[selector] is None:
                row[selector] = None
                continue
            family_counts = counts[seed][selector]
            row[selector] = {**metrics(raw[selector]), "counts": family_counts}
            row[selector]["learned_success"] = fmean(
                family_counts[f]["successes"] / family_counts[f]["episodes"]
                for f in comparison.GENERATORS
            )
        row["learned_difference"] = None
        if row["reliability_first"] is not None:
            row["learned_difference"] = (
                row["reliability_first"]["learned_success"] - row["win_first"]["learned_success"]
            )
            row["family_comparisons"] = {
                family: paired_comparison(
                    raw["reliability_first"]["families"][family],
                    raw["win_first"]["families"][family],
                    resamples=2000,
                )
                for family in raw["win_first"]["families"]
            }
        row["final_no_purchase_loss_rate"] = final_empty[seed]
        pairs[seed] = row
    differences = [row["learned_difference"] for row in pairs.values()]
    mean, interval = None, None
    if all(d is not None for d in differences):
        mean = fmean(differences)
        rng = np.random.default_rng(25_500_000)
        boot = rng.choice(differences, size=(10_000, 5)).mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
    gates = comparison.acceptance(pairs)
    return {
        "pairs": pairs,
        "gates": gates,
        "mean_learned_difference": mean,
        "seed_bootstrap_95_percent": interval,
        "missing_reliable_seeds": [
            s for s, row in pairs.items() if row["reliability_first"] is None
        ],
        "practical_screen_passed": all(gates.values()),
        "recipe_confirmed": False,
        "evidence_review_required": True,
        "new_training_decisions": 0,
    }


def verify_summary(summary, expected):
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Comparison summary differs from raw evidence: {field}")


def prepare_replay_inputs(output, parent, selection):
    """Supply the legacy inspector's adjacent-manifest layout without editing inputs.

    Periodic weights live one directory below their training manifest. Copy the
    exact weights and that run's frozen manifest together in the new audit only.
    """
    inputs = {}
    for name, item in selection["models"].items():
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Replay source model changed")
        source_manifest = parent / f"control-seed{item['seed']}" / "run_manifest.json"
        folder = output / "replay-inputs" / name
        model_path, manifest_path = folder / "model.zip", folder / "run_manifest.json"
        model_sha = copy_checked(item["path"], model_path)
        manifest_sha = copy_checked(source_manifest, manifest_path)
        if model_sha != item["sha256"]:
            raise ValueError("Copied replay model differs from selected model")
        inputs[name] = {
            "path": str(model_path),
            "sha256": model_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "source_path": item["path"],
            "source_manifest_path": str(source_manifest),
        }
    return inputs


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent, protocol, selection = comparison.frozen(root)
    if (root / "pipeline_failed.json").exists() or not read(root / "pipeline_complete.json")[
        "all_stages_complete"
    ]:
        raise ValueError("Requires completed, non-failed selector comparison")
    comparison.require_reloads(root, selection)
    selection_sha = file_digest(root / "selection.json")
    inventory = read(root / "inventory.json")
    hashes, curves, tests, reloads, cases = {}, {}, {}, {}, []

    def evidence(path):
        hashes[str(path)] = file_digest(path)
        return read(path)

    for filename in (
        "registration.json",
        "inventory.json",
        "selection.json",
        "pipeline_complete.json",
    ):
        evidence(root / filename)
    final_empty = {}
    for seed in comparison.SEEDS:
        seed = str(seed)
        validated = evidence(root / f"validated-{seed}.json")
        originals = {r["id"]: r for r in inventory["models"][seed]}
        candidates = validated["candidates"]
        if len(candidates) != 12 or {r["id"] for r in candidates} != set(originals):
            raise ValueError("Missing actual checkpoint grid entries")
        if comparison.choose(candidates) != selection["pairs"][seed]:
            raise ValueError("Frozen selector does not match complete grid")
        curves[seed] = {}
        for item in candidates:
            if any(item[k] != v for k, v in originals[item["id"]].items()):
                raise ValueError("Candidate inventory binding changed")
            comparison.check_model(item, parent, protocol, f"control-seed{seed}")
            if file_digest(item["validation_path"]) != item["validation_sha256"]:
                raise ValueError("Candidate validation changed")
            data = evidence(Path(item["validation_path"]))
            counted = recount_suite(
                data,
                protocol["paths"]["validation"],
                200,
                comparison.VAL_SEED,
                protocol["environment_contract"],
            )
            if (
                counted != item["counts"]
                or metrics(data) != item["metrics"]
                or midgame_checkpoint_score(data) != tuple(item["score"])
            ):
                raise ValueError("Candidate ranking or metrics differ from raw episodes")
            if item["cached_validation"]:
                compare_reload(data, evidence(Path(item["cached_validation"])))
            curves[seed][item["id"]] = counted
            if item["id"] == f"{seed}-final_model":
                if item["timesteps"] != 8_388_608:
                    raise ValueError("Final model budget incomplete")
                final_empty[seed] = metrics(data)["worst_family_no_purchase_loss_rate"]

    selected_results = {}
    for name, item in selection["models"].items():
        for stage, split, episodes, episode_seed in (
            ("reload", "validation", 200, comparison.VAL_SEED),
            ("test", "test", 400, comparison.TEST_SEED),
        ):
            data = evidence(root / f"{stage}-{name}.json")
            started = evidence(root / f"{stage}-{name}-started.json")
            result_binding(data, started, item, selection_sha)
            counted = recount_suite(
                data,
                protocol["paths"][split],
                episodes,
                episode_seed,
                protocol["environment_contract"],
            )
            if counted != data["recounted_families"]:
                raise ValueError("Inference recount differs from rows")
            if stage == "reload":
                if not data["all_validation_rows_exact"]:
                    raise ValueError("Selected validation reload failed")
                compare_reload(data, evidence(Path(item["validation_path"])))
                reloads[name] = counted
            else:
                tests[name], selected_results[name] = counted, data
                for case in choose_cases(data["families"]):
                    cases.append(
                        {
                            "model": name,
                            "path": item["path"],
                            "model_sha256": item["sha256"],
                            **case,
                        }
                    )
    if len(cases) > 30:
        raise ValueError("Replay diagnostic cap exceeded")
    results, counts = {}, {}
    for seed, choices in selection["pairs"].items():
        results[seed] = {s: selected_results[n] if n else None for s, n in choices.items()}
        counts[seed] = {s: tests[n] if n else None for s, n in choices.items()}
    expected = comparison_from_rows(results, counts, final_empty)
    verify_summary(evidence(root / "comparison_summary.json"), expected)
    totals = {
        "candidate_validation_rows_including_cached": sum(
            f["episodes"]
            for grid in curves.values()
            for result in grid.values()
            for f in result.values()
        ),
        "selected_reload_rows": sum(f["episodes"] for r in reloads.values() for f in r.values()),
        "selected_test_rows": sum(f["episodes"] for r in tests.values() for f in r.values()),
    }
    if totals["candidate_validation_rows_including_cached"] != 48_000 or not (
        totals["selected_reload_rows"] <= 8_000 and totals["selected_test_rows"] <= 20_000
    ):
        raise ValueError("Evaluation budgets differ from registration")
    output.mkdir(parents=True, exist_ok=False)
    replay_inputs = prepare_replay_inputs(output, parent, selection)
    for item in replay_inputs.values():
        hashes[item["path"]] = hashes[item["source_path"]] = item["sha256"]
        hashes[item["manifest_path"]] = hashes[item["source_manifest_path"]] = item[
            "manifest_sha256"
        ]
    save_new(
        output / "counts.json", {"candidate_validation": curves, "reload": reloads, "test": tests}
    )
    save_new(
        output / "plan.json",
        {
            "script_sha256": file_digest(__file__),
            "selection_sha256": selection_sha,
            "registration_sha256": file_digest(root / "registration.json"),
            "evidence_sha256": hashes,
            "cases": cases,
            "replay_inputs": replay_inputs,
            "replay_layout": "Byte-identical selected weights and their frozen training "
            "manifest copied side by side in this audit. Original checkpoint layout unchanged.",
            "selection": "Per distinct chosen model: earliest empty loss (else loss), earliest "
            "forced/truncated episode, earliest success; deduplicate. At most 30. Outcome-selected "
            "diagnostics, never used as population-rate evidence or for reselection.",
        },
    )
    replays = []
    for case in cases:
        row, family = case["expected"], case["family"]
        label = f"{case['model']}-{family}-{row['seed']}"
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(
                replay_inputs[case["model"]]["path"],
                protocol["paths"]["test"][family],
                output / label,
                1,
                row["seed"],
            )
        record = read(output / label / f"episode-{row['seed']}.json")
        compare_replay(record["summary"], row)
        if file_digest(case["path"]) != case["model_sha256"]:
            raise ValueError("Replay model changed")
        replays.append(
            {
                "directory": label,
                "model": case["model"],
                "family": family,
                "summary": record["summary"],
                "behavior": summarize_episode(record, protocol["base_config"]["gamma"]),
            }
        )
        print(f"Exact selector-comparison replay: {label}", flush=True)
    comparison.frozen(root)
    for path, digest in hashes.items():
        if file_digest(path) != digest:
            raise ValueError("Evidence changed during audit")
    save_new(
        output / "summary.json",
        {
            "all_counts_exact": True,
            "all_replays_exact": True,
            "all_summary_fields_reconstructed": True,
            **totals,
            "replays": replays,
            "training_updates": 0,
            "model_reselection": False,
            "recipe_confirmed": False,
            "missing_reliable_seeds": expected["missing_reliable_seeds"],
            "note": "Raw-count and replay consistency, not independent game-client parity or "
            "universal generalization. Human interpretation remains required.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
