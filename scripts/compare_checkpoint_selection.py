"""Frozen, read-only selector comparison on the five original control runs.

Register before primary testing. Run only after the primary pipeline has exited.
Never reads primary test metrics or trains; no missing-selection fallback.
"""

import argparse
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from audit_stability_confirmation import (
    check_training_configuration,
    history_selection,
    recount_suite,
)

from sap_rl_lab.evaluation import evaluate_suite, file_digest
from sap_rl_lab.expanded_confirmation import compare_reload, read, save_new
from sap_rl_lab.experiments import paired_comparison
from sap_rl_lab.exploration_pilot import metrics
from sap_rl_lab.historical_confirmation import fallback_counts
from sap_rl_lab.midgame import ROOT
from sap_rl_lab.stability_confirmation import GENERATORS, SEEDS, verify
from sap_rl_lab.training import midgame_checkpoint_score, policy_digest

SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/CHECKPOINT_SELECTION_PLAN.md"
SELECTORS = ("win_first", "reliability_first")
GRID = tuple(838_856 * i for i in range(1, 11))
VAL_SEED, TEST_SEED = 22_000_000, 25_000_000
PARENT_SHA = "af38e83eacc26603d9d8ad1e4b3cb49aa93789bd7113b4bbaf677ca7aab9daf4"


def eligible(summary):
    return (
        summary["worst_family_forced_episode_rate"] <= 0.02
        and summary["worst_family_no_purchase_loss_rate"] <= 0.01
        and summary["worst_family_truncation_rate"] == 0
    )


def choose(candidates):
    """Both selectors use this same grid; ties prefer earlier actual save steps."""
    if not candidates or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("Missing or duplicate checkpoint candidates")
    ordered = sorted(candidates, key=lambda r: (r["timesteps"], r["id"]))
    reliable = [r for r in ordered if eligible(r["metrics"])]
    return {
        "win_first": max(ordered, key=lambda r: tuple(r["score"]))["id"],
        "reliability_first": (
            max(reliable, key=lambda r: tuple(r["score"]))["id"] if reliable else None
        ),
    }


def primary_not_tested(parent):
    # Check names only, never read the primary test metrics.
    if list(parent.glob("test-*.json")) or (parent / "confirmation_summary.json").exists():
        raise ValueError("Registration requires no primary tests started")


def register(output, parent):
    primary_not_tested(parent)
    verify(parent)
    if file_digest(parent / "protocol.json") != PARENT_SHA:
        raise ValueError("Unexpected parent protocol")
    sources = [
        SCRIPT,
        PLAN,
        *[
            SCRIPT.parent / f"{name}.py"
            for name in (
                "audit_stability_confirmation",
                "audit_exploration_pilot",
                "audit_midgame_confirmation",
                "inspect_confirmation_delivery",
                "summarize_policy_replays",
            )
        ],
    ]
    primary_not_tested(parent)
    output.mkdir(parents=True, exist_ok=False)
    save_new(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent": str(parent),
            "parent_protocol_sha256": PARENT_SHA,
            "registered_before_primary_tests": True,
            "source_sha256": {str(p): file_digest(p) for p in sources},
            "seeds": list(SEEDS),
            "selectors": list(SELECTORS),
            "periodic_steps": list(GRID),
            "additional_candidates": ["best_model", "final_model"],
            "validation_episodes_per_family": 200,
            "validation_seed": VAL_SEED,
            "test_episodes_per_family": 400,
            "test_seed": TEST_SEED,
            "new_training_decisions": 0,
            "max_workers": 2,
            "episode_caps": {"periodic_validation": 40_000, "reload": 8_000, "test": 20_000},
            "family_bootstrap_resamples": 2000,
            "seed_bootstrap_resamples": 10_000,
            "seed_bootstrap_rng": 25_500_000,
            "selection_rule": "same grid; reliability eligibility then v5 score; "
            "earlier steps/id ties",
            "failure_policy": "Missing eligible checkpoints are failures, no fallback or retry.",
            "primary_test_metrics_read": False,
        },
    )


def checked(output):
    reg = read(output / "registration.json")
    if reg["parent_protocol_sha256"] != PARENT_SHA:
        raise ValueError("Registration parent changed")
    for path, digest in reg["source_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError(f"Registered code or plan changed: {path}")
    parent = Path(reg["parent"])
    if file_digest(parent / "protocol.json") != PARENT_SHA:
        raise ValueError("Parent protocol changed")
    return parent, verify(parent)


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        return SimpleNamespace(**json.loads(archive.read("data")))


def check_model(item, parent, protocol, name):
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Model changed")
    model = metadata(item["path"])
    if model.num_timesteps != item["timesteps"]:
        raise ValueError("Actual saved model steps differ")
    expected = {
        **protocol["base_config"],
        **protocol["arms"][name],
        "output_dir": str(parent / name),
    }
    check_training_configuration(read(parent / name / "run_manifest.json"), expected, model)


def require_parent_complete(parent):
    if (parent / "pipeline_failed.json").exists():
        raise ValueError("Primary pipeline failed; do not continue automatically")
    if not read(parent / "pipeline_complete.json")["training_and_evaluation_complete"]:
        raise ValueError("Primary pipeline not complete")


def index(output):
    parent, p = checked(output)
    require_parent_complete(parent)
    primary_selection = read(parent / "selection.json")
    if primary_selection["protocol_sha256"] != PARENT_SHA:
        raise ValueError("Primary selection protocol changed")
    inventory, evidence = (
        {},
        {str(parent / "selection.json"): file_digest(parent / "selection.json")},
    )
    for seed in SEEDS:
        name = f"control-seed{seed}"
        folder = parent / name
        done = read(parent / f"{name}_complete.json")
        history = read(folder / "validation_history.json")
        if not done["training_complete"] or history[-1]["timesteps"] != 8_388_608:
            raise ValueError("Control training incomplete")
        selected = history_selection(history)
        primary_item = primary_selection["models"][name]
        reload_path = parent / f"reload-{name}.json"
        reloaded = read(reload_path)
        if not (
            primary_item["selected"] == selected
            and primary_item["path"] == done["best_model"]
            and primary_item["sha256"] == done["best_model_sha256"]
            and reloaded["all_validation_rows_exact"]
            and reloaded["model_sha256"] == done["best_model_sha256"]
            and reloaded["selection_sha256"] == file_digest(parent / "selection.json")
        ):
            raise ValueError("Cached best validation lacks an exact model-bound reload")
        compare_reload(reloaded, read(folder / selected["evaluation_file"]))
        evidence[str(reload_path)] = file_digest(reload_path)
        paths = [folder / "checkpoints" / f"sap_ppo_{step}_steps.zip" for step in GRID]
        if set(paths) != set((folder / "checkpoints").glob("*.zip")):
            raise ValueError("Periodic saved grid differs from registration")
        candidates = []
        for path, steps in zip(paths, GRID):
            candidates.append(
                {
                    "id": f"{seed}-{path.stem}",
                    "path": str(path),
                    "sha256": file_digest(path),
                    "timesteps": steps,
                    "cached_validation": None,
                }
            )
        for field, point in (("best_model", selected), ("final_model", history[-1])):
            path = Path(done[field])
            if path != folder / f"{field}.zip" or file_digest(path) != done[field + "_sha256"]:
                raise ValueError("Completed model binding differs")
            validation = folder / point["evaluation_file"]
            data = read(validation)
            recount_suite(data, p["paths"]["validation"], 200, VAL_SEED, p["environment_contract"])
            if tuple(point["selection_score"]) != midgame_checkpoint_score(data):
                raise ValueError("Cached validation score differs from history")
            candidates.append(
                {
                    "id": f"{seed}-{field}",
                    "path": str(path),
                    "sha256": done[field + "_sha256"],
                    "timesteps": point["timesteps"],
                    "cached_validation": str(validation),
                }
            )
            evidence[str(validation)] = file_digest(validation)
        for item in candidates:
            check_model(item, parent, p, name)
        inventory[str(seed)] = candidates
        for path in (
            parent / f"{name}_complete.json",
            folder / "validation_history.json",
            folder / "run_manifest.json",
        ):
            evidence[str(path)] = file_digest(path)
    save_new(
        output / "inventory.json",
        {
            "models": inventory,
            "evidence_sha256": evidence,
            "registration_sha256": file_digest(output / "registration.json"),
        },
    )


def indexed(output):
    parent, protocol = checked(output)
    inv = read(output / "inventory.json")
    if inv["registration_sha256"] != file_digest(output / "registration.json"):
        raise ValueError("Inventory registration changed")
    for path, digest in inv["evidence_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError("Indexed evidence changed")
    return parent, protocol, inv


def load(item, parent, protocol, seed):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    check_model(item, parent, protocol, f"control-seed{seed}")
    model = MaskablePPO.load(item["path"], device="cpu")
    if model.sap_environment_contract != protocol["environment_contract"]:
        raise ValueError("Saved environment differs")
    return model


def validate(output, seed):
    parent, p, inv = indexed(output)
    folder = output / f"validation-{seed}"
    folder.mkdir(exist_ok=False)
    evaluated = []
    for item in inv["models"][str(seed)]:
        model = load(item, parent, p, seed)
        if item["cached_validation"]:
            data = read(item["cached_validation"])
        else:
            data = evaluate_suite(model, p["paths"]["validation"], episodes=200, seed=VAL_SEED)
        counts = recount_suite(
            data, p["paths"]["validation"], 200, VAL_SEED, p["environment_contract"]
        )
        path = folder / f"{item['id']}.json"
        save_new(path, data)
        evaluated.append(
            {
                **item,
                "validation_path": str(path),
                "validation_sha256": file_digest(path),
                "policy_sha256": policy_digest(model.policy),
                "counts": counts,
                "score": midgame_checkpoint_score(data),
                "metrics": metrics(data),
            }
        )
        del model
    indexed(output)
    save_new(
        output / f"validated-{seed}.json",
        {
            "candidates": evaluated,
            "choices": choose(evaluated),
            "inventory_sha256": file_digest(output / "inventory.json"),
        },
    )


def freeze(output):
    _, p, inv = indexed(output)
    pairs, selected, hashes = {}, {}, {}
    for seed in SEEDS:
        path = output / f"validated-{seed}.json"
        result = read(path)
        if result["inventory_sha256"] != file_digest(output / "inventory.json"):
            raise ValueError("Validation belongs to different inventory")
        candidates = result["candidates"]
        originals = {r["id"]: r for r in inv["models"][str(seed)]}
        if {r["id"] for r in candidates} != set(originals) or len(candidates) != 12:
            raise ValueError("Both selectors require all twelve actual checkpoints")
        for item in candidates:
            if any(item[k] != value for k, value in originals[item["id"]].items()):
                raise ValueError("Validation model binding differs")
            if (
                file_digest(item["path"]) != item["sha256"]
                or file_digest(item["validation_path"]) != item["validation_sha256"]
            ):
                raise ValueError("Candidate or validation changed")
            data = read(item["validation_path"])
            counts = recount_suite(
                data, p["paths"]["validation"], 200, VAL_SEED, p["environment_contract"]
            )
            if (
                item["metrics"] != metrics(data)
                or item["counts"] != counts
                or tuple(item["score"]) != midgame_checkpoint_score(data)
            ):
                raise ValueError("Selection evidence differs from raw validation")
        choices = choose(candidates)
        if choices != result["choices"]:
            raise ValueError("Selector changed")
        pairs[str(seed)] = choices
        for item in candidates:
            if item["id"] in choices.values():
                selected[item["id"]] = {**item, "seed": seed}
        hashes[str(path)] = file_digest(path)
    save_new(
        output / "selection.json",
        {
            "pairs": pairs,
            "models": selected,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_records_sha256": hashes,
            "inventory_sha256": file_digest(output / "inventory.json"),
            "primary_test_metrics_read": False,
        },
    )


def frozen(output):
    parent, protocol, _ = indexed(output)
    selection = read(output / "selection.json")
    if selection["inventory_sha256"] != file_digest(output / "inventory.json"):
        raise ValueError("Selection inventory changed")
    if set(selection["pairs"]) != {str(s) for s in SEEDS}:
        raise ValueError("Missing seed in frozen selection")
    for path, digest in selection["validation_records_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError("Frozen validation records changed")
    expected_models = {}
    for seed in SEEDS:
        validated = read(output / f"validated-{seed}.json")
        choices = choose(validated["candidates"])
        if choices != selection["pairs"][str(seed)]:
            raise ValueError("Frozen choices differ from registered selectors")
        for item in validated["candidates"]:
            if item["id"] in choices.values():
                expected_models[item["id"]] = {**item, "seed": seed}
    if selection["models"] != expected_models:
        raise ValueError("Frozen models differ from selected validation evidence")
    for item in selection["models"].values():
        if (
            file_digest(item["path"]) != item["sha256"]
            or file_digest(item["validation_path"]) != item["validation_sha256"]
        ):
            raise ValueError("Frozen model or validation changed")
    return parent, protocol, selection


def require_reloads(output, selection):
    for name, item in selection["models"].items():
        result = read(output / f"reload-{name}.json")
        if not (
            result["all_validation_rows_exact"]
            and result["model_sha256"] == item["sha256"]
            and result["selection_sha256"] == file_digest(output / "selection.json")
        ):
            raise ValueError("All frozen selections must reload before any auxiliary test")


def inference(output, name, stage):
    parent, p, selection = frozen(output)
    item = selection["models"][name]
    if stage == "test":
        require_reloads(output, selection)
    split, episodes, seed = (
        ("test", 400, TEST_SEED) if stage == "test" else ("validation", 200, VAL_SEED)
    )
    save_new(
        output / f"{stage}-{name}-started.json",
        {
            "model_sha256": item["sha256"],
            "selection_sha256": file_digest(output / "selection.json"),
        },
    )
    model = load(item, parent, p, item["seed"])
    if policy_digest(model.policy) != item["policy_sha256"]:
        raise ValueError("Loaded policy differs from validation")
    data = evaluate_suite(model, p["paths"][split], episodes=episodes, seed=seed)
    counts = recount_suite(data, p["paths"][split], episodes, seed, p["environment_contract"])
    if stage == "reload":
        compare_reload(data, read(item["validation_path"]))
        data["all_validation_rows_exact"] = True
    data.update(
        model_sha256=item["sha256"],
        selection_sha256=file_digest(output / "selection.json"),
        recounted_families=counts,
    )
    data["opponent_fallback_coverage"] = {
        family: fallback_counts(
            data["families"][family]["episode_results"],
            {s["turn"] for s in read(path)["snapshots"]},
        )
        for family, path in p["paths"][split].items()
    }
    frozen(output)
    save_new(output / f"{stage}-{name}.json", data)


def acceptance(pairs):
    complete = set(pairs) == {str(s) for s in SEEDS} and all(
        row["reliability_first"] is not None for row in pairs.values()
    )
    differences = [
        r["learned_difference"] for r in pairs.values() if r["learned_difference"] is not None
    ]
    return {
        "all_five_reliable_selections_exist": complete,
        "all_reliability_first_test_families_pass": complete
        and all(eligible(row["reliability_first"]) for row in pairs.values()),
        "no_final_empty_relapse": complete
        and all(r["final_no_purchase_loss_rate"] <= 0.01 for r in pairs.values()),
        "mean_learned_performance_not_lower": complete and fmean(differences) >= 0,
        "at_least_four_nonnegative_pairs": complete and sum(d >= 0 for d in differences) >= 4,
    }


def summarize(output):
    import numpy as np

    parent, p, selection = frozen(output)
    require_reloads(output, selection)
    pairs = {}
    for seed, choices in selection["pairs"].items():
        results, row = {}, {}
        for selector, name in choices.items():
            if name is None:
                row[selector] = None
                continue
            data = read(output / f"test-{name}.json")
            item = selection["models"][name]
            if data["model_sha256"] != item["sha256"] or data["selection_sha256"] != file_digest(
                output / "selection.json"
            ):
                raise ValueError("Auxiliary test binding differs")
            counts = recount_suite(
                data, p["paths"]["test"], 400, TEST_SEED, p["environment_contract"]
            )
            results[selector], row[selector] = data, {**metrics(data), "counts": counts}
            row[selector]["learned_success"] = fmean(
                data["families"][f]["success_rate"] for f in GENERATORS
            )
        row["learned_difference"] = None
        if len(results) == 2:
            row["learned_difference"] = (
                row["reliability_first"]["learned_success"] - row["win_first"]["learned_success"]
            )
            row["family_comparisons"] = {
                family: paired_comparison(
                    results["reliability_first"]["families"][family],
                    results["win_first"]["families"][family],
                    resamples=2000,
                )
                for family in p["paths"]["test"]
            }
        final = read(parent / f"control-seed{seed}" / "validation_history.json")[-1]
        data = read(parent / f"control-seed{seed}" / final["evaluation_file"])
        row["final_no_purchase_loss_rate"] = metrics(data)["worst_family_no_purchase_loss_rate"]
        pairs[seed] = row
    differences = [r["learned_difference"] for r in pairs.values()]
    mean, interval = None, None
    if all(d is not None for d in differences):
        mean = fmean(differences)
        samples = np.random.default_rng(25_500_000).choice(differences, size=(10_000, 5)).mean(1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    gates = acceptance(pairs)
    save_new(
        output / "comparison_summary.json",
        {
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
            "limitations": "Finite shared opponent ancestry and late-turn fallback; "
            "episode CIs differ from training-seed CIs. No survivor-only average.",
        },
    )


def batch(output, stage, names):
    for offset in range(0, len(names), 2):
        children = []
        try:
            for name in names[offset : offset + 2]:
                print(f"Starting selector comparison {stage}: {name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(SCRIPT),
                            stage,
                            "--output",
                            str(output),
                            "--name",
                            str(name),
                        ],
                        cwd=ROOT,
                    )
                )
            deadline = time.monotonic() + 7200
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError("Auxiliary worker failed; no automatic retry")
                if time.monotonic() > deadline:
                    raise TimeoutError("Auxiliary pair exceeded two hours")
                time.sleep(1)
            if any(c.returncode != 0 for c in children):
                raise RuntimeError("Auxiliary worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)


def pipeline(output):
    started = time.monotonic()
    parent, _ = checked(output)
    require_parent_complete(parent)
    save_new(
        output / "pipeline_started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat()}
    )
    try:
        index(output)
        batch(output, "validate", list(SEEDS))
        freeze(output)
        names = list(frozen(output)[2]["models"])
        batch(output, "reload", names)
        batch(output, "test", names)
        summarize(output)
        save_new(
            output / "pipeline_complete.json",
            {
                "all_stages_complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "evidence_review_required": True,
            },
        )
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "pipeline", "validate", "reload", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default=str(ROOT / "runs/stability-confirmation-v1"))
    parser.add_argument("--name")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "register":
        register(output, Path(args.parent).resolve())
    elif args.command == "pipeline":
        pipeline(output)
    elif args.command == "validate":
        validate(output, int(args.name))
    else:
        inference(output, args.name, args.command)
