"""Bounded historical-opponent pilot. No automatic promotion to a full experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .evaluation import file_digest
from .expanded import FAMILIES, run_arm, verify_inputs
from .expanded_confirmation import read, save_new
from .opponents import OpponentMixture, SnapshotLeague
from .round3 import source_archive
from .training import TrainingConfig, policy_digest

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "runs/expanded-confirmation-v3"
PILOT = ROOT / "runs/expanded-pilot-v8"
GENERATORS = {
    "train": {
        "pilot_action": PILOT / "action_cost/best_model.zip",
        "pilot_swap": PILOT / "swap_cost/best_model.zip",
        "control1811": PRIOR / "control-seed1811/best_model.zip",
    },
    "validation": {"control1907": PRIOR / "control-seed1907/best_model.zip"},
    "reserved_test": {"control2027": PRIOR / "control-seed2027/best_model.zip"},
}


def validate_roles(generators, initializers):
    """Reject duplicate tensor weights even when ZIP bytes or filenames differ."""
    seen = set()
    for entries in generators.values():
        for item in entries.values():
            digest = item["policy_sha256"]
            if digest in seen:
                raise ValueError("Generator weights overlap between roles or sources")
            seen.add(digest)
    if seen & {item["policy_sha256"] for item in initializers.values()}:
        raise ValueError("A learner initializer is also an opponent generator")


def profile_pool(path, metadata):
    data = read(path)
    snapshots = data["snapshots"]
    turns = Counter(s["turn"] for s in snapshots)
    species = Counter(p["spec_id"] for s in snapshots for p in s["pets"])
    keys = [(s["label"], s["turn"]) for s in snapshots]
    if Counter(keys) != Counter((s["label"], s["turn"]) for s in metadata):
        raise ValueError("Battle metadata does not align with captured snapshots")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate battle identity in a generated pool")
    identities = [json.dumps([s["turn"], s["pets"]], sort_keys=True) for s in snapshots]
    return {
        "snapshots": len(snapshots),
        "by_turn": dict(sorted(turns.items())),
        "pet_counts": dict(species),
        "pet_counts_by_turn": {
            str(t): dict(
                Counter(p["spec_id"] for s in snapshots if s["turn"] == t for p in s["pets"])
            )
            for t in sorted(turns)
        },
        "duplicate_fraction": 1 - len(set(identities)) / max(1, len(identities)),
        "missing_turns": [t for t in range(1, 31) if t not in turns],
        "fallback_bucket_by_turn": {
            str(t): max((s for s in turns if s <= t), default=None)
            for t in range(1, 31)
            if t not in turns
        },
        "wins_before_battle": dict(Counter(s["wins_before_battle"] for s in metadata)),
        "lives_before_battle": dict(Counter(s["lives_before_battle"] for s in metadata)),
        "warning": "Coverage audit, not proof of distinct strategic competence. "
        "Late-round populations are conditional on reaching that round. "
        "Matchmaking remains turn-only; wins/lives are metadata, not new matching rules.",
    }


def arm_configs(base, scripted, learned, initializer, expected_digest):
    if len(scripted) != 3 or len(learned) != 3:
        raise ValueError("50/50 mixture requires exactly three pools on each side")
    common = dict(base)
    common.update(initialize_from=initializer, expected_initial_policy_sha256=expected_digest)
    return {
        "scripted": {**common, "opponent_leagues": tuple(scripted)},
        "historical_mix": {**common, "opponent_leagues": tuple(scripted) + tuple(learned)},
    }


def prepare(output):
    import torch
    from sb3_contrib import MaskablePPO

    from .learned_pool import collect

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    prior = read(PRIOR / "protocol.json")
    initializers, generators = {}, {}
    reference_contract = None

    def freeze_model(path, name):
        nonlocal reference_contract
        model = MaskablePPO.load(path, device="cpu")
        contract = model.sap_environment_contract
        if reference_contract is None:
            reference_contract = contract
        if contract != reference_contract:
            raise ValueError("Opponent and learner rules differ")
        destination = output / "models" / f"{name}.zip"
        original_hash = file_digest(str(path))
        shutil.copyfile(path, destination)
        if file_digest(str(destination)) != original_hash:
            raise ValueError("Model copy changed")
        return {
            "path": str(destination),
            "original_path": str(path),
            "sha256": original_hash,
            "policy_sha256": policy_digest(model.policy),
            "environment_contract": contract,
        }

    for seed in (1811, 1907, 2027):
        initializers[str(seed)] = freeze_model(
            PRIOR / f"action_cost-seed{seed}/best_model.zip", f"initial-{seed}"
        )
    for role, sources in GENERATORS.items():
        generators[role] = {
            name: freeze_model(path, f"{role}-{name}") for name, path in sources.items()
        }
    validate_roles(generators, initializers)
    save_new(
        output / "design.json",
        {
            "generators": generators,
            "initializers": initializers,
            "pilot_decisions_per_arm": 1048576,
            "pilot_continuation_seed": 8101,
            "pilot_initial_seed": 1907,
            "train_generation_seed": 8_800_000,
            "train_episodes_per_generator": 120,
            "validation_generation_seed": 8_900_000,
            "validation_generation_episodes": 60,
            "validation_episode_seed": 9_000_000,
            "validation_episodes_per_family": 60,
            "scope": "Fixed historical mixture, not OSFP or an adaptive league. "
            "Only opponent distribution differs. Policy/value weights transferred, "
            "both optimizers reset identically. Two workers maximum, local CPU.",
            "test_boundary": "No reserved-test pool generation or evaluation in this pilot. "
            "Old v8-control challenge is diagnostic only. Before full confirmation, "
            "complete a multi-generator held-out test design; reserved control2027 "
            "alone does not establish broad generalization. Models share training "
            "ancestry/settings; generator-disjoint is not strategy-family independent.",
        },
    )
    paths = {"train_scripted": {}, "validation": {}, "train_learned": {}}
    for split, key in (("train", "train_scripted"), ("validation", "validation")):
        folder = output / "data" / key
        folder.mkdir(parents=True)
        for family in FAMILIES:
            destination = folder / f"{family}.json"
            shutil.copyfile(prior["paths"][split][family], destination)
            paths[key][family] = str(destination)
    against = OpponentMixture(
        [(1.0, SnapshotLeague.load(path)) for path in paths["train_scripted"].values()]
    )
    profiles = {}
    for role, seed, episodes in (("train", 8_800_000, 120), ("validation", 8_900_000, 60)):
        for name, item in generators[role].items():
            print(f"Generating {role}/{name}: {episodes} fresh episodes", flush=True)
            model = MaskablePPO.load(item["path"], device="cpu")
            metadata = []
            league, rows = collect(
                model,
                episodes=episodes,
                seed=seed,
                opponent_provider=against,
                snapshot_metadata=metadata,
            )
            folder = output / "data" / f"generated-{role}-{name}"
            folder.mkdir(parents=True)
            path = folder / "pool.json"
            league.save(path)
            save_new(
                folder / "generation.json",
                {
                    "source": item,
                    "snapshot_metadata": metadata,
                    "episode_results": rows,
                    "against_sha256": {p: file_digest(p) for p in paths["train_scripted"].values()},
                    "candidate_evaluation_performed": False,
                    "sampling": "Every battle including forced endings and unsuccessful episodes",
                },
            )
            key = "train_learned" if role == "train" else "validation"
            paths[key][name] = str(path)
            profiles[f"{role}/{name}"] = profile_pool(path, metadata)
    save_new(output / "pool_profiles.json", profiles)
    original = read(PRIOR / "action_cost-seed1907/run_manifest.json")["config"]
    original.update(
        timesteps=1048576,
        seed=8101,
        device="cpu",
        torch_threads=1,
        opponent_league="",
        opponent_leagues=(),
        validation_league="",
        validation_leagues=paths["validation"],
        validation_seed=9_000_000,
        validation_episodes=60,
        evaluation_interval=262144,
    )
    initial = initializers["1907"]
    configs = arm_configs(
        original,
        list(paths["train_scripted"].values()),
        list(paths["train_learned"].values()),
        initial["path"],
        initial["policy_sha256"],
    )
    for config in configs.values():
        TrainingConfig(**config).validate()
    protocol = {
        "base_config": {},
        "arms": configs,
        "paths": paths,
        "source_files_sha256": source_archive(output),
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "model_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "models").glob("*.zip"))
        },
        "selection": "Same four-family validation for both arms: three scripted, "
        "one generator-disjoint learned family, equal family weights; maximize "
        "unassisted success then raw return, earliest tie. Initial model eligible.",
        "pilot_gate": "Both budgets complete, paired initial weights and initial "
        "validation identical, reload exact, no numerical failures. Human review "
        "of coverage, forcing and learning curves before any confirmation launch. "
        "No significance claim from this single pilot pair.",
        "formal_plan": "Three paired starts (1811,1907,2027), 8388608 fresh decisions "
        "each; restart from original initializers, not pilot outputs. Not auto-launched.",
    }
    save_new(output / "protocol.json", protocol)
    print("Prepared and frozen pilot; no final test opened", flush=True)


def verify(output):
    protocol = read(output / "protocol.json")
    verify_inputs(output, protocol)
    for relative, digest in protocol["model_sha256"].items():
        if file_digest(str(output / relative)) != digest:
            raise ValueError(f"Frozen model changed: {relative}")
    return protocol


def execute_arm(output, arm):
    verify(output)
    run_arm(output, arm)
    verify(output)


def summarize_pilot(output):
    import torch
    from sb3_contrib import MaskablePPO

    from .evaluation import evaluate_suite

    torch.set_num_threads(1)
    protocol = verify(output)
    summary, initial_rows = {}, []
    for arm in protocol["arms"]:
        folder = output / arm
        complete = read(output / f"{arm}_complete.json")
        config = protocol["arms"][arm]
        manifest = read(folder / "run_manifest.json")
        history = read(folder / "validation_history.json")
        if history[-1]["timesteps"] != config["timesteps"]:
            raise ValueError("Incomplete pilot budget")
        if manifest["initial_policy_sha256"] != config["expected_initial_policy_sha256"]:
            raise ValueError("Initial policy mismatch")
        initial_rows.append(read(folder / history[0]["evaluation_file"])["families"])
        chosen = [row for row in history if row["selected"]][-1]
        if file_digest(str(folder / "best_model.zip")) != complete["best_model_sha256"]:
            raise ValueError("Selected model changed")
        model = MaskablePPO.load(folder / "best_model.zip", device="cpu")
        actual = evaluate_suite(
            model,
            config["validation_leagues"],
            episodes=config["validation_episodes"],
            seed=config["validation_seed"],
        )
        expected = read(folder / chosen["evaluation_file"])
        for family in actual["families"]:
            if (
                actual["families"][family]["episode_results"]
                != expected["families"][family]["episode_results"]
            ):
                raise ValueError(f"Reload mismatch: {arm}/{family}")
        save_new(output / f"{arm}_reload.json", actual)
        summary[arm] = {
            "selected_validation": chosen,
            "elapsed_seconds": complete["elapsed_seconds"],
            "reload_exact": True,
        }
    if initial_rows[0] != initial_rows[1]:
        raise ValueError("Paired starting evaluations differ")
    save_new(
        output / "pilot_summary.json",
        {
            "arms": summary,
            "paired_initial_evaluation_exact": True,
            "held_out_test_opened": False,
            "formal_training_started": False,
            "status": "Pilot complete; coverage/performance review still required",
        },
    )


def pilot(output):
    started = time.monotonic()
    prepare(output)
    children = []
    try:
        for arm in ("scripted", "historical_mix"):
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "sap_rl_lab.generalization",
                        "arm",
                        "--output",
                        str(output),
                        "--arm",
                        arm,
                    ],
                    cwd=ROOT,
                )
            )
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(f"Pilot worker failure: {codes}; no full training launched")
        summarize_pilot(output)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
    print(f"Pilot finished in {time.monotonic() - started:.0f}s; review required", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "arm", "summary"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("scripted", "historical_mix"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "pilot":
        pilot(output)
    elif args.command == "summary":
        summarize_pilot(output)
    else:
        if args.arm is None:
            parser.error("arm command requires --arm")
        execute_arm(output, args.arm)


if __name__ == "__main__":
    main()
