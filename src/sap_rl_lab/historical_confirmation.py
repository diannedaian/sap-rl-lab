"""Finite historical-mixture confirmation, sealed opponents and paired evaluation.

Two local CPU workers maximum. Fail closed; never retry or extend a training budget.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from statistics import fmean

from .evaluation import evaluate_suite, file_digest
from .expanded import FAMILIES, run_arm, verify_inputs
from .expanded_confirmation import read, save_new
from .generalization import ROOT, arm_configs, profile_pool, validate_roles, verify
from .opponents import OpponentMixture, SnapshotLeague, build_scripted_league
from .round3 import source_archive
from .training import TrainingConfig, policy_digest, train

PILOT_ROOT = ROOT / "runs/historical-opponents-pilot-v1"
SEEDS = (1811, 1907, 2027)
ARMS = ("scripted", "historical_mix")


def fallback_counts(rows, available_turns):
    """One battle per turn, starting at turn one; do not infer extra unplayed turns."""
    available = set(map(int, available_turns))
    missing = [sum(t not in available for t in range(1, r["battles"] + 1)) for r in rows]
    battles = sum(r["battles"] for r in rows)
    return {
        "battles": battles,
        "fallback_battles": sum(missing),
        "fallback_battle_rate": sum(missing) / max(1, battles),
        "episodes_with_fallback": sum(x > 0 for x in missing),
    }


def audit_pilot():
    verify(PILOT_ROOT)
    summary = read(PILOT_ROOT / "pilot_summary.json")
    if not summary["paired_initial_evaluation_exact"]:
        raise ValueError("Pilot initial evaluations differ")
    curves, fallbacks, reviewed_truncations = {}, {}, []
    profile = read(PILOT_ROOT / "pool_profiles.json")["validation/control1907"]
    for arm in ARMS:
        if not summary["arms"][arm]["reload_exact"]:
            raise ValueError("Pilot reload failed")
        history = read(PILOT_ROOT / arm / "validation_history.json")
        if history[-1]["timesteps"] != 1048576:
            raise ValueError("Pilot budget incomplete")
        curves[arm] = [
            {
                "step": r["timesteps"],
                "evaluation_file": r["evaluation_file"],
                "learned_success": r["families"]["control1907"]["success_rate"],
                "scripted_success": fmean(r["families"][f]["success_rate"] for f in FAMILIES),
                "forced": r["forced_end_turns"],
                "truncation": r["truncation_rate"],
                "selected": r["selected"],
            }
            for r in history
        ]
        if any(r["forced"] for r in curves[arm]):
            raise ValueError("Pilot forcing needs review before confirmation")
        selected = summary["arms"][arm]["selected_validation"]
        if selected["truncation_rate"] or selected["forced_end_turns"]:
            raise ValueError("Selected pilot model fails stability screen")
        for checkpoint in history:
            if not checkpoint["truncation_rate"]:
                continue
            data = read(PILOT_ROOT / arm / checkpoint["evaluation_file"])
            for family, results in data["families"].items():
                for row in results["episode_results"]:
                    if not row["truncated"]:
                        continue
                    if row["reason"] != "turn_limit" or row["battles"] != 30:
                        raise ValueError("Unreviewed pilot truncation type")
                    reviewed_truncations.append(
                        {
                            "arm": arm,
                            "checkpoint": checkpoint["evaluation_file"],
                            "family": family,
                            "episode": row,
                        }
                    )
        rows = read(PILOT_ROOT / f"{arm}_reload.json")["families"]["control1907"]["episode_results"]
        fallbacks[arm] = fallback_counts(rows, profile["by_turn"])
    return {
        "curves": curves,
        "selected_validation_fallbacks": fallbacks,
        "reviewed_intermediate_truncations": reviewed_truncations,
        "decision": "Proceed with unchanged algorithm/reward; larger reachable pools. "
        "Mixed leads at all four pre-update checkpoints, but final update drops "
        "learned success from 29/60 to 20/60. Do not conceal final-policy regression. "
        "Validation checkpoint selection remains prespecified and identical across arms. "
        "The intermediate 524288-step mixed checkpoint has one 30-round turn_limit "
        "episode (17 natural draws), not a shop-action loop. Selected/final pilot "
        "validation has zero truncations. This is disclosed, not removed from results; "
        "formal candidate test acceptance still requires zero truncations.",
        "limits": "Single pilot pair, only 60 learned validation episodes, shared "
        "generator ancestry. No significance or broad generalization claim.",
    }


def copy_checked(source, destination):
    digest = file_digest(str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if file_digest(str(destination)) != digest:
        raise ValueError("Copied artifact changed")
    return digest


def collect_pool(output, name, model_path, against, episodes, seed):
    from sb3_contrib import MaskablePPO

    from .learned_pool import collect

    folder = output / "data" / name
    folder.mkdir(parents=True, exist_ok=False)
    digest = file_digest(str(model_path))
    model = MaskablePPO.load(model_path, device="cpu")
    metadata = []
    pool, rows = collect(
        model, episodes=episodes, seed=seed, opponent_provider=against, snapshot_metadata=metadata
    )
    path = folder / "pool.json"
    pool.save(path)
    if file_digest(str(model_path)) != digest:
        raise ValueError("Generator changed during collection")
    save_new(
        folder / "generation.json",
        {
            "model": str(model_path),
            "model_sha256": digest,
            "policy_sha256": policy_digest(model.policy),
            "environment_contract": model.sap_environment_contract,
            "episodes": episodes,
            "seed": seed,
            "snapshot_metadata": metadata,
            "episode_results": rows,
            "candidate_evaluation_performed": False,
        },
    )
    save_new(folder / "profile.json", profile_pool(path, metadata))
    return str(path)


def prepare(output):
    import torch

    torch.set_num_threads(1)
    audit = audit_pilot()
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "pilot_audit.json", audit)
    old = read(PILOT_ROOT / "design.json")
    initializers, generators = {}, {}
    for seed, item in old["initializers"].items():
        destination = output / "models" / f"initial-{seed}.zip"
        copy_checked(item["path"], destination)
        initializers[seed] = {**item, "path": str(destination)}
    for role, items in old["generators"].items():
        generators[role] = {}
        for name, item in items.items():
            destination = output / "models" / f"{role}-{name}.zip"
            copy_checked(item["path"], destination)
            generators[role][name] = {**item, "path": str(destination)}
    validate_roles(generators, initializers)
    design = {
        "initializers": initializers,
        "generators": generators,
        "seeds": SEEDS,
        "continuation_seeds": [11101, 11203, 11311],
        "decisions_per_candidate": 8388608,
        "train_pool_episodes_per_generator": 1000,
        "validation_pool_episodes": 500,
        "test_pool_episodes_per_generator": 500,
        "test_episodes_per_family": 1000,
        "extra_test_generators": {
            "fresh_stats": {"seed": 12101, "families": [FAMILIES[0]], "timesteps": 4194304},
            "fresh_summon_tempo": {
                "seed": 12203,
                "families": list(FAMILIES[1:]),
                "timesteps": 4194304,
            },
        },
        "test_generation_seed": 12300000,
        "test_episode_seed": 13000000,
        "training_seed_region": 10100000,
        "validation_generation_seed": 10200000,
        "validation_episode_seed": 11000000,
        "stage_order": "Prepare pools; train two novel held-out generators from scratch; "
        "seal all test pools; train six paired candidates; freeze all choices; "
        "reload every selected model; only then open tests and summarize.",
        "scope": "Same rules, PPO/network and action cost 0.005. Candidate arms differ "
        "only in opponent mixture. No test score used to choose generator, checkpoint, "
        "seed, reward or run budget. Final held-out generator checkpoints used, no tuning.",
        "limit": "Two CPU workers, bounded stages, no retries or extensions, no cluster jobs.",
        "success_criteria": "Exploratory three-seed confirmation: learned-family macro "
        "ten-win improvement >=5 percentage points averaged over pairs, positive in "
        "at least two pairs; scripted macro >=90% each candidate and <=2pp regression "
        "against its paired control; zero truncations and <=1% forced episodes in "
        "every candidate test family. Report per-family/worst-family and all failures. "
        "These are newly prespecified practical thresholds, not a statistical theorem.",
    }
    save_new(output / "design.json", design)
    old_paths = read(PILOT_ROOT / "protocol.json")["paths"]
    paths = {"train_scripted": {}, "train_learned": {}, "validation": {}}
    for split in ("train_scripted", "validation"):
        for family in FAMILIES:
            destination = output / "data" / split / f"{family}.json"
            copy_checked(old_paths[split][family], destination)
            paths[split][family] = str(destination)
    against = OpponentMixture(
        [(1, SnapshotLeague.load(p)) for p in paths["train_scripted"].values()]
    )
    for name, item in generators["train"].items():
        print(f"Preparing training pool {name}: 1000 fresh episodes", flush=True)
        paths["train_learned"][name] = collect_pool(
            output, f"train-{name}", item["path"], against, 1000, 10100000
        )
    val = generators["validation"]["control1907"]
    paths["validation"]["control1907"] = collect_pool(
        output, "validation-control1907", val["path"], against, 500, 10200000
    )
    base = dict(read(PILOT_ROOT / "protocol.json")["arms"]["scripted"])
    base.update(
        timesteps=8388608,
        validation_leagues=paths["validation"],
        validation_episodes=200,
        validation_seed=11000000,
        evaluation_interval=1048576,
    )
    arms, generator_configs = {}, {}
    for seed, new_seed in zip(SEEDS, design["continuation_seeds"]):
        item = initializers[str(seed)]
        configs = arm_configs(
            {**base, "seed": new_seed},
            list(paths["train_scripted"].values()),
            list(paths["train_learned"].values()),
            item["path"],
            item["policy_sha256"],
        )
        arms.update({f"{arm}-seed{seed}": cfg for arm, cfg in configs.items()})
    for name, item in design["extra_test_generators"].items():
        cfg = dict(base)
        cfg.update(
            seed=item["seed"],
            timesteps=item["timesteps"],
            initialize_from="",
            expected_initial_policy_sha256="",
            validation_leagues={},
            validation_league="",
            opponent_leagues=[paths["train_scripted"][f] for f in item["families"]],
            output_dir=str(output / "test_generators" / name),
        )
        generator_configs[name] = cfg
    for cfg in list(arms.values()) + list(generator_configs.values()):
        TrainingConfig(**cfg).validate()
    save_new(
        output / "protocol.json",
        {
            "base_config": {},
            "arms": arms,
            "generator_configs": generator_configs,
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
            "design_sha256": file_digest(str(output / "design.json")),
            "selection": "Four validation families equally weighted; unassisted ten-win "
            "rate, then raw return, earliest tie. Initial and final checkpoints eligible. "
            "No choice from test scores. Final-policy validation also reported.",
        },
    )
    print("Prepared frozen confirmation; all held-out scores unopened", flush=True)


def checked(output):
    protocol = read(output / "protocol.json")
    verify_inputs(output, protocol)
    if file_digest(str(output / "design.json")) != protocol["design_sha256"]:
        raise ValueError("Design changed")
    for relative, digest in protocol["model_sha256"].items():
        if file_digest(str(output / relative)) != digest:
            raise ValueError("Frozen model changed")
    return protocol


def generator(output, name):
    protocol = checked(output)
    started = time.monotonic()
    with (output / f"generator-{name}.log").open("x") as log:
        with redirect_stdout(log), redirect_stderr(log):
            model = train(TrainingConfig(**protocol["generator_configs"][name]))
    checked(output)
    save_new(
        output / f"generator-{name}_complete.json",
        {
            "path": str(model),
            "sha256": file_digest(str(model)),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def seal_tests(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = checked(output)
    design = read(output / "design.json")
    against = OpponentMixture(
        [(1, SnapshotLeague.load(p)) for p in protocol["paths"]["train_scripted"].values()]
    )
    sources = {"reserved_control2027": design["generators"]["reserved_test"]["control2027"]["path"]}
    for name in protocol["generator_configs"]:
        complete = read(output / f"generator-{name}_complete.json")
        if file_digest(complete["path"]) != complete["sha256"]:
            raise ValueError("Held-out generator changed")
        sources[name] = complete["path"]
    roles = {k: v for k, v in design["generators"].items() if k != "reserved_test"}
    roles["test"] = {}
    reference = design["initializers"]["1907"]["environment_contract"]
    for name, path in sources.items():
        model = MaskablePPO.load(path, device="cpu")
        if model.sap_environment_contract != reference:
            raise ValueError("Held-out generator rules differ")
        roles["test"][name] = {
            "policy_sha256": policy_digest(model.policy),
            "path": path,
            "sha256": file_digest(path),
        }
    validate_roles(roles, design["initializers"])
    paths = {}
    for i, (name, path) in enumerate(sources.items()):
        print(f"Sealing test pool {name}; no candidate evaluated", flush=True)
        paths[name] = collect_pool(output, f"test-{name}", path, against, 500, 12300000 + i * 10000)
    cfg = TrainingConfig(**next(iter(protocol["arms"].values())))
    catalog, game = cfg.environment()
    for i, family in enumerate(FAMILIES):
        folder = output / "data" / f"test-{family}"
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "pool.json"
        build_scripted_league(family, 500, 12400000 + i * 10000, catalog=catalog, config=game).save(
            path
        )
        paths[family] = str(path)
    save_new(
        output / "sealed_tests.json",
        {
            "paths": paths,
            "generators": roles["test"],
            "sha256": {path: file_digest(path) for path in paths.values()},
            "all_test_data_sha256": {
                str(p): file_digest(str(p))
                for folder in (output / "data").glob("test-*")
                for p in folder.glob("*.json")
            },
            "episodes_per_family": 1000,
            "episode_seed": 13000000,
            "candidate_evaluation_started": False,
        },
    )


def check_tests(output):
    sealed = read(output / "sealed_tests.json")
    for path, digest in sealed["all_test_data_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError("Sealed test data changed")
    for item in sealed["generators"].values():
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Sealed generator changed")
    return sealed


def freeze_selection(output):
    protocol = checked(output)
    check_tests(output)
    models = {}
    for name, cfg in protocol["arms"].items():
        complete = read(output / f"{name}_complete.json")
        history = read(output / name / "validation_history.json")
        manifest = read(output / name / "run_manifest.json")
        if not complete["training_complete"] or history[-1]["timesteps"] != cfg["timesteps"]:
            raise ValueError("Unequal or incomplete candidate budgets")
        if manifest["initial_policy_sha256"] != cfg["expected_initial_policy_sha256"]:
            raise ValueError("Paired initial weights differ")
        path = str(output / name / "best_model.zip")
        if file_digest(path) != complete["best_model_sha256"]:
            raise ValueError("Candidate checkpoint changed")
        selected = [row for row in history if row["selected"]][-1]
        models[name] = {
            "path": path,
            "sha256": file_digest(path),
            "selected": selected,
            "final_validation": history[-1],
            "initial_policy_sha256": manifest["initial_policy_sha256"],
        }
    for seed in SEEDS:
        first, second = [output / f"{arm}-seed{seed}" for arm in ARMS]
        h1, h2 = [read(p / "validation_history.json") for p in (first, second)]
        a = read(first / h1[0]["evaluation_file"])["families"]
        b = read(second / h2[0]["evaluation_file"])["families"]
        if a != b:
            raise ValueError("Paired starting validation differs")
    delivery = max(
        (n for n in models if n.startswith("historical_mix")),
        key=lambda n: (
            models[n]["selected"]["success_without_forcing_rate"],
            models[n]["selected"]["mean_return"],
            -int(n.rsplit("seed", 1)[1]),
        ),
    )
    save_new(
        output / "selection.json",
        {"models": models, "delivery": delivery, "test_evaluation_started": False},
    )


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = checked(output)
    sealed = check_tests(output)
    selection = read(output / "selection.json")
    item = selection["models"][name]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Selected model changed")
    if stage == "test":
        if not all((output / f"reload-{n}.json").exists() for n in protocol["arms"]):
            raise ValueError("All exact reloads must finish before any test")
        paths, episodes, seed = sealed["paths"], 1000, sealed["episode_seed"]
    else:
        cfg = protocol["arms"][name]
        paths, episodes, seed = (
            cfg["validation_leagues"],
            cfg["validation_episodes"],
            cfg["validation_seed"],
        )
    model = MaskablePPO.load(item["path"], device="cpu")
    actual = evaluate_suite(model, paths, episodes=episodes, seed=seed)
    if stage == "reload":
        expected = read(output / name / item["selected"]["evaluation_file"])
        for family, result in actual["families"].items():
            if result["episode_results"] != expected["families"][family]["episode_results"]:
                raise ValueError(f"Exact reload failed: {name}/{family}")
    coverage = {}
    for family, path in paths.items():
        pool = read(path)
        coverage[family] = fallback_counts(
            actual["families"][family]["episode_results"], {s["turn"] for s in pool["snapshots"]}
        )
    actual["opponent_fallback_coverage"] = coverage
    save_new(output / f"{stage}-{name}.json", actual)


def summarize(output):
    from .experiments import paired_comparison

    checked(output)
    sealed = check_tests(output)
    learned = [f for f in sealed["paths"] if f not in FAMILIES]
    pairs, deltas, guards = {}, [], []
    for seed in SEEDS:
        a, b = [read(output / f"test-{arm}-seed{seed}.json") for arm in ARMS]
        rates = {}
        for arm, data in zip(ARMS, (a, b)):
            rates[arm] = {
                "learned_success": fmean(data["families"][f]["success_rate"] for f in learned),
                "scripted_success": fmean(data["families"][f]["success_rate"] for f in FAMILIES),
                "worst_learned_family": min(data["families"][f]["success_rate"] for f in learned),
                "fallback_coverage": data["opponent_fallback_coverage"],
            }
        delta = rates[ARMS[1]]["learned_success"] - rates[ARMS[0]]["learned_success"]
        deltas.append(delta)
        guards.append(
            rates[ARMS[1]]["scripted_success"] >= 0.9 - 1e-12
            and rates[ARMS[1]]["scripted_success"]
            >= rates[ARMS[0]]["scripted_success"] - 0.02 - 1e-12
            and all(
                f["truncation_rate"] == 0 and f["forced_episode_rate"] <= 0.01
                for f in b["families"].values()
            )
        )
        pairs[str(seed)] = {
            "rates": rates,
            "learned_difference": delta,
            "family_comparisons": {
                f: paired_comparison(b["families"][f], a["families"][f], resamples=2000)
                for f in sealed["paths"]
            },
        }
    passed = fmean(deltas) >= 0.05 - 1e-12 and sum(d > 0 for d in deltas) >= 2 and all(guards)
    save_new(
        output / "confirmation_summary.json",
        {
            "pairs": pairs,
            "mean_learned_difference": fmean(deltas),
            "positive_pairs": sum(d > 0 for d in deltas),
            "safety_guards": guards,
            "practical_screen_passed": passed,
            "interpretation": "Episode bootstrap intervals are conditional on each fixed "
            "policy pair and pool, not uncertainty across training seeds or strategies. "
            "Three paired continuations share their earlier training ecosystem; not "
            "proof of universal strength or full Turtle performance. Inspect worst-family "
            "and fallback usage before deployment; no automatic replacement/publishing.",
        },
    )


def batch(output, stage, names):
    """Only this runner's owned children can be terminated on failure/timeout."""
    for start in range(0, len(names), 2):
        children = []
        try:
            for name in names[start : start + 2]:
                print(f"Starting {stage}/{name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "sap_rl_lab.historical_confirmation",
                            stage,
                            "--output",
                            str(output),
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                    )
                )
            deadline = time.monotonic() + 7200
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None, 0) for p in children):
                    raise RuntimeError(f"{stage} worker failed")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{stage} pair exceeded two hours")
                time.sleep(1)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError(f"{stage} worker failed")
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=30)


def pipeline(output):
    started = time.monotonic()
    prepare(output)
    try:
        protocol = checked(output)
        batch(output, "generator", list(protocol["generator_configs"]))
        seal_tests(output)
        batch(output, "arm", list(protocol["arms"]))
        freeze_selection(output)
        batch(output, "reload", list(protocol["arms"]))
        batch(output, "test", list(protocol["arms"]))
        summarize(output)
        save_new(
            output / "pipeline_complete.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "training_and_evaluation_complete": True,
                "deployment_performed": False,
            },
        )
        print("Confirmation complete; results ready for review", flush=True)
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pipeline", "generator", "arm", "reload", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.stage == "pipeline":
        pipeline(output)
    else:
        if not args.name:
            parser.error("Worker stage requires --name")
        if args.stage == "generator":
            generator(output, args.name)
        elif args.stage == "arm":
            checked(output)
            check_tests(output)
            run_arm(output, args.name)
        else:
            inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
