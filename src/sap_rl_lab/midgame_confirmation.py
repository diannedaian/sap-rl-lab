"""Finite v5 confirmation: six independent generators, three paired learners.

Write-once outputs, at most two owned CPU workers, no automatic retries/tuning.
All selected models are frozen and exactly reloaded before any held-out test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .expanded import run_arm, verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .generalization import validate_roles
from .historical_confirmation import collect_pool, fallback_counts
from .midgame import CATALOG_ID, FAMILIES, ROOT
from .opponents import OpponentMixture, SnapshotLeague, build_scripted_league
from .round3 import league_profile, source_archive
from .training import TrainingConfig, midgame_checkpoint_score, policy_digest, train

SEEDS = (17101, 17201, 17301)
ARMS = ("scripted", "historical_mix")
GENERATOR_STEPS = 4_194_304
CANDIDATE_STEPS = 8_388_608
GENERATORS = {
    "train_stats": (16101, (FAMILIES[0],), "train", 300),
    "train_summon": (16201, (FAMILIES[1],), "train", 300),
    "train_tempo": (16301, (FAMILIES[2],), "train", 300),
    "validation_balanced": (16401, FAMILIES, "validation", 200),
    "test_balanced": (16501, FAMILIES, "test", 400),
    "test_summon_tempo": (16601, FAMILIES[1:], "test", 400),
}


def new_config(**kwargs):
    config = TrainingConfig(
        catalog_id=CATALOG_ID,
        allow_development=True,
        shop_action_limit_mode="force_battle",
        action_cost=0.005,
        environments=8,
        rollout_steps=256,
        batch_size=256,
        device="cpu",
        torch_threads=1,
        **kwargs,
    )
    config.validate()
    return asdict(config)


def checked(output):
    design = read(output / "design.json")
    verify_inputs(output, design)
    return design


def prepare(output):
    pilot = ROOT / "runs/midgame-pilot-v1"
    verify_inputs(pilot, read(pilot / "protocol.json"))
    summary = read(pilot / "pilot_summary.json")
    if not summary["budget_complete"] or summary["reload_rows_exact"] != 180:
        raise ValueError("Pilot budget/reload gate incomplete")
    if summary["selected_validation"]["truncation_rate"] != 0:
        raise ValueError("Review pilot truncation before formal training")
    output.mkdir(parents=True, exist_ok=False)
    catalog, game = TrainingConfig(**new_config()).environment()
    paths, profiles = {}, {}
    for split, seed, episodes in (
        ("train", 16_000_000, 300),
        ("validation", 16_100_000, 200),
        ("test", 16_200_000, 400),
    ):
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split] = {}
        for index, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            build_scripted_league(
                family, episodes, seed + index * 10_000, catalog=catalog, config=game
            ).save(path)
            paths[split][family] = str(path)
            profiles[f"{split}/{family}"] = league_profile(path)
    configs = {
        name: new_config(
            seed=seed,
            timesteps=GENERATOR_STEPS,
            output_dir=str(output / f"generator-{name}"),
            opponent_leagues=tuple(paths["train"][family] for family in families),
        )
        for name, (seed, families, _, _) in GENERATORS.items()
    }
    save_new(output / "scripted_profiles.json", profiles)
    save_new(
        output / "design.json",
        {
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(p.relative_to(output)): file_digest(str(p))
                for p in sorted((output / "data").rglob("*.json"))
            },
            "paths": paths,
            "generator_configs": configs,
            "generators": GENERATORS,
            "candidate_seeds": SEEDS,
            "candidate_steps": CANDIDATE_STEPS,
            "validation_episodes": 100,
            "validation_seed": 17_000_000,
            "evaluation_interval": 1_048_576,
            "test_episodes": 1000,
            "test_seed": 18_000_000,
            "pilot_sha256": file_digest(str(pilot / "pilot_summary.json")),
            "pilot_review": {
                "initial_mean_wins": summary["curves"][0]["mean_wins"],
                "final_mean_wins": summary["selected_validation"]["mean_wins"],
                "final_success": summary["selected_validation"]["success_rate"],
                "final_forced_episode_rate": summary["selected_validation"]["forced_episode_rate"],
                "interpretation": "Learns beyond initialization, still below scripted baselines. "
                "No strong-model claim. No selected-model truncation or reload mismatch. "
                "Proceed with fixed larger budgets, not another hyperparameter search.",
            },
            "selection": "win-first_worst-family_forcing_return-v1; exact tie keeps earlier; "
            "delivery tie chooses lower seed. No arbitrary 1% eligibility gate.",
            "limits": "6 generators x 4194304; 6 fresh candidates x 8388608 decisions. "
            "2 CPU workers maximum. No retries, extensions or test-driven reselection.",
            "scope": "Normal shops through Tier 3, true Tier 4 level-up dependencies, "
            "40 pets; not normal Tier 4 shops or full Turtle/client parity.",
        },
    )
    print("Design frozen; fresh independent generators next", flush=True)


def generator(output, name):
    design = checked(output)
    started = time.monotonic()
    with (output / f"generator-{name}.log").open("x") as log:
        with redirect_stdout(log), redirect_stderr(log):
            path = train(TrainingConfig(**design["generator_configs"][name]))
    checked(output)
    save_new(
        output / f"generator-{name}_complete.json",
        {
            "path": str(path),
            "sha256": file_digest(str(path)),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def seal(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    design = checked(output)
    paths = {split: dict(items) for split, items in design["paths"].items()}
    roles = {split: {} for split in paths}
    against = OpponentMixture([(1, SnapshotLeague.load(p)) for p in paths["train"].values()])
    contract = None
    for index, (name, (_, _, role, episodes)) in enumerate(design["generators"].items()):
        item = read(output / f"generator-{name}_complete.json")
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Generator checkpoint changed")
        model = MaskablePPO.load(item["path"], device="cpu")
        if model.num_timesteps != GENERATOR_STEPS:
            raise ValueError("Generator budget incomplete")
        if contract is None:
            contract = model.sap_environment_contract
        if model.sap_environment_contract != contract:
            raise ValueError("Generator contracts differ")
        roles[role][name] = {**item, "policy_sha256": policy_digest(model.policy)}
        paths[role][name] = collect_pool(
            output, name, item["path"], against, episodes, 16_400_000 + index * 10_000
        )
    validate_roles(roles, {})
    if contract["catalog_id"] != CATALOG_ID:
        raise ValueError("Unexpected generator catalog")
    base = new_config(
        timesteps=design["candidate_steps"],
        validation_leagues=paths["validation"],
        validation_episodes=design["validation_episodes"],
        validation_seed=design["validation_seed"],
        evaluation_interval=design["evaluation_interval"],
    )
    arms = {
        f"{arm}-seed{seed}": {
            "seed": seed,
            "opponent_leagues": list(
                design["paths"]["train"].values() if arm == "scripted" else paths["train"].values()
            ),
        }
        for seed in design["candidate_seeds"]
        for arm in ARMS
    }
    save_new(
        output / "protocol.json",
        {
            "base_config": base,
            "arms": arms,
            "paths": paths,
            "roles": roles,
            "environment_contract": contract,
            "design_sha256": file_digest(str(output / "design.json")),
            "source_files_sha256": design["source_files_sha256"],
            "data_sha256": {
                str(p.relative_to(output)): file_digest(str(p))
                for p in sorted((output / "data").rglob("*.json"))
            },
            "test_evaluation_started": False,
        },
    )
    print("Train/validation/test pools sealed; candidate training next", flush=True)


def check_protocol(output):
    checked(output)
    protocol = read(output / "protocol.json")
    verify_inputs(output, protocol)
    if protocol["design_sha256"] != file_digest(str(output / "design.json")):
        raise ValueError("Design changed after pool freeze")
    for entries in protocol["roles"].values():
        for item in entries.values():
            if file_digest(item["path"]) != item["sha256"]:
                raise ValueError("Frozen generator changed")
    return protocol


def freeze(output):
    from sb3_contrib import MaskablePPO

    protocol = check_protocol(output)
    models = {}
    for name in protocol["arms"]:
        done = read(output / f"{name}_complete.json")
        history = read(output / name / "validation_history.json")
        manifest = read(output / name / "run_manifest.json")
        final = MaskablePPO.load(done["final_model"], device="cpu")
        if final.num_timesteps != CANDIDATE_STEPS or not done["training_complete"]:
            raise ValueError("Incomplete candidate budget")
        if final.sap_environment_contract != protocol["environment_contract"]:
            raise ValueError("Candidate/generator contract mismatch")
        for field in ("best_model", "final_model"):
            if file_digest(done[field]) != done[f"{field}_sha256"]:
                raise ValueError("Candidate changed after completion")
        selected = [row for row in history if row["selected"]][-1]
        if tuple(selected["selection_score"]) != max(map(midgame_checkpoint_score, history)):
            raise ValueError("Selected checkpoint does not satisfy frozen rule")
        models[name] = {
            "path": done["best_model"],
            "sha256": done["best_model_sha256"],
            "selected": selected,
            "final_validation": history[-1],
            "initial_policy_sha256": manifest["initial_policy_sha256"],
        }
    initializers = {}
    for seed in SEEDS:
        a, b = [f"{arm}-seed{seed}" for arm in ARMS]
        if models[a]["initial_policy_sha256"] != models[b]["initial_policy_sha256"]:
            raise ValueError("Paired initial weights differ")
        initializers[str(seed)] = {"policy_sha256": models[a]["initial_policy_sha256"]}
        first = []
        for name in (a, b):
            history = read(output / name / "validation_history.json")
            first.append(read(output / name / history[0]["evaluation_file"]))
        compare_reload(*first)
    if len({i["policy_sha256"] for i in initializers.values()}) != len(SEEDS):
        raise ValueError("Independent candidate seeds share weights")
    validate_roles(protocol["roles"], initializers)
    delivery = max(
        (name for name in models if name.startswith("historical_mix")),
        key=lambda name: (
            *midgame_checkpoint_score(models[name]["selected"]),
            -int(name.rsplit("seed", 1)[1]),
        ),
    )
    save_new(
        output / "selection.json",
        {
            "models": models,
            "delivery": delivery,
            "protocol_sha256": file_digest(str(output / "protocol.json")),
            "test_evaluation_started": False,
        },
    )


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = check_protocol(output)
    design = read(output / "design.json")
    selection = read(output / "selection.json")
    if selection["protocol_sha256"] != file_digest(str(output / "protocol.json")):
        raise ValueError("Protocol changed after model selection")
    item = selection["models"][name]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Selected model changed")
    if stage == "test":
        for candidate, info in selection["models"].items():
            reload = read(output / f"reload-{candidate}.json")
            if (
                not reload["all_validation_rows_exact"]
                or reload["model_sha256"] != info["sha256"]
                or reload["selection_sha256"] != file_digest(str(output / "selection.json"))
            ):
                raise ValueError("All frozen-model reloads must pass before any test")
        paths = protocol["paths"]["test"]
        episodes, seed = design["test_episodes"], design["test_seed"]
    else:
        paths = protocol["paths"]["validation"]
        episodes, seed = design["validation_episodes"], design["validation_seed"]
    model = MaskablePPO.load(item["path"], device="cpu")
    actual = evaluate_suite(model, paths, episodes=episodes, seed=seed)
    if stage == "reload":
        compare_reload(actual, read(output / name / item["selected"]["evaluation_file"]))
        actual.update(
            all_validation_rows_exact=True,
            model_sha256=item["sha256"],
            selection_sha256=file_digest(str(output / "selection.json")),
        )
    actual["opponent_fallback_coverage"] = {
        family: fallback_counts(
            actual["families"][family]["episode_results"],
            {s["turn"] for s in read(path)["snapshots"]},
        )
        for family, path in paths.items()
    }
    check_protocol(output)
    save_new(output / f"{stage}-{name}.json", actual)


def summarize(output):
    from .experiments import paired_comparison

    protocol = check_protocol(output)
    pairs = {}
    for seed in SEEDS:
        results = {arm: read(output / f"test-{arm}-seed{seed}.json") for arm in ARMS}
        learned = [f for f in protocol["paths"]["test"] if f not in FAMILIES]
        rates = {
            arm: {
                "learned_success": fmean(data["families"][f]["success_rate"] for f in learned),
                "scripted_success": fmean(data["families"][f]["success_rate"] for f in FAMILIES),
                "worst_learned_success": min(data["families"][f]["success_rate"] for f in learned),
                "evaluation": compact_evaluation(data),
            }
            for arm, data in results.items()
        }
        pairs[str(seed)] = {
            "rates": rates,
            "learned_difference": rates[ARMS[1]]["learned_success"]
            - rates[ARMS[0]]["learned_success"],
            "family_comparisons": {
                family: paired_comparison(
                    results[ARMS[1]]["families"][family],
                    results[ARMS[0]]["families"][family],
                    resamples=2000,
                )
                for family in protocol["paths"]["test"]
            },
        }
    save_new(
        output / "confirmation_summary.json",
        {
            "pairs": pairs,
            "mean_learned_difference": fmean(p["learned_difference"] for p in pairs.values()),
            "positive_pairs": sum(p["learned_difference"] > 0 for p in pairs.values()),
            "delivery": read(output / "selection.json")["delivery"],
            "interpretation": "Conditional on these frozen pools and three fresh seed pairs. "
            "Two held-out learned generators share algorithm/engine/script ecosystem, not "
            "universal strategy coverage. Historical arrays are opponent lineups, not learner "
            "offline trajectories. Late-round survival and turn-only matching bias remain. "
            "No retroactive 1% pass/fail gate and no test-based model replacement.",
        },
    )


def batch(output, stage, names):
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
                            "sap_rl_lab.midgame_confirmation",
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
    prepare(output)
    try:
        batch(output, "generator", list(checked(output)["generator_configs"]))
        seal(output)
        names = list(check_protocol(output)["arms"])
        batch(output, "arm", names)
        freeze(output)
        batch(output, "reload", names)
        batch(output, "test", names)
        summarize(output)
        save_new(
            output / "pipeline_complete.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "training_and_evaluation_complete": True,
                "deployment_performed": False,
                "human_readable_report_pending": True,
            },
        )
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pipeline", "generator", "arm", "reload", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "pipeline":
        pipeline(output)
    elif args.stage == "generator":
        generator(output, args.name)
    elif args.stage == "arm":
        check_protocol(output)
        run_arm(output, args.name)
    else:
        inference(output, args.name, args.stage)


if __name__ == "__main__":
    main()
