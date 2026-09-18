"""Bounded v6 BC + KL-PPO pilot with separate learned train/validation/test generators."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from . import imitation
from .catalog import catalog_digest, load_catalog_by_id
from .evaluation import evaluate_suite, file_digest
from .learned_pool import collect as collect_learned
from .opponents import OpponentMixture, SnapshotLeague, build_scripted_league
from .ppo_guardrails import GuardrailConfig, ValidationGuard, train_guarded, write_new
from .round3 import league_profile, source_archive
from .training import TrainingConfig, policy_digest

ROOT = Path(__file__).resolve().parents[2]
CATALOG_ID = "turtle-v0.46-tier4-v6"
FAMILIES = ("tier4_stats", "tier4_summon", "tier4_tempo")
GENERATORS = {
    "train_stats": (53101, FAMILIES[:1], "train"),
    "train_summon": (53201, FAMILIES[1:2], "train"),
    "train_tempo": (53301, FAMILIES[2:], "train"),
    "validation_balanced": (53401, FAMILIES, "validation"),
    "validation_stats_tempo": (53501, (FAMILIES[0], FAMILIES[2]), "validation"),
    "test_balanced": (53601, FAMILIES, "test"),
    "test_summon_tempo": (53701, FAMILIES[1:], "test"),
}
SEEDS = (54101, 54201, 54301)
GENERATOR_STEPS, CANDIDATE_STEPS = 1_048_576, 2_097_152
VAL_EPISODES, TEST_EPISODES = 200, 500
VAL_SEED, TEST_SEED = 59000000, 60000000
PLAN = ROOT / "docs/TIER4_PLAN.md"
RULE_MODULES = (
    "catalog",
    "domain",
    "engine",
    "events",
    "midgame_events",
    "tier4_events",
    "env",
    "shop",
    "actions",
    "baselines",
    "expanded_baselines",
    "midgame_baselines",
    "tier4_baselines",
)


def read(path):
    return json.loads(Path(path).read_text())


class LearnedFirstGuard(ValidationGuard):
    """New, predeclared v6 selection rule; inherited safety and regression checks."""

    def score(self, families):
        learned = [f for name, f in families.items() if name.startswith("validation_")]
        if len(learned) != 2:
            raise ValueError("v6 selector needs exactly two learned validation families")
        return (
            fmean(f["success"] for f in learned),
            min(f["success"] for f in learned),
            fmean(f["success"] for f in families.values()),
            -max(f["forcing"] for f in families.values()),
            fmean(f["mean_return"] for f in families.values()),
        )


def configuration(**kwargs):
    return TrainingConfig(
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


def seal(path):
    write_new(path.with_suffix(".seal.json"), {"sha256": file_digest(path)})


def checked(output):
    path = output / "protocol.json"
    if file_digest(path) != read(path.with_suffix(".seal.json"))["sha256"]:
        raise ValueError("v6 protocol changed")
    p = read(path)
    for relative, digest in p["source_files_sha256"].items():
        if file_digest(ROOT / relative) != digest:
            raise ValueError(f"source changed during frozen run: {relative}")
    for absolute, digest in p["external_sha256"].items():
        if file_digest(absolute) != digest:
            raise ValueError(f"external input changed: {absolute}")
    for relative, digest in p["data_sha256"].items():
        if file_digest(output / relative) != digest:
            raise ValueError(f"data changed: {relative}")
    return p


def prepare(output):
    import torch

    torch.set_num_threads(1)
    stress_path = ROOT / "runs/tier4-stress-v1/summary.json"
    stress = read(stress_path)
    smoke_path = ROOT / "runs/tier4-training-smoke-v1/summary.json"
    smoke = read(smoke_path)
    if smoke["ppo_decisions"] != 32 or not smoke["exact_reload"]:
        raise ValueError("BC/PPO/reload smoke gate incomplete")
    for relative, digest in smoke["source_files_sha256"].items():
        if file_digest(ROOT / relative) != digest:
            raise ValueError(f"code changed since training smoke: {relative}")
    catalog = load_catalog_by_id(CATALOG_ID)
    if stress["catalog_sha256"] != catalog_digest(catalog):
        raise ValueError("stress catalog differs")
    for module in RULE_MODULES:
        relative = f"src/sap_rl_lab/{module}.py"
        if file_digest(ROOT / relative) != stress["source_files_sha256"][relative]:
            raise ValueError(f"rules changed since stress: {module}")
    if stress["battles_twice_reproduced"] < 6000 or stress["episodes_exactly_replayed"] < 160:
        raise ValueError("stress gate incomplete")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "preparation_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    paths = {}
    for index, split in enumerate(("train", "generator_validation", "validation", "test")):
        paths[split] = {}
        for f_index, family in enumerate(FAMILIES):
            target = output / "data" / split / f"{family}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 300 if split == "train" else 200
            print(f"Building {split}/{family}: {count} episodes", flush=True)
            league = build_scripted_league(
                family, count, 55000000 + index * 1000000 + f_index * 10000, catalog=catalog
            )
            league.save(target)
            paths[split][family] = str(target)
    config = configuration(opponent_leagues=tuple(paths["train"].values()))
    demos = {}
    for split, count, seed in (("train", 600, 61000000), ("validation", 100, 62000000)):
        print(f"Collecting BC {split} demonstrations", flush=True)
        target = output / "demonstrations" / f"{split}.npz"
        demos[split] = imitation.collect(config, FAMILIES, count, seed, target)
        if any(r["truncated"] for r in demos[split]["episodes"]):
            raise ValueError("Teacher demonstration truncated")
    external = {
        str(PLAN): file_digest(PLAN),
        str(stress_path): file_digest(stress_path),
        str(smoke_path): file_digest(smoke_path),
    }
    # Protect old selected checkpoints and reporting; never reselection by old test scores.
    for seed in (17301, 204101, 204201):
        complete_path = ROOT / f"runs/kl-80-v1/kl-seed{seed}/complete.json"
        selected_path = Path(read(complete_path)["selected"]["checkpoint"]["path"])
        external[str(complete_path)] = file_digest(complete_path)
        external[str(selected_path)] = file_digest(selected_path)
    external[str(ROOT / "runs/kl-80-v1/final_review.json")] = file_digest(
        ROOT / "runs/kl-80-v1/final_review.json"
    )
    write_new(
        output / "protocol.json",
        {
            "catalog_id": CATALOG_ID,
            "catalog_sha256": catalog_digest(catalog),
            "paths": paths,
            "demonstrations": {
                k: {"path": v["path"], "sha256": v["sha256"]} for k, v in demos.items()
            },
            "generators": GENERATORS,
            "candidate_seeds": SEEDS,
            "generator_steps": GENERATOR_STEPS,
            "candidate_steps": CANDIDATE_STEPS,
            "maximum_ppo_decisions": len(GENERATORS) * GENERATOR_STEPS
            + len(SEEDS) * CANDIDATE_STEPS,
            "maximum_workers": 2,
            "maximum_pipeline_seconds": 14400,
            "validation_episodes": VAL_EPISODES,
            "test_episodes": TEST_EPISODES,
            "validation_seed": VAL_SEED,
            "test_seed": TEST_SEED,
            "selection": "two learned validation families first; "
            "all-family reliability/regression checks",
            "bc": {"epochs": 20, "batch_size": 512, "lr": 0.0003},
            "guardrails": asdict(GuardrailConfig(stop_on_regression=False)),
            "source_files_sha256": source_archive(output),
            "external_sha256": external,
            "data_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for folder in ("data", "demonstrations")
                for p in sorted((output / folder).rglob("*"))
                if p.is_file()
            },
            "no_old_checkpoint_overwrite": True,
            "no_80_percent_acceptance_claim": True,
            "no_automatic_expansion": True,
            "test_used_for_selection": False,
            "scope": "Fresh BC + PPO, not weight transfer or a from-scratch PPO ablation. "
            "Independent generator policies share architecture and teacher sources; "
            "not real arena or exhaustive ecology generalization.",
        },
    )
    seal(output / "protocol.json")
    checked(output)


def fit_bc(output, name, seed):
    p = checked(output)
    config = configuration(seed=seed, opponent_leagues=tuple(p["paths"]["train"].values()))
    model = imitation.fresh_model(config)
    try:
        train = imitation.load_data(Path(p["demonstrations"]["train"]["path"]))
        validation = imitation.load_data(Path(p["demonstrations"]["validation"]["path"]))
        result = imitation.fit(model, train, validation, output / "bc" / name, seed=seed, **p["bc"])
    finally:
        model.get_env().close()
    checked(output)
    return result


def generator(output, name):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = checked(output)
    seed, families, role = p["generators"][name]
    bc = fit_bc(output, name, seed)
    config = configuration(
        seed=seed,
        timesteps=GENERATOR_STEPS,
        output_dir=str(output / "generators" / name),
        initialize_from=bc["selected"]["path"],
        expected_initial_policy_sha256=bc["selected"]["policy_sha256"],
        opponent_leagues=tuple(p["paths"]["train"][f] for f in families),
        validation_leagues=p["paths"]["generator_validation"],
        validation_seed=63000000,
        validation_episodes=30,
        evaluation_interval=GENERATOR_STEPS,
    )
    print(f"Starting generator PPO: {name}, {GENERATOR_STEPS} decisions", flush=True)
    result = train_guarded(config, GuardrailConfig(**p["guardrails"]))
    if result["stop_reason"] != "budget_complete" or result["actual_timesteps"] != GENERATOR_STEPS:
        raise ValueError("Generator did not complete its fixed budget")
    model = MaskablePPO.load(result["last_model"], device="cpu")
    provider = OpponentMixture(
        [(1.0, SnapshotLeague.load(p["paths"]["train"][f])) for f in families]
    )
    pool, rows = collect_learned(
        model,
        episodes=300,
        seed=64000000 + list(GENERATORS).index(name) * 100000,
        opponent_provider=provider,
    )
    if any(r["truncated"] for r in rows):
        raise ValueError("Learned generator produced truncations")
    destination = output / "data" / role / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    pool.save(destination)
    write_new(
        output / f"generator-{name}.json",
        {
            "name": name,
            "role": role,
            "seed": seed,
            "model": result["last_model"],
            "policy_sha256": policy_digest(model.policy),
            "pool": str(destination),
            "pool_sha256": file_digest(destination),
            "profile": league_profile(destination),
            "episodes": rows,
            "selection": "fixed endpoint, never candidate test performance",
        },
    )
    checked(output)


def candidate_config(output, seed):
    p = checked(output)
    sealed = output / "candidate_protocol.json"
    if file_digest(sealed) != read(sealed.with_suffix(".seal.json"))["sha256"]:
        raise ValueError("Candidate protocol changed")
    candidate = read(sealed)
    for path, sha in candidate["data_sha256"].items():
        if file_digest(path) != sha:
            raise ValueError("Learned pool changed")
    return p, candidate, configuration(**candidate["configs"][str(seed)])


def candidate(output, name):
    seed = int(name)
    p, cp, config = candidate_config(output, seed)
    bc = fit_bc(output, f"candidate-{seed}", seed)
    config = replace(
        config,
        initialize_from=bc["selected"]["path"],
        expected_initial_policy_sha256=bc["selected"]["policy_sha256"],
    )
    print(f"Starting candidate PPO: {name}, {CANDIDATE_STEPS} decisions", flush=True)
    result = train_guarded(
        config, GuardrailConfig(**p["guardrails"]), guard_factory=LearnedFirstGuard
    )
    if result["stop_reason"] != "budget_complete" or result["actual_timesteps"] != CANDIDATE_STEPS:
        raise ValueError("Candidate did not complete fixed budget")
    if result["selected"] is None:
        raise ValueError("No eligible validation checkpoint; stop, do not fall back")
    candidate_config(output, seed)


def prepare_candidates(output):
    p = checked(output)
    paths = {split: dict(p["paths"][split]) for split in ("train", "validation", "test")}
    for name, (_, _, role) in GENERATORS.items():
        manifest = read(output / f"generator-{name}.json")
        if file_digest(manifest["pool"]) != manifest["pool_sha256"]:
            raise ValueError("Generated pool changed")
        paths[role][name] = manifest["pool"]
    configs = {}
    for seed in SEEDS:
        configs[str(seed)] = dict(
            seed=seed,
            timesteps=CANDIDATE_STEPS,
            output_dir=str(output / f"candidate-{seed}"),
            opponent_leagues=list(paths["train"].values()),
            validation_leagues=paths["validation"],
            validation_seed=VAL_SEED,
            validation_episodes=VAL_EPISODES,
            evaluation_interval=262144,
        )
    write_new(
        output / "candidate_protocol.json",
        {
            "paths": paths,
            "configs": configs,
            "data_sha256": {
                path: file_digest(path)
                for paths_for_role in paths.values()
                for path in paths_for_role.values()
            },
            "candidate_training_started": False,
            "test_scores_opened": False,
        },
    )
    seal(output / "candidate_protocol.json")


def selections(output):
    bindings = {}
    for seed in SEEDS:
        p, cp, config = candidate_config(output, seed)
        complete = read(output / f"candidate-{seed}/complete.json")
        # Recompute the selector against every saved raw validation record.
        guard = LearnedFirstGuard(GuardrailConfig(**p["guardrails"]))
        for entry in complete["evaluations"]:
            record = read(Path(entry["checkpoint"]["path"]).with_suffix(".json"))
            assert record["checkpoint"] == entry["checkpoint"]
            assert file_digest(entry["checkpoint"]["path"]) == entry["checkpoint"]["sha256"]
            guard.consider(record["evaluation"], record["checkpoint"])
        if guard.best["checkpoint"] != complete["selected"]["checkpoint"]:
            raise ValueError("Independent checkpoint selector disagrees")
        for kind, item in (
            ("initial", complete["evaluations"][0]["checkpoint"]),
            ("final", complete["evaluations"][-1]["checkpoint"]),
            ("selected", complete["selected"]["checkpoint"]),
        ):
            bindings[f"{seed}-{kind}"] = item
    write_new(output / "selection.json", {"models": bindings, "test_opened": False})
    seal(output / "selection.json")


def bound_model(output, name):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    path = output / "selection.json"
    if file_digest(path) != read(path.with_suffix(".seal.json"))["sha256"]:
        raise ValueError("Selection changed after sealing")
    item = read(path)["models"][name]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Selected checkpoint changed")
    model = MaskablePPO.load(item["path"], device="cpu")
    if policy_digest(model.policy) != item["policy_sha256"]:
        raise ValueError("Loaded tensor digest differs")
    return item, model


def reload_model(output, name):
    seed = int(name.split("-")[0])
    p, cp, config = candidate_config(output, seed)
    item, model = bound_model(output, name)
    expected = read(Path(item["path"]).with_suffix(".json"))["evaluation"]
    actual = evaluate_suite(model, cp["paths"]["validation"], episodes=VAL_EPISODES, seed=VAL_SEED)
    for family in expected["families"]:
        if (
            expected["families"][family]["episode_results"]
            != actual["families"][family]["episode_results"]
        ):
            raise ValueError(f"Exact reload differs: {name}/{family}")
    write_new(
        output / f"reload-{name}.json",
        {"exact_rows": 5 * VAL_EPISODES, "policy_sha256": item["policy_sha256"]},
    )


def test_model(output, name):
    for model_name in read(output / "selection.json")["models"]:
        if read(output / f"reload-{model_name}.json")["exact_rows"] != 5 * VAL_EPISODES:
            raise ValueError("All exact reloads must precede all tests")
    seed = int(name.split("-")[0])
    _, cp, _ = candidate_config(output, seed)
    item, model = bound_model(output, name)
    result = evaluate_suite(model, cp["paths"]["test"], episodes=TEST_EPISODES, seed=TEST_SEED)
    write_new(output / f"test-{name}.json", {"checkpoint": item, "evaluation": result})


def summarize(output):
    metrics = {}
    for name in read(output / "selection.json")["models"]:
        families = read(output / f"test-{name}.json")["evaluation"]["families"]
        rows = [r for f in families.values() for r in f["episode_results"]]
        learned = [
            r
            for f, data in families.items()
            if f.startswith("test_")
            for r in data["episode_results"]
        ]
        assert len(rows) == 5 * TEST_EPISODES and len(learned) == 2 * TEST_EPISODES
        metrics[name] = {
            "learned_successes": sum(r["success"] for r in learned),
            "learned_episodes": len(learned),
            "learned_success": fmean(r["success"] for r in learned),
            "forced_episode_rate": fmean(r["forced_end_turns"] > 0 for r in rows),
            "truncations": sum(r["truncated"] for r in rows),
            "no_purchase_zero_win_losses": sum(
                r["wins"] == 0 and not r["action_counts"].get("buy_pet", 0) for r in rows
            ),
            "swap_freeze_per_battle": sum(
                r["action_counts"].get("swap_adjacent", 0)
                + r["action_counts"].get("toggle_freeze", 0)
                for r in rows
            )
            / sum(r["battles"] for r in rows),
            "family_success": {
                f: fmean(r["success"] for r in d["episode_results"]) for f, d in families.items()
            },
        }
    write_new(
        output / "summary.json",
        {
            "results": metrics,
            "full_pack_complete": False,
            "test_used_for_selection": False,
            "automatic_delivery": False,
        },
    )
    checked(output)


def batch(output, action, names, deadline):
    pending, running = list(names), {}
    try:
        while pending or running:
            if time.monotonic() >= deadline:
                raise TimeoutError("Pipeline hard wall-clock limit")
            while pending and len(running) < 2:
                name = str(pending.pop(0))
                log = (output / f"{action}-{name}.log").open("x")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "sap_rl_lab.tier4_experiment",
                            "worker",
                            "--output",
                            str(output),
                            "--action",
                            action,
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log.close()
                    raise
                running[name] = (process, log)
                print(f"Started {action}/{name} pid={process.pid}", flush=True)
            for name, (process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[name]
                if code:
                    raise RuntimeError(f"Worker failed: {action}/{name} exit={code}")
                print(f"Completed {action}/{name}", flush=True)
            if running:
                time.sleep(1)
    finally:
        for process, log in running.values():
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
    deadline = started + 14400
    prepare(output)
    batch(output, "generator", GENERATORS, deadline)
    prepare_candidates(output)
    batch(output, "candidate", SEEDS, deadline)
    selections(output)
    names = list(read(output / "selection.json")["models"])
    batch(output, "reload", names, deadline)
    batch(output, "test", names, deadline)
    summarize(output)
    write_new(
        output / "pipeline_complete.json",
        {"elapsed_seconds": time.monotonic() - started, "training_and_evaluation_complete": True},
    )


def smoke(output):
    """32 diagnostic PPO decisions, separate from the experiment training budget."""
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    config = replace(configuration(seed=53001), environments=1, rollout_steps=8, batch_size=8)
    catalog, game = config.environment()
    pool_path = output / "pool.json"
    build_scripted_league(FAMILIES[0], 2, 53000000, catalog=catalog, config=game).save(pool_path)
    config = replace(config, opponent_leagues=(str(pool_path),))
    train_path, val_path = output / "train.npz", output / "validation.npz"
    imitation.collect(config, FAMILIES, 2, 53010000, train_path)
    imitation.collect(config, FAMILIES, 1, 53310000, val_path)
    model = imitation.fresh_model(config)
    try:
        parameters = sum(p.numel() for p in model.policy.parameters())
        bc = imitation.fit(
            model,
            imitation.load_data(train_path),
            imitation.load_data(val_path),
            output / "bc",
            epochs=2,
            seed=config.seed,
        )
    finally:
        model.get_env().close()
    families = {}
    for i in range(2):
        target = output / f"validation-pool-{i}.json"
        build_scripted_league(
            FAMILIES[i], 2, 53510000 + i * 10000, catalog=catalog, config=game
        ).save(target)
        families[f"validation_{i}"] = str(target)
    result = train_guarded(
        replace(
            config,
            timesteps=32,
            output_dir=str(output / "ppo"),
            initialize_from=bc["selected"]["path"],
            expected_initial_policy_sha256=bc["selected"]["policy_sha256"],
            validation_leagues=families,
            validation_episodes=2,
            validation_seed=53610000,
            evaluation_interval=16,
        ),
        GuardrailConfig(stop_on_regression=False),
        guard_factory=LearnedFirstGuard,
    )
    assert result["actual_timesteps"] == 32 and result["stop_reason"] == "budget_complete"
    final = MaskablePPO.load(result["last_model"], device="cpu")
    binding = result["evaluations"][-1]["checkpoint"]
    assert policy_digest(final.policy) == binding["policy_sha256"]
    expected = read(Path(binding["path"]).with_suffix(".json"))["evaluation"]
    actual = evaluate_suite(final, families, episodes=2, seed=53610000)
    for family in families:
        assert (
            actual["families"][family]["episode_results"]
            == expected["families"][family]["episode_results"]
        )
    pool, rows = collect_learned(
        final, episodes=1, seed=53710000, opponent_provider=SnapshotLeague.load(pool_path)
    )
    assert len(pool) == rows[0]["battles"] and not rows[0]["truncated"]
    pool.save(output / "learned.json")
    write_new(
        output / "summary.json",
        {
            "ppo_decisions": 32,
            "exact_reload": True,
            "reload_episodes": 4,
            "parameters": parameters,
            "learned_snapshots": len(pool),
            "catalog_sha256": catalog_digest(catalog),
            "source_files_sha256": source_archive(output),
        },
    )


def wall_timeout(signum, frame):
    raise TimeoutError("Pipeline four-hour wall-clock limit")


def launch(output):
    """Detach only this bounded pipeline; keep durable logs and exact owned PID."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_name(output.name + ".launch.log")
    manifest_path = output.with_name(output.name + ".launch.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "sap_rl_lab.tier4_experiment",
                "pipeline",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "pid": process.pid,
        "utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "log": str(log_path),
        "maximum_seconds": 14400,
        "maximum_workers": 2,
        "completion_not_yet_verified": True,
    }
    write_new(manifest_path, record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pipeline", "worker", "smoke", "launch"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action", choices=("generator", "candidate", "reload", "test"))
    parser.add_argument("--name")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.command == "launch":
            launch(output)
        elif args.command == "pipeline":
            previous = signal.signal(signal.SIGALRM, wall_timeout)
            signal.alarm(14400)
            try:
                pipeline(output)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
        elif args.command == "smoke":
            smoke(output)
        else:
            {
                "generator": generator,
                "candidate": candidate,
                "reload": reload_model,
                "test": test_model,
            }[args.action](output, args.name)
            write_new(output / f"worker-{args.action}-{args.name}-complete.json", {"exit_code": 0})
    except Exception:
        if output.exists():
            path = output / f"failure-{args.action or 'pipeline'}-{args.name or 'root'}.json"
            if not path.exists():
                write_new(path, {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
