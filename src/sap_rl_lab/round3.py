"""Bounded from-scratch opponent-diversity experiment, with sealed test selection."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from .evaluation import compact_evaluation, evaluate_policy, file_digest
from .experiments import paired_comparison, save_json
from .opponents import SnapshotLeague, build_scripted_league
from .training import TrainingConfig, train

FAMILIES = ("greedy", "stats", "summon")


def source_archive(output: Path):
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py"))
    files += sorted((root / "src").rglob("*.json"))
    files += [root / "pyproject.toml"]
    hashes = {str(p.relative_to(root)): file_digest(str(p)) for p in files}
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(root)))
    return hashes


def league_profile(path: Path):
    data = json.loads(path.read_text())
    pets = [p for team in data["snapshots"] for p in team["pets"]]
    return {
        "snapshots": len(data["snapshots"]),
        "by_turn": dict(Counter(team["turn"] for team in data["snapshots"])),
        "pet_counts": dict(Counter(p["spec_id"] for p in pets)),
        "honey_count": sum(p["perk"] == "honey" for p in pets),
        "temporary_buff_count": sum(
            bool(p["temporary_attack"] or p["temporary_health"]) for p in pets
        ),
        "mean_team_size": len(pets) / len(data["snapshots"]),
    }


def build_model_league(model, *, episodes: int, seed: int, opponent_league: str):
    from .actions import ActionKind
    from .env import SapAutoBattlerEnv

    env = SapAutoBattlerEnv(opponent_provider=SnapshotLeague.load(opponent_league))
    league = SnapshotLeague(catalog_id=env.engine.catalog.catalog_id)
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            while True:
                action_id, _ = model.predict(
                    observation, action_masks=env.action_masks(), deterministic=True
                )
                if env.engine.codec.decode(int(action_id)).kind is ActionKind.END_TURN:
                    league.add(
                        env.engine.state.turn,
                        env.engine.state.team,
                        f"round2-frozen-seed-{seed + episode}",
                    )
                observation, _, terminated, truncated, _ = env.step(int(action_id))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return league


def selected_models(output: Path, protocol):
    """Freeze all checkpoint choices/hashes before opening any test results."""
    selection = {}
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            name = f"{arm}-seed{seed}"
            root = output / name
            history = json.loads((root / "validation_history.json").read_text())
            winner = [row for row in history if row["selected"]][-1]
            path = root / "best_model.zip"
            selection[name] = {
                "path": str(path),
                "sha256": file_digest(str(path)),
                "timesteps": winner["timesteps"],
                "validation_success": winner["success_rate"],
                "validation_return": winner["mean_return"],
            }
    return selection


def run(args):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    champion = Path(args.champion).resolve()
    if not champion.is_file():
        raise FileNotFoundError(champion)
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    profiles = {}
    offset = args.seed_offset
    split_starts = {"train": 100000 + offset, "validation": 200000 + offset}
    if not args.pilot:
        split_starts["test"] = 300000 + offset
    for split, start in split_starts.items():
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split] = {}
        for index, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            build_scripted_league(family, args.pool_episodes, start + index * 10000).save(path)
            paths[split][family] = str(path)
            profiles[f"{split}/{family}"] = league_profile(path)
    # The challenge is never supplied to training or validation.
    if not args.pilot:
        model = MaskablePPO.load(str(champion), device="cpu")
        path = output / "data/test/round2_challenge.json"
        build_model_league(
            model,
            episodes=args.pool_episodes,
            seed=360000 + offset,
            opponent_league=paths["train"]["greedy"],
        ).save(path)
        paths["challenge"] = {"round2_challenge": str(path)}
        profiles["test/round2_challenge"] = league_profile(path)
        del model
    save_json(output / "pool_profiles.json", profiles)
    base_config = TrainingConfig(
        timesteps=args.timesteps,
        environments=8,
        rollout_steps=512,
        batch_size=256,
        output_dir="unused",
        device="cpu",
        vector_backend="dummy",
        torch_threads=1,
        validation_leagues=paths["validation"],
        validation_episodes=args.validation_episodes,
        validation_seed=400000 + offset,
        evaluation_interval=args.evaluation_interval,
    )
    protocol = {
        "schema_version": 1,
        "question": "Does frozen opponent diversity improve fresh-start generalization?",
        "base_config": asdict(base_config),
        "arms": {"single": [paths["train"]["greedy"]], "mixed": list(paths["train"].values())},
        "seeds": args.seeds,
        "paths": paths,
        "pilot": args.pilot,
        "pool_episode_seed_starts": split_starts,
        "family_seed_stride": 10000,
        "pool_episodes_per_family": args.pool_episodes,
        "challenge_generation_seed": 360000 + offset,
        "validation_episode_seed": 400000 + offset,
        "test_episode_seed": 500000 + offset,
        "challenge_episode_seed": 600000 + offset,
        "seed_offset": offset,
        "test_episodes_per_family": args.test_episodes,
        "champion_path": str(champion),
        "champion_sha256": file_digest(str(champion)),
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "source_files_sha256": source_archive(output),
        "selection": "maximum equal-family validation success, then return; earliest exact tie",
        "snapshot_fix": "v2 preserves temporary shop buffs; v1 loads with zero defaults",
        "frozen_rules": "Original unshaped rewards, legal mask, gamma=1, and truncation rules",
        "limits": "Three independent initializations, paired between arms. Same test pools. "
        "Episode intervals are conditional on fixed policies, not seed uncertainty. "
        "Frozen challenge is excluded from checkpoint and arm selection.",
    }
    if len(set(protocol["data_sha256"].values())) != len(protocol["data_sha256"]):
        raise ValueError("duplicated league files in experiment")
    save_json(output / "protocol.json", protocol)
    for seed in args.seeds:
        initial_hash = ""
        for arm, pools in protocol["arms"].items():
            name = f"{arm}-seed{seed}"
            config = TrainingConfig(
                **{
                    **asdict(base_config),
                    "seed": seed,
                    "output_dir": str(output / name),
                    "opponent_leagues": tuple(pools),
                    "expected_initial_policy_sha256": initial_hash,
                }
            )
            print(f"START {name}", flush=True)
            started = time.monotonic()
            with (output / f"{name}-training.log").open("x") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    train(config)
            manifest = json.loads((output / name / "run_manifest.json").read_text())
            initial_hash = manifest["initial_policy_sha256"]
            save_json(output / name / "runtime.json", {"seconds": time.monotonic() - started})
            print(f"FINISHED {name}", flush=True)
    selection = selected_models(output, protocol)
    save_json(output / "selection.json", selection)
    if args.pilot:
        print("PILOT COMPLETE: no final test pools generated or evaluated", flush=True)
        return
    evaluate_completed(output)


def evaluate_completed(output: Path):
    """Recover interrupted final evaluation without retraining or reselection."""
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["pilot"]:
        raise ValueError("cannot test a pilot")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    for relative, digest in protocol["data_sha256"].items():
        if file_digest(str(output / relative)) != digest:
            raise ValueError("experiment dataset changed")
    source_root = Path(__file__).resolve().parents[2]
    for relative, digest in protocol["source_files_sha256"].items():
        if file_digest(str(source_root / relative)) != digest:
            raise ValueError("experiment source changed; restore the archived source to evaluate")
    selection = json.loads((output / "selection.json").read_text())
    if selection != selected_models(output, protocol):
        raise ValueError("checkpoint selection changed after freezing")
    champion = protocol["champion_path"]
    if file_digest(champion) != protocol["champion_sha256"]:
        raise ValueError("reference checkpoint changed")
    candidates = {name: record["path"] for name, record in selection.items()}
    candidates.update(round2=champion, greedy="greedy", random="random")
    results = {}
    for name, path in candidates.items():
        policy = path if path in {"greedy", "random"} else MaskablePPO.load(path, device="cpu")
        result = {"families": {}}
        for family, league in {
            **protocol["paths"]["test"],
            **protocol["paths"]["challenge"],
        }.items():
            is_challenge = family == "round2_challenge"
            report = output / f"test-{name}-{family}.json"
            expected_hash = None if isinstance(policy, str) else file_digest(path)
            expected_seed = protocol[
                "challenge_episode_seed" if is_challenge else "test_episode_seed"
            ]
            if report.exists():
                evaluation = json.loads(report.read_text())
                if (
                    evaluation.get("model_sha256") != expected_hash
                    or evaluation["league_sha256"] != file_digest(league)
                    or evaluation["seed_start"] != expected_seed
                    or evaluation["episodes"] != protocol["test_episodes_per_family"]
                    or evaluation.get("policy_name") != name
                ):
                    raise ValueError("existing test report does not match frozen protocol")
            else:
                evaluation = evaluate_policy(
                    policy,
                    episodes=protocol["test_episodes_per_family"],
                    seed=expected_seed,
                    opponent_league=league,
                )
                evaluation.update(model_sha256=expected_hash, policy_name=name)
                save_json(report, evaluation)
            result["families"][family] = evaluation
            print(
                f"TEST {name}/{family}: success={evaluation['success_rate']:.3f} "
                f"cutoffs={evaluation['truncation_rate']:.3f}",
                flush=True,
            )
        for metric in ("success_rate", "mean_wins", "mean_return", "truncation_rate"):
            result[metric] = fmean(result["families"][f][metric] for f in FAMILIES)
        results[name] = result
        del policy
    comparisons = {
        str(seed): {
            family: paired_comparison(
                results[f"mixed-seed{seed}"]["families"][family],
                results[f"single-seed{seed}"]["families"][family],
            )
            for family in (*FAMILIES, "round2_challenge")
        }
        for seed in protocol["seeds"]
    }
    summary = {
        "results": {name: compact_evaluation(result) for name, result in results.items()},
        "paired_comparisons": comparisons,
        "mean_success_by_arm": {
            arm: fmean(results[f"{arm}-seed{s}"]["success_rate"] for s in protocol["seeds"])
            for arm in protocol["arms"]
        },
        "limits": protocol["limits"],
    }
    save_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps(summary["mean_success_by_arm"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--champion")
    parser.add_argument("--timesteps", type=int, default=1500000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307])
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Use a disjoint namespace for engineering smoke tests",
    )
    parser.add_argument("--pool-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=150000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_completed(Path(args.output).resolve())
    else:
        if not args.champion:
            parser.error("--champion is required")
        if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
            parser.error("seeds must be distinct nonnegative integers")
        if not 0 < args.pool_episodes <= 10000:
            parser.error("pool episodes must be 1..10000 (disjoint seed blocks)")
        if args.seed_offset < 0 or max(args.validation_episodes, args.test_episodes) > 100000:
            parser.error("seed offset must be nonnegative and evaluation blocks at most 100000")
        if (
            min(
                args.timesteps,
                args.validation_episodes,
                args.evaluation_interval,
                args.test_episodes,
            )
            < 1
        ):
            parser.error("budgets must be positive")
        run(args)


if __name__ == "__main__":
    main()
