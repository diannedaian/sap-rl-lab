"""Finite, versioned expanded-curriculum pilots; never a completion certificate."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

from .catalog import catalog_digest, load_catalog_by_id
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .experiments import save_json
from .opponents import build_scripted_league
from .round3 import league_profile, source_archive
from .training import TrainingConfig, train

CATALOG_ID = "turtle-v0.46-tier12-curriculum-v4"
FAMILIES = ("expanded_stats", "expanded_summon", "expanded_tempo")
ARMS = {
    "control": {"swap_cost": 0.0, "action_cost": 0.0},
    "swap_cost": {"swap_cost": 0.005, "action_cost": 0.0},
    "action_cost": {"swap_cost": 0.0, "action_cost": 0.005},
}


def prepare(output: Path, timesteps: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog_by_id(CATALOG_ID)
    paths, profiles = {}, {}
    for split, start in (("train", 2100000), ("validation", 2200000)):
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split], profiles[split] = {}, {}
        for index, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            build_scripted_league(family, 60, start + index * 10000, catalog=catalog).save(path)
            paths[split][family] = str(path)
            profiles[split][family] = league_profile(path)
    base = TrainingConfig(
        catalog_id=CATALOG_ID,
        allow_development=True,
        shop_action_limit_mode="force_battle",
        timesteps=timesteps,
        seed=seed,
        environments=8,
        rollout_steps=256,
        batch_size=256,
        device="cpu",
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=60,
        validation_seed=2300000,
        evaluation_interval=max(32768, timesteps // 4),
    )
    protocol = {
        "status": "experimental pilot; client-parity gates remain open",
        "catalog_id": CATALOG_ID,
        "catalog_sha256": catalog_digest(catalog),
        "base_config": asdict(base),
        "arms": ARMS,
        "paths": paths,
        "source_files_sha256": source_archive(output),
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "selection": "Highest unassisted validation success, then raw return; earliest tie",
        "scope": "Three equal-budget same-seed fresh initializations: control, swap-only cost "
        "0.005, or all-action cost 0.005. Each treatment differs from control in one coefficient. "
        "This single-seed pilot is not the three-seed confirmation or held-out test.",
        "test_status": "No test episodes evaluated. A fresh split is reserved for confirmation.",
        "completion_contract": "docs/EXPANDED_GOAL.md",
    }
    save_json(output / "protocol.json", protocol)
    save_json(output / "pool_profiles.json", profiles)
    save_json(
        output / "baseline_validation.json",
        {
            name: compact_evaluation(
                evaluate_suite(
                    name,
                    paths["validation"],
                    episodes=60,
                    seed=2300000,
                    env_kwargs={"catalog": catalog, "allow_development": True},
                )
            )
            for name in FAMILIES
        },
    )
    print(f"Prepared {output}; no learning has started", flush=True)


def verify_inputs(output: Path, protocol) -> None:
    root = Path(__file__).resolve().parents[2]
    for relative, expected in protocol["source_files_sha256"].items():
        if file_digest(str(root / relative)) != expected:
            raise ValueError(f"Source changed after protocol freeze: {relative}")
    for relative, expected in protocol["data_sha256"].items():
        if file_digest(str(output / relative)) != expected:
            raise ValueError(f"Opponent pool changed after protocol freeze: {relative}")


def run_arm(output: Path, arm: str) -> None:
    protocol = json.loads((output / "protocol.json").read_text())
    verify_inputs(output, protocol)
    config = dict(protocol["base_config"])
    config.update(protocol["arms"][arm], output_dir=str(output / arm))
    started = time.monotonic()
    with (output / f"{arm}.log").open("x") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            final_model = train(TrainingConfig(**config))
    verify_inputs(output, protocol)
    save_json(
        output / f"{arm}_complete.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "final_model": str(final_model),
            "final_model_sha256": file_digest(str(final_model)),
            "best_model": str(output / arm / "best_model.zip"),
            "best_model_sha256": file_digest(str(output / arm / "best_model.zip")),
            "training_complete": True,
            "goal_complete": False,
        },
    )
    print(f"Pilot arm {arm} completed; goal is not complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), default="control")
    parser.add_argument("--timesteps", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=1709)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    output = Path(args.output).resolve()
    if args.command == "prepare":
        prepare(output, args.timesteps, args.seed)
    else:
        run_arm(output, args.arm)


if __name__ == "__main__":
    main()
