"""One bounded forty-pet pilot, using the inherited PPO/action-cost recipe.

This preparatory pilot has no held-out test and makes no generalization claim.
Full mixed-opponent training and independent-generator evaluation remain required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .catalog import catalog_digest, load_catalog_by_id
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .expanded import run_arm, verify_inputs
from .experiments import save_json
from .opponents import build_scripted_league
from .round3 import league_profile, source_archive
from .training import TrainingConfig

CATALOG_ID = "turtle-v0.46-midgame-v5"
FAMILIES = ("midgame_stats", "midgame_summon", "midgame_tempo")
DECISIONS = 1_048_576
ROOT = Path(__file__).resolve().parents[2]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog_by_id(CATALOG_ID)
    stress = json.loads((ROOT / "runs/midgame-stress-v1/summary.json").read_text())
    if stress["catalog_sha256"] != catalog_digest(catalog):
        raise ValueError("Stress evidence does not cover this catalog")
    for module in (
        "catalog",
        "domain",
        "engine",
        "env",
        "events",
        "midgame_events",
        "shop",
        "opponents",
        "midgame_baselines",
    ):
        relative = f"src/sap_rl_lab/{module}.py"
        if stress["source_files_sha256"][relative] != file_digest(str(ROOT / relative)):
            raise ValueError(f"Rules changed after stress audit: {relative}")
    if stress["battles_twice_reproduced"] < 6000 or stress["episodes_exactly_replayed"] < 160:
        raise ValueError("Midgame stress gate incomplete")
    paths, profiles = {}, {}
    for split, seed, count in (("train", 15_000_000, 120), ("validation", 15_100_000, 60)):
        folder = output / "data" / split
        folder.mkdir(parents=True)
        paths[split] = {}
        for index, family in enumerate(FAMILIES):
            path = folder / f"{family}.json"
            build_scripted_league(family, count, seed + 10_000 * index, catalog=catalog).save(path)
            paths[split][family] = str(path)
            profiles[f"{split}/{family}"] = league_profile(path)
    config = TrainingConfig(
        catalog_id=CATALOG_ID,
        allow_development=True,
        shop_action_limit_mode="force_battle",
        action_cost=0.005,
        timesteps=DECISIONS,
        seed=15101,
        environments=8,
        rollout_steps=256,
        batch_size=256,
        device="cpu",
        torch_threads=1,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=60,
        validation_seed=15_200_000,
        evaluation_interval=262_144,
    )
    config.validate()
    save_json(
        output / "protocol.json",
        {
            "catalog_id": CATALOG_ID,
            "catalog_sha256": catalog_digest(catalog),
            "base_config": asdict(config),
            "arms": {"baseline": {}},
            "paths": paths,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(p.relative_to(output)): file_digest(str(p))
                for p in sorted((output / "data").rglob("*.json"))
            },
            "selection": "win-first_worst-family_forcing_return-v1; earliest exact tie",
            "scope": "Single fixed-budget fresh initialization. Same small PPO, gamma=1, "
            "learning rate 3e-4, entropy=0, all-action cost 0.005. No hyperparameter search.",
            "test_status": "No held-out test opened or used for model selection. New independent "
            "generators and mixed-opponent confirmation are still required.",
            "stress_evidence_sha256": file_digest(
                str(ROOT / "runs/midgame-stress-v1/summary.json")
            ),
        },
    )
    save_json(output / "pool_profiles.json", profiles)
    print(f"Prepared {output}; 1,048,576-decision pilot, no test set", flush=True)


def inspect(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = json.loads((output / "protocol.json").read_text())
    verify_inputs(output, protocol)
    config = protocol["base_config"]
    folder = output / "baseline"
    history = json.loads((folder / "validation_history.json").read_text())
    final = MaskablePPO.load(folder / "final_model.zip", device="cpu")
    if final.num_timesteps != DECISIONS:
        raise ValueError("Pilot training budget incomplete")
    model = MaskablePPO.load(folder / "best_model.zip", device="cpu")
    selected = [r for r in history if r["selected"]][-1]
    recorded = json.loads((folder / selected["evaluation_file"]).read_text())
    reloaded = evaluate_suite(
        model,
        protocol["paths"]["validation"],
        episodes=config["validation_episodes"],
        seed=config["validation_seed"],
    )
    for family in FAMILIES:
        if (
            recorded["families"][family]["episode_results"]
            != reloaded["families"][family]["episode_results"]
        ):
            raise ValueError(f"Selected-model reload differs for {family}")
    save_json(output / "reload.json", reloaded)
    baseline = {
        name: compact_evaluation(
            evaluate_suite(
                name,
                protocol["paths"]["validation"],
                episodes=config["validation_episodes"],
                seed=config["validation_seed"],
                env_kwargs={"catalog": load_catalog_by_id(CATALOG_ID), "allow_development": True},
            )
        )
        for name in FAMILIES
    }
    save_json(output / "baseline_validation.json", baseline)
    save_json(
        output / "pilot_summary.json",
        {
            "budget_complete": True,
            "reload_rows_exact": 3 * config["validation_episodes"],
            "selected_step": selected["timesteps"],
            "selected_validation": compact_evaluation(reloaded),
            "final_validation": history[-1],
            "curves": history,
            "elapsed_seconds": json.loads((output / "baseline_complete.json").read_text())[
                "elapsed_seconds"
            ],
            "goal_complete": False,
            "next_gate": "Review learning, forcing, truncation and coverage before bounded "
            "mixed-opponent training. Do not confuse this with independent test results.",
        },
    )
    verify_inputs(output, protocol)
    print("Pilot complete and exactly reloaded; full goal remains incomplete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "inspect", "pipeline"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command in {"prepare", "pipeline"}:
        prepare(output)
    if args.command in {"run", "pipeline"}:
        run_arm(output, "baseline")
    if args.command in {"inspect", "pipeline"}:
        inspect(output)


if __name__ == "__main__":
    main()
