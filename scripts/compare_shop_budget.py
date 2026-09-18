"""Bounded, inference-only comparison of two shop-limit contracts on one frozen policy."""

import argparse
import json
import time
from pathlib import Path


def run(root, output, episodes):
    import torch
    from sb3_contrib import MaskablePPO

    from sap_rl_lab.domain import GameConfig
    from sap_rl_lab.evaluation import evaluate_policy, file_digest

    if not 1 <= episodes <= 1000:
        raise ValueError("Use 1–1000 existing diagnostic episodes per family")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    release = json.loads((root / "release/manifest.json").read_text())
    model_path = root / "release/model.zip"
    if file_digest(model_path) != release["sha256"]:
        raise ValueError("Frozen model changed")
    policy = MaskablePPO.load(model_path, device="cpu")
    protocol = json.loads((root / "protocol.json").read_text())
    seed = protocol["test_episode_seed"]
    started = time.monotonic()
    result = {
        "purpose": "Environment-design diagnostic, not retraining or a new benchmark",
        "model_sha256": release["sha256"],
        "source_model": release["model"],
        "episodes_per_family": episodes,
        "seed_start": seed,
        "training_steps": 0,
        "budget": 30,
        "limitations": "Reuses known test seeds with one existing checkpoint. An intervention "
        "changes later trajectories. It cannot establish how a newly trained policy would learn.",
        "families": {},
    }
    for family in ("greedy", "stats", "summon"):
        pool = root / f"data/test/{family}.json"
        original = json.loads((root / f"test-{release['model']}.json").read_text())
        original = original["families"][family]["episode_results"][:episodes]
        pair = {}
        for mode in ("truncate", "force_battle"):
            evaluation = evaluate_policy(
                policy,
                episodes=episodes,
                seed=seed,
                opponent_league=str(pool),
                env_kwargs={"config": GameConfig(shop_action_limit_mode=mode)},
            )
            if mode == "truncate" and evaluation["episode_results"] != original:
                raise ValueError("Legacy trajectories changed: " + family)
            pair[mode] = evaluation
        with (output / f"{family}.json").open("x") as handle:
            json.dump(pair, handle, indent=2)
        left, right = pair["truncate"], pair["force_battle"]
        paired = list(zip(left["episode_results"], right["episode_results"]))
        result["families"][family] = {
            "legacy_rows_exactly_matched": len(original),
            "league_sha256": file_digest(pool),
            "truncate_successes": sum(a["success"] for a, _ in paired),
            "force_successes": sum(b["success"] for _, b in paired),
            "truncate_cutoffs": sum(a["truncated"] for a, _ in paired),
            "force_cutoffs": sum(b["truncated"] for _, b in paired),
            "previous_cutoffs_now_success": sum(a["truncated"] and b["success"] for a, b in paired),
            **{
                key: right[key]
                for key in (
                    "forced_end_turns",
                    "forced_end_turn_rate",
                    "forced_episode_rate",
                    "success_without_forcing_rate",
                )
            },
            "mean_actions_truncate": left["mean_episode_actions"],
            "mean_actions_force": right["mean_episode_actions"],
        }
        print("VERIFIED " + family, flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/round5-confirmation-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    run(Path(args.run).resolve(), Path(args.output).resolve(), args.episodes)
