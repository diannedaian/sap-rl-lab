"""Dependency-free commands for inspecting and exercising the sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import RandomPolicy, SpendGoldPolicy, play_episode
from .engine import AutoBattler
from .opponents import SnapshotLeague, build_spend_gold_league


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "random", "greedy", "build-league"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent-league")
    parser.add_argument("--output", default="data/leagues/spend_gold.json")
    args = parser.parse_args()

    if args.command == "build-league":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        league = build_spend_gold_league(args.episodes, args.seed)
        league.save(output)
        print(json.dumps({"output": str(output), "snapshots": len(league)}, indent=2))
        return

    provider = SnapshotLeague.load(args.opponent_league) if args.opponent_league else None
    engine = AutoBattler(opponent_provider=provider)
    if args.command == "inspect":
        engine.reset(seed=args.seed)
        print(
            json.dumps(
                {
                    "catalog": engine.catalog.catalog_id,
                    "pets": engine.catalog.rollable_pet_ids,
                    "foods": engine.catalog.rollable_food_ids,
                    "action_space": engine.codec.size,
                    "legal_actions": [
                        engine.codec.describe(action_id) for action_id in engine.legal_action_ids()
                    ],
                    "state": engine.state.to_dict(),
                },
                indent=2,
            )
        )
        return

    policy = RandomPolicy() if args.command == "random" else SpendGoldPolicy()
    summaries = [
        play_episode(engine, policy, seed=args.seed + episode) for episode in range(args.episodes)
    ]
    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "mean_return": sum(item.return_ for item in summaries) / len(summaries),
                "mean_wins": sum(item.wins for item in summaries) / len(summaries),
                "successes": sum(item.success for item in summaries),
                "truncations": sum(item.truncated for item in summaries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
