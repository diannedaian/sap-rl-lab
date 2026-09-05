"""Batched, seeded evaluation with episode-level evidence and failure diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_policy(
    policy: Any,
    *,
    episodes: int = 500,
    seed: int = 20000,
    opponent_league: str = "",
    batch_size: int = 32,
    deterministic: bool = True,
    env_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate a loaded masked policy or the names ``random`` and ``greedy``.

    Each episode owns its simulator and baseline-policy RNG. Neural inference
    is batched; no training happens here. Keep per-episode rows for paired
    comparisons and separate truncation causes from ordinary game losses.
    """
    import numpy as np

    from .actions import ActionKind
    from .baselines import RandomPolicy, SpendGoldPolicy
    from .env import SapAutoBattlerEnv
    from .opponents import SnapshotLeague

    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    baseline = None
    if isinstance(policy, str):
        if policy not in {"random", "greedy"}:
            raise ValueError("unknown baseline")
        baseline = RandomPolicy() if policy == "random" else SpendGoldPolicy()
    provider = SnapshotLeague.load(opponent_league) if opponent_league else None
    rows = []
    total_actions: Counter = Counter()
    total_battles: Counter = Counter()
    endings: Counter = Counter()
    examples = []
    for start in range(0, episodes, batch_size):
        envs = [
            SapAutoBattlerEnv(opponent_provider=provider, **(env_kwargs or {}))
            for _ in range(min(batch_size, episodes - start))
        ]
        episode_seeds = [seed + start + i for i in range(len(envs))]
        rngs = [random.Random(s + 1_000_003) for s in episode_seeds]
        observations = [env.reset(seed=s)[0] for env, s in zip(envs, episode_seeds)]
        records = [
            {
                "seed": s,
                "return": 0.0,
                "actions": 0,
                "battles": 0,
                "gold_at_end_turn": 0,
                "empty_slots_at_end_turn": 0,
                "action_counts": Counter(),
                "battle_counts": Counter(),
                "tail": [],
            }
            for s in episode_seeds
        ]
        active = list(range(len(envs)))
        try:
            while active:
                if baseline is None:
                    obs_batch = {
                        key: np.stack([observations[i][key] for i in active])
                        for key in observations[active[0]]
                    }
                    masks = np.stack([envs[i].action_masks() for i in active])
                    actions, _ = policy.predict(
                        obs_batch, action_masks=masks, deterministic=deterministic
                    )
                else:
                    actions = [
                        envs[i].engine.codec.encode(baseline.choose(envs[i].engine, rngs[i]))
                        for i in active
                    ]
                next_active = []
                for i, action_id in zip(active, actions):
                    env = envs[i]
                    record = records[i]
                    action_id = int(action_id)
                    action = env.engine.codec.decode(action_id)
                    record["action_counts"][action.kind.value] += 1
                    record["tail"].append(env.engine.codec.describe(action_id))
                    record["tail"] = record["tail"][-12:]
                    if action.kind is ActionKind.END_TURN:
                        record["gold_at_end_turn"] += env.engine.state.gold
                        record["empty_slots_at_end_turn"] += env.engine.config.max_team_size - len(
                            env.engine.state.team
                        )
                    observations[i], reward, terminated, truncated, info = env.step(action_id)
                    record["return"] += reward
                    record["actions"] += 1
                    outcome = info.get("battle_outcome")
                    if outcome:
                        record["battle_counts"][outcome] += 1
                        record["battles"] += 1
                    if not (terminated or truncated):
                        next_active.append(i)
                        continue
                    success = env.engine.state.wins >= env.engine.config.target_wins
                    reason = info.get("reason", "success" if success else "lives_exhausted")
                    record.update(
                        wins=env.engine.state.wins,
                        success=success,
                        truncated=truncated,
                        reason=reason,
                        final_turn=env.engine.state.turn,
                    )
                    if truncated and len(examples) < 8:
                        examples.append(
                            {
                                "seed": record["seed"],
                                "reason": reason,
                                "tail": list(record["tail"]),
                                "state": env.engine.state.to_dict(),
                            }
                        )
                    record.pop("tail")
                    total_actions.update(record["action_counts"])
                    total_battles.update(record["battle_counts"])
                    endings[reason] += 1
                    rows.append(record)
                active = next_active
        finally:
            for env in envs:
                env.close()
    rows.sort(key=lambda row: row["seed"])
    battle_total = sum(total_battles.values())
    return {
        "episodes": episodes,
        "seed_start": seed,
        "deterministic": deterministic,
        "league_sha256": file_digest(opponent_league) if opponent_league else None,
        "mean_return": fmean(row["return"] for row in rows),
        "return_std": pstdev(row["return"] for row in rows),
        "mean_wins": fmean(row["wins"] for row in rows),
        "success_rate": fmean(row["success"] for row in rows),
        "truncation_rate": fmean(row["truncated"] for row in rows),
        "mean_episode_actions": fmean(row["actions"] for row in rows),
        "battle_counts": dict(total_battles),
        "battle_rates": {
            k: total_battles[k] / max(battle_total, 1) for k in ("win", "draw", "loss")
        },
        "action_counts": dict(total_actions),
        "ending_reasons": dict(endings),
        "mean_unspent_gold_per_battle": (
            sum(row["gold_at_end_turn"] for row in rows) / max(battle_total, 1)
        ),
        "mean_empty_slots_per_battle": (
            sum(row["empty_slots_at_end_turn"] for row in rows) / max(battle_total, 1)
        ),
        "failure_examples": examples,
        "episode_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="Model zip, random, or greedy")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--opponent-league", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.policy in {"random", "greedy"}:
        policy = args.policy
    else:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        policy = MaskablePPO.load(args.policy, device=args.device)
    results = evaluate_policy(
        policy,
        episodes=args.episodes,
        seed=args.seed,
        opponent_league=args.opponent_league,
        batch_size=args.batch_size,
    )
    if not isinstance(policy, str):
        results["model_sha256"] = file_digest(args.policy)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in {"episode_results", "failure_examples"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
