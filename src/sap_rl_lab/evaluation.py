"""Batched, seeded evaluation with episode-level evidence and failure diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_environment_options(policy: Any, overrides=None):
    """Restore the saved transition contract, with deliberate diagnostic overrides."""
    from .catalog import catalog_digest, load_catalog_by_id
    from .domain import GameConfig

    options = {}
    contract = getattr(policy, "sap_environment_contract", None)
    if contract is not None:
        if contract.get("schema_version") != 1:
            raise ValueError("Unsupported saved environment contract")
        options["config"] = GameConfig(**contract["game_config"])
        options["catalog"] = load_catalog_by_id(contract["catalog_id"])
        if contract.get("catalog_sha256") and (
            catalog_digest(options["catalog"]) != contract["catalog_sha256"]
        ):
            raise ValueError("Saved model catalog content or observation vocabulary changed")
        if contract.get("allow_development"):
            options["allow_development"] = True
    options.update(overrides or {})
    return options


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
    from .baselines import scripted_policy
    from .env import SapAutoBattlerEnv
    from .opponents import SnapshotLeague

    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    baseline = None
    if isinstance(policy, str):
        baseline = scripted_policy(policy)
    # Observation compatibility is separate from reward shaping: evaluation
    # retains zero costs/bonuses unless explicitly overridden by its caller.
    environment_options = policy_environment_options(policy, env_kwargs)
    if baseline is None and hasattr(policy, "observation_space"):
        width = policy.observation_space["global"].shape
        base_width = (
            10
            if environment_options.get("catalog")
            and environment_options["catalog"].rules_version == 6
            else 8
        )
        if width not in {(base_width,), (base_width + 1,)}:
            raise ValueError(f"unsupported global observation shape: {width}")
        environment_options.setdefault("observe_episode_actions", width == (base_width + 1,))
    provider = SnapshotLeague.load(opponent_league) if opponent_league else None
    rows = []
    total_actions: Counter = Counter()
    total_battles: Counter = Counter()
    endings: Counter = Counter()
    examples = []
    for start in range(0, episodes, batch_size):
        envs = [
            SapAutoBattlerEnv(opponent_provider=provider, **environment_options)
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
        force_mode = envs[0].engine.config.shop_action_limit_mode == "force_battle"
        track_attack_limit = envs[0].engine.catalog.battle_attack_limit is not None
        if track_attack_limit:
            for record in records:
                record["battle_attack_limit_draws"] = 0
        if force_mode:
            for record in records:
                record["forced_end_turns"] = 0
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
                    if info.get("forced_end_turn"):
                        record["forced_end_turns"] += 1
                        record["gold_at_end_turn"] += info["gold_before_battle"]
                        record["empty_slots_at_end_turn"] += info["empty_slots_before_battle"]
                    record["return"] += reward
                    record["actions"] += 1
                    outcome = info.get("battle_outcome")
                    if info.get("battle_attack_limit_reached"):
                        record["battle_attack_limit_draws"] += 1
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
    result = {
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
    if track_attack_limit:
        limit_draws = sum(row["battle_attack_limit_draws"] for row in rows)
        result.update(
            battle_attack_limit=envs[0].engine.catalog.battle_attack_limit,
            battle_attack_limit_draws=limit_draws,
            battle_attack_limit_draw_rate=limit_draws / max(battle_total, 1),
            battle_attack_limit_episode_rate=fmean(
                row["battle_attack_limit_draws"] > 0 for row in rows
            ),
        )
    if force_mode:
        forced = sum(row["forced_end_turns"] for row in rows)
        result.update(
            environment_contract={
                "schema_version": 1,
                "catalog_id": envs[0].engine.catalog.catalog_id,
                "game_config": asdict(envs[0].engine.config),
            },
            forced_end_turns=forced,
            forced_end_turn_rate=forced / max(battle_total, 1),
            forced_episode_rate=fmean(row["forced_end_turns"] > 0 for row in rows),
            success_without_forcing_rate=fmean(
                row["success"] and not row["forced_end_turns"] for row in rows
            ),
        )
        if envs[0].engine.catalog.development_only:
            from .catalog import catalog_digest

            result["environment_contract"]["catalog_sha256"] = catalog_digest(
                envs[0].engine.catalog
            )
    return result


def compact_evaluation(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep per-family summaries but omit episode rows from learning histories."""
    return {
        key: (
            {name: compact_evaluation(value) for name, value in item.items()}
            if key == "families"
            else item
        )
        for key, item in result.items()
        if key not in {"episode_results", "failure_examples"}
    }


def evaluate_suite(
    policy: Any, leagues: Dict[str, str], *, episodes: int, seed: int, env_kwargs=None
):
    """Equal-family macro average; the stress-test family is supplied separately."""
    if not leagues:
        raise ValueError("suite needs at least one league")
    families = {
        name: evaluate_policy(
            policy, episodes=episodes, seed=seed, opponent_league=path, env_kwargs=env_kwargs
        )
        for name, path in leagues.items()
    }
    result = {
        "families": families,
        "episodes_per_family": episodes,
        "seed_start": seed,
        **{
            metric: fmean(result[metric] for result in families.values())
            for metric in (
                "success_rate",
                "mean_return",
                "mean_wins",
                "truncation_rate",
                "mean_episode_actions",
            )
        },
    }
    values = list(families.values())
    if all("forced_end_turns" in value for value in values):
        forced = sum(value["forced_end_turns"] for value in values)
        battles = sum(sum(value["battle_counts"].values()) for value in values)
        result.update(
            environment_contract=values[0]["environment_contract"],
            forced_end_turns=forced,
            forced_end_turn_rate=forced / max(battles, 1),
            forced_episode_rate=fmean(value["forced_episode_rate"] for value in values),
            success_without_forcing_rate=fmean(
                value["success_without_forcing_rate"] for value in values
            ),
        )
    return result


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
    if args.policy in {"random", "greedy", "stats", "summon"}:
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
