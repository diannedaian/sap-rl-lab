"""Read-only rollout/value inspection for expanded checkpoints (not training)."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .evaluation import file_digest, policy_environment_options
from .experiments import save_json


def inspect(model_path: str, league_path: str, output: Path, episodes: int, seed: int):
    import torch
    from sb3_contrib import MaskablePPO

    from .env import SapAutoBattlerEnv
    from .opponents import SnapshotLeague

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = MaskablePPO.load(model_path, device="cpu")
    config = json.loads((Path(model_path).parent / "run_manifest.json").read_text())["config"]
    shaping = {
        k: config[k]
        for k in (
            "action_cost",
            "swap_cost",
            "success_bonus_max",
            "success_action_cost",
            "observe_episode_actions",
        )
    }
    env = SapAutoBattlerEnv(
        opponent_provider=SnapshotLeague.load(league_path),
        **policy_environment_options(model),
        **shaping,
    )
    summary = []
    for episode_seed in range(seed, seed + episodes):
        obs, _ = env.reset(seed=episode_seed)
        steps, counts, repeats, seen = [], Counter(), 0, set()
        raw_return = objective_return = 0.0
        while True:
            state = env.engine.state.to_dict()
            key = json.dumps(
                {k: v for k, v in state.items() if k != "actions_this_turn"}, sort_keys=True
            )
            repeats += int(key in seen)
            seen.add(key)
            tensor, _ = model.policy.obs_to_tensor(obs)
            masks = env.action_masks()
            with torch.no_grad():
                distribution = model.policy.get_distribution(tensor, action_masks=masks)
                probabilities = distribution.distribution.probs.cpu().numpy()[0]
                value = float(model.policy.predict_values(tensor).item())
            action_id = int(probabilities.argmax())
            decoded = env.engine.codec.decode(action_id)
            counts[decoded.kind.value] += 1
            top = sorted(env.engine.legal_action_ids(), key=lambda a: -probabilities[a])[:5]
            obs, reward, terminated, truncated, info = env.step(action_id)
            raw_return += info["game_reward"]
            objective_return += reward
            steps.append(
                {
                    "state": state,
                    "action": env.engine.codec.describe(action_id),
                    "action_id": action_id,
                    "value": value,
                    "top_actions": [
                        {
                            "action": env.engine.codec.describe(a),
                            "probability": float(probabilities[a]),
                        }
                        for a in top
                    ],
                    "objective_reward": reward,
                    "info": info,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if terminated or truncated:
                break
        result = {
            "seed": episode_seed,
            "success": env.engine.state.wins >= 10,
            "wins": env.engine.state.wins,
            "raw_return": raw_return,
            "objective_return": objective_return,
            "actions": len(steps),
            "action_counts": dict(counts),
            "repeated_shop_states": repeats,
            "forced_turns": sum(bool(s["info"].get("forced_end_turn")) for s in steps),
            "truncated": truncated,
        }
        summary.append(result)
        save_json(output / f"episode-{episode_seed}.json", {"summary": result, "steps": steps})
    env.close()
    save_json(
        output / "summary.json",
        {
            "model": model_path,
            "model_sha256": file_digest(model_path),
            "league": league_path,
            "league_sha256": file_digest(league_path),
            "note": "Selected-checkpoint reconstructions, not historical PPO rollout values. "
            "Repeated state excludes only the shop-action counter; this diagnoses loops, "
            "not proof that every repeated action is strategically useless.",
            "episodes": summary,
        },
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--league", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2300000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    inspect(args.model, args.league, Path(args.output), args.episodes, args.seed)


if __name__ == "__main__":
    main()
