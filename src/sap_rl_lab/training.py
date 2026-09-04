"""Small, explicit Maskable PPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrainingConfig:
    timesteps: int = 100_000
    seed: int = 7
    environments: int = 4
    learning_rate: float = 3e-4
    rollout_steps: int = 512
    batch_size: int = 256
    gamma: float = 1.0
    gae_lambda: float = 0.95
    output_dir: str = "runs/ppo"


def train(config: TrainingConfig) -> Path:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = output / "matplotlib-cache"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc

    from .env import SapAutoBattlerEnv

    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def make_env(rank: int):
        def factory() -> SapAutoBattlerEnv:
            env = SapAutoBattlerEnv()
            env.reset(seed=config.seed + rank)
            return env

        return factory

    vec_env = DummyVecEnv([make_env(rank) for rank in range(config.environments)])
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(config.timesteps // max(config.environments * 10, 1), 1),
        save_path=str(output / "checkpoints"),
        name_prefix="sap_ppo",
    )
    model.learn(total_timesteps=config.timesteps, callback=checkpoint, progress_bar=False)
    model_path = output / "final_model"
    model.save(str(model_path))
    vec_env.close()
    return model_path.with_suffix(".zip")


def evaluate_model(model_path: str, episodes: int = 100, seed: int = 10_000) -> Dict[str, Any]:
    model_parent = Path(model_path).expanduser().resolve().parent
    matplotlib_cache = model_parent / "matplotlib-cache"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        import numpy as np
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc

    from .env import SapAutoBattlerEnv

    env = SapAutoBattlerEnv()
    model = MaskablePPO.load(model_path)
    returns: List[float] = []
    wins: List[int] = []
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
        wins.append(env.engine.state.wins)
        successes += int(env.engine.state.wins >= env.engine.config.target_wins)
    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_wins": float(np.mean(wins)),
        "success_rate": successes / episodes,
        "seed_start": seed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--environments", type=int, default=4)
    train_parser.add_argument("--output-dir", default="runs/ppo")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("model")
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=10_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        path = train(
            TrainingConfig(
                timesteps=args.timesteps,
                seed=args.seed,
                environments=args.environments,
                output_dir=args.output_dir,
            )
        )
        print(path)
    else:
        print(json.dumps(evaluate_model(args.model, args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
