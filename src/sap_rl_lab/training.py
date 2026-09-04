"""Small, explicit Maskable PPO training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
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
    opponent_league: str = ""
    vector_backend: str = "dummy"
    device: str = "auto"
    source_commit: str = ""

    def validate(self) -> None:
        if self.timesteps < 1 or self.environments < 1 or self.rollout_steps < 1:
            raise ValueError("timesteps, environments, and rollout_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.vector_backend not in {"dummy", "subproc"}:
            raise ValueError("vector_backend must be 'dummy' or 'subproc'")
        rollout_size = self.rollout_steps * self.environments
        if rollout_size % self.batch_size:
            raise ValueError(
                "rollout_steps * environments must be divisible by batch_size "
                "to avoid partial PPO minibatches"
            )


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def train(config: TrainingConfig) -> Path:
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = output / "matplotlib-cache"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc

    from .env import SapAutoBattlerEnv
    from .opponents import SnapshotLeague

    catalog_id = SapAutoBattlerEnv().engine.catalog.catalog_id
    manifest = {
        "config": asdict(config),
        "catalog_id": catalog_id,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            package: _installed_version(package)
            for package in (
                "gymnasium",
                "numpy",
                "sb3-contrib",
                "stable-baselines3",
                "torch",
            )
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def make_env(rank: int):
        def factory() -> SapAutoBattlerEnv:
            opponent_provider = (
                SnapshotLeague.load(config.opponent_league) if config.opponent_league else None
            )
            return SapAutoBattlerEnv(opponent_provider=opponent_provider)

        return factory

    factories = [make_env(rank) for rank in range(config.environments)]
    if config.vector_backend == "subproc":
        raw_vec_env = SubprocVecEnv(factories, start_method="forkserver")
    else:
        raw_vec_env = DummyVecEnv(factories)
    vec_env = VecMonitor(raw_vec_env, filename=str(output / "training"))
    # VecEnv seeds are consumed by its next reset. This is the authoritative
    # seed path; seeding a factory reset can be silently lost by model.learn().
    vec_env.seed(config.seed)
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device=config.device,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(config.timesteps // max(config.environments * 10, 1), 1),
        save_path=str(output / "checkpoints"),
        name_prefix="sap_ppo",
    )
    try:
        model.learn(total_timesteps=config.timesteps, callback=checkpoint, progress_bar=False)
        model_path = output / "final_model"
        model.save(str(model_path))
    finally:
        vec_env.close()
    return model_path.with_suffix(".zip")


def evaluate_model(
    model_path: str,
    episodes: int = 100,
    seed: int = 10_000,
    opponent_league: str = "",
    device: str = "auto",
) -> Dict[str, Any]:
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
    from .opponents import SnapshotLeague

    opponent_provider = SnapshotLeague.load(opponent_league) if opponent_league else None
    env = SapAutoBattlerEnv(opponent_provider=opponent_provider)
    model = MaskablePPO.load(model_path, device=device)
    returns: List[float] = []
    wins: List[int] = []
    episode_lengths: List[int] = []
    battle_counts = {"win": 0, "draw": 0, "loss": 0}
    successes = 0
    truncations = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        while True:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += reward
            episode_length += 1
            outcome = info.get("battle_outcome")
            if outcome in battle_counts:
                battle_counts[outcome] += 1
            if terminated or truncated:
                truncations += int(truncated)
                break
        returns.append(episode_return)
        wins.append(env.engine.state.wins)
        episode_lengths.append(episode_length)
        successes += int(env.engine.state.wins >= env.engine.config.target_wins)
    env.close()
    total_battles = max(sum(battle_counts.values()), 1)
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_wins": float(np.mean(wins)),
        "success_rate": successes / episodes,
        "truncation_rate": truncations / episodes,
        "mean_episode_actions": float(np.mean(episode_lengths)),
        "battle_counts": battle_counts,
        "battle_rates": {
            outcome: count / total_battles for outcome, count in battle_counts.items()
        },
        "seed_start": seed,
        "opponent_league": opponent_league or None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--environments", type=int, default=4)
    train_parser.add_argument("--output-dir", default="runs/ppo")
    train_parser.add_argument("--opponent-league", default="")
    train_parser.add_argument("--vector-backend", choices=["dummy", "subproc"], default="dummy")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--source-commit", default=os.environ.get("SAP_RL_COMMIT", ""))
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--rollout-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--gamma", type=float, default=1.0)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("model")
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=10_000)
    eval_parser.add_argument("--opponent-league", default="")
    eval_parser.add_argument("--device", default="auto")
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
                opponent_league=args.opponent_league,
                vector_backend=args.vector_backend,
                device=args.device,
                source_commit=args.source_commit,
                learning_rate=args.learning_rate,
                rollout_steps=args.rollout_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
        )
        print(path)
    else:
        print(
            json.dumps(
                evaluate_model(
                    args.model,
                    args.episodes,
                    args.seed,
                    args.opponent_league,
                    args.device,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
