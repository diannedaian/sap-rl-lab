"""Small, explicit Maskable PPO training entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
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
    initialize_from: str = ""
    action_cost: float = 0.0
    forfeit_on_limit: bool = False
    swap_cost: float = 0.0
    success_bonus_max: float = 0.0
    success_action_cost: float = 0.0
    observe_episode_actions: bool = False
    entropy_coefficient: float = 0.0
    torch_threads: int = 1
    validation_league: str = ""
    validation_episodes: int = 200
    validation_seed: int = 30000
    evaluation_interval: int = 100_000
    opponent_leagues: tuple[str, ...] = ()
    validation_leagues: Dict[str, str] = field(default_factory=dict)
    expected_initial_policy_sha256: str = ""

    def validate(self) -> None:
        if self.timesteps < 1 or self.environments < 1 or self.rollout_steps < 1:
            raise ValueError("timesteps, environments, and rollout_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        coefficients = (
            self.action_cost,
            self.entropy_coefficient,
            self.swap_cost,
            self.success_bonus_max,
            self.success_action_cost,
        )
        if any(not math.isfinite(x) or x < 0 for x in coefficients):
            raise ValueError("reward coefficients must be finite and nonnegative")
        if self.success_bonus_max and (
            not self.success_action_cost or not self.observe_episode_actions
        ):
            raise ValueError("success bonus requires a positive action cost and observed count")
        if not 0 < self.learning_rate or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid learning_rate, gamma, or gae_lambda")
        if min(self.torch_threads, self.validation_episodes, self.evaluation_interval) < 1:
            raise ValueError(
                "threads, validation episodes, and evaluation interval must be positive"
            )
        if (self.validation_league and self.validation_leagues) or (
            self.opponent_league and self.opponent_leagues
        ):
            raise ValueError("use a single league or multiple leagues, not both")
        training_paths = self.opponent_leagues or (
            (self.opponent_league,) if self.opponent_league else ()
        )
        validation_paths = tuple(self.validation_leagues.values()) or (
            (self.validation_league,) if self.validation_league else ()
        )
        if training_paths or validation_paths:
            from .evaluation import file_digest

            train_hashes = [file_digest(path) for path in training_paths]
            val_hashes = [file_digest(path) for path in validation_paths]
            if set(train_hashes) & set(val_hashes):
                raise ValueError("validation league must differ from the training league")
            if len(set(train_hashes)) != len(train_hashes):
                raise ValueError("training mixture contains duplicated pools")
            if len(set(val_hashes)) != len(val_hashes):
                raise ValueError("validation suite contains duplicated pools")
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


def policy_digest(policy) -> str:
    """Fingerprint tensor values, not ZIP timestamps or Python object identity."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def copy_initial_policy(target, source) -> str:
    """Preserve a policy exactly when appending the optional episode-count input.

    CombinedExtractor concatenates Dict spaces in their declared order. Insert
    one zero-weight input column in both first MLP layers; all other tensors
    must match. Never silently accept a different action/feature vocabulary.
    """
    import numpy as np
    import torch

    old_spaces = source.observation_space.spaces
    new_spaces = target.observation_space.spaces
    if list(old_spaces) != list(new_spaces) or source.action_space != target.action_space:
        raise ValueError("incompatible policy spaces")
    if all(old_spaces[k] == new_spaces[k] for k in old_spaces):
        target.policy.load_state_dict(source.policy.state_dict())
        return "exact_policy_weights"
    if old_spaces["global"].shape != (8,) or new_spaces["global"].shape != (9,):
        raise ValueError("only the episode-action input extension is supported")
    if any(old_spaces[k] != new_spaces[k] for k in old_spaces if k != "global"):
        raise ValueError("non-global observation features changed")
    insertion = 0
    for name, space in old_spaces.items():
        insertion += int(np.prod(space.shape))
        if name == "global":
            break
    initial = source.policy.state_dict()
    expanded = target.policy.state_dict()
    if initial.keys() != expanded.keys():
        raise ValueError("policy parameter names changed")
    for name, value in initial.items():
        if value.shape == expanded[name].shape:
            expanded[name] = value
        elif name in {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}:
            if expanded[name].shape != (value.shape[0], value.shape[1] + 1):
                raise ValueError("unexpected input weight shape")
            expanded[name] = torch.cat(
                [value[:, :insertion], value.new_zeros((value.shape[0], 1)), value[:, insertion:]],
                dim=1,
            )
        else:
            raise ValueError(f"unexpected changed parameter: {name}")
    target.policy.load_state_dict(expanded)
    return "zero_weight_episode_action_input"


def train(config: TrainingConfig) -> Path:
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    matplotlib_cache = output / "matplotlib-cache"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc

    from .env import SapAutoBattlerEnv
    from .evaluation import compact_evaluation, evaluate_policy, evaluate_suite, file_digest
    from .opponents import OpponentMixture, SnapshotLeague

    torch.set_num_threads(config.torch_threads)
    catalog_id = SapAutoBattlerEnv().engine.catalog.catalog_id
    manifest = {
        "config": asdict(config),
        "catalog_id": catalog_id,
        "training_league_sha256": file_digest(config.opponent_league)
        if config.opponent_league
        else None,
        "validation_league_sha256": file_digest(config.validation_league)
        if config.validation_league
        else None,
        "training_mixture_sha256": {path: file_digest(path) for path in config.opponent_leagues},
        "validation_suite_sha256": {
            name: file_digest(path) for name, path in config.validation_leagues.items()
        },
        "initial_model_sha256": file_digest(config.initialize_from)
        if config.initialize_from
        else None,
        "initialization": "policy_weights_only_fresh_optimizer"
        if config.initialize_from
        else "from_scratch",
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
            if config.opponent_leagues:
                opponent_provider = OpponentMixture(
                    [(1.0, SnapshotLeague.load(path)) for path in config.opponent_leagues]
                )
            return SapAutoBattlerEnv(
                opponent_provider=opponent_provider,
                action_cost=config.action_cost,
                forfeit_on_limit=config.forfeit_on_limit,
                swap_cost=config.swap_cost,
                success_bonus_max=config.success_bonus_max,
                success_action_cost=config.success_action_cost,
                observe_episode_actions=config.observe_episode_actions,
            )

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
        ent_coef=config.entropy_coefficient,
        device=config.device,
        verbose=1,
    )
    if config.initialize_from:
        initial = MaskablePPO.load(config.initialize_from, device=config.device)
        manifest["input_migration"] = copy_initial_policy(model, initial)
        del initial
        # Loading the reference model can reseed global RNGs. Restore this
        # experiment's seed after copying weights, before collecting rollouts.
        model.set_random_seed(config.seed)
    manifest["initial_policy_sha256"] = policy_digest(model.policy)
    if config.expected_initial_policy_sha256 and (
        manifest["initial_policy_sha256"] != config.expected_initial_policy_sha256
    ):
        vec_env.close()
        raise ValueError("paired initial policy weights do not match")
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = CheckpointCallback(
        save_freq=max(config.timesteps // max(config.environments * 10, 1), 1),
        save_path=str(output / "checkpoints"),
        name_prefix="sap_ppo",
    )

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = config.evaluation_interval
            self.best_score = (-1.0, float("-inf"))
            self.history: List[Dict[str, Any]] = []

        def evaluate(self) -> None:
            # Deterministic batched inference does not sample Torch's RNG. The
            # evaluator uses independent environments and episode-policy RNGs.
            if config.validation_leagues:
                result = evaluate_suite(
                    self.model,
                    config.validation_leagues,
                    episodes=config.validation_episodes,
                    seed=config.validation_seed,
                )
            else:
                result = evaluate_policy(
                    self.model,
                    episodes=config.validation_episodes,
                    seed=config.validation_seed,
                    opponent_league=config.validation_league,
                )
            result["timesteps"] = self.num_timesteps
            # The final PPO update can follow a periodic evaluation at the same
            # decision count. Preserve both records even when their steps match.
            result["evaluation_file"] = (
                f"validation_{self.num_timesteps}_eval{len(self.history):03d}.json"
            )
            score = (result["success_rate"], result["mean_return"])
            result["selected"] = score > self.best_score
            if result["selected"]:
                self.best_score = score
                self.model.save(str(output / "best_model"))
            self.history.append(compact_evaluation(result))
            (output / result["evaluation_file"]).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            (output / "validation_history.json").write_text(
                json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"validation step={self.num_timesteps} success={score[0]:.3f} "
                f"return={score[1]:.3f} selected={result['selected']}",
                flush=True,
            )

        def _on_training_start(self) -> None:
            self.evaluate()

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_evaluation:
                self.evaluate()
                self.next_evaluation += config.evaluation_interval
            return True

        def _on_training_end(self) -> None:
            self.evaluate()

    callbacks = [checkpoint]
    if config.validation_league or config.validation_leagues:
        callbacks.append(ValidationCallback())
    try:
        model.learn(total_timesteps=config.timesteps, callback=callbacks, progress_bar=False)
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
    model = MaskablePPO.load(model_path, device=device)
    env = SapAutoBattlerEnv(
        opponent_provider=opponent_provider,
        observe_episode_actions=model.observation_space["global"].shape == (9,),
    )
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
    train_parser.add_argument("--initialize-from", default="")
    train_parser.add_argument("--action-cost", type=float, default=0.0)
    train_parser.add_argument("--forfeit-on-limit", action="store_true")
    train_parser.add_argument("--swap-cost", type=float, default=0.0)
    train_parser.add_argument("--success-bonus-max", type=float, default=0.0)
    train_parser.add_argument("--success-action-cost", type=float, default=0.0)
    train_parser.add_argument("--observe-episode-actions", action="store_true")
    train_parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    train_parser.add_argument("--torch-threads", type=int, default=1)
    train_parser.add_argument("--validation-league", default="")
    train_parser.add_argument("--validation-episodes", type=int, default=200)
    train_parser.add_argument("--validation-seed", type=int, default=30000)
    train_parser.add_argument("--evaluation-interval", type=int, default=100_000)
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
                initialize_from=args.initialize_from,
                action_cost=args.action_cost,
                forfeit_on_limit=args.forfeit_on_limit,
                swap_cost=args.swap_cost,
                success_bonus_max=args.success_bonus_max,
                success_action_cost=args.success_action_cost,
                observe_episode_actions=args.observe_episode_actions,
                entropy_coefficient=args.entropy_coefficient,
                torch_threads=args.torch_threads,
                validation_league=args.validation_league,
                validation_episodes=args.validation_episodes,
                validation_seed=args.validation_seed,
                evaluation_interval=args.evaluation_interval,
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
