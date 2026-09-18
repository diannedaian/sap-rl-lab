"""Opt-in PPO update diagnostics and validation guardrails; legacy runs are untouched.

Run with a JSON file containing `training` (TrainingConfig) and `guardrails`.
This is a new, weights-only fine-tuning branch, not an exact RNG/optimizer resume.
No test pools, reward changes, BC loss, automatic rollback, or automatic retries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .catalog import catalog_digest
from .env import SapAutoBattlerEnv
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .opponents import OpponentMixture, SnapshotLeague
from .round3 import source_archive
from .stable_masking import StableMaskableMultiInputPolicy
from .training import TrainingConfig, _installed_version, copy_initial_policy, policy_digest


@dataclass(frozen=True)
class GuardrailConfig:
    # Operational pilot limits, NOT industry constants or confidence bounds.
    target_kl: float | None = 0.01
    n_epochs: int = 10
    clip_range: float = 0.2
    final_lr_fraction: float = 1.0
    max_forced_episode_rate: float = 0.05
    max_no_purchase_loss_rate: float = 0.05
    max_family_success_drop: float = 0.10
    regression_patience: int = 2
    min_mean_wins: float = 1.0
    stop_on_regression: bool = True
    max_seconds: float = 3600.0

    def validate(self):
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be positive or None")
        for value in (
            self.clip_range,
            self.final_lr_fraction,
            self.max_forced_episode_rate,
            self.max_no_purchase_loss_rate,
            self.max_family_success_drop,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("guardrail fractions must be finite and in [0, 1]")
        if not self.clip_range or not self.final_lr_fraction:
            raise ValueError("clip and final learning rate must be positive")
        if self.n_epochs < 1 or self.regression_patience < 1:
            raise ValueError("epochs and patience must be positive")
        if not math.isfinite(self.min_mean_wins) or self.min_mean_wins < 0:
            raise ValueError("minimum strength must be finite and nonnegative")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("wall-clock limit must be finite and positive")


def write_new(path, data):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(text)


@contextmanager
def isolated_evaluation_rng(policy):
    """Keep evaluation from changing CPU sampling RNGs or policy training mode."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, mode = torch.get_rng_state(), policy.training
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        policy.set_training_mode(mode)


def family_metrics(result):
    """Fail closed on absent diagnostics; zero means measured zero, not missing."""
    if not result.get("families"):
        raise ValueError("validation needs per-family episode evidence")
    values = {}
    for name, family in result["families"].items():
        rows = family["episode_results"]
        if not rows or len(rows) != family["episodes"]:
            raise ValueError("incomplete validation episode rows")
        values[name] = {
            "success": float(np.mean([r["success"] for r in rows])),
            "forcing": float(np.mean([r["forced_end_turns"] > 0 for r in rows])),
            "truncation": float(np.mean([r["truncated"] for r in rows])),
            "no_purchase_loss": float(
                np.mean(
                    [
                        r["wins"] == 0
                        and not r["success"]
                        and r["action_counts"].get("buy_pet", 0) == 0
                        for r in rows
                    ]
                )
            ),
            "mean_wins": float(np.mean([r["wins"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
        }
        if not all(math.isfinite(v) for v in values[name].values()):
            raise ValueError("nonfinite validation metrics")
    return values


class ValidationGuard:
    """Select only eligible weights; stop repeated regressions after competence.

    This protects selection, not the learning trajectory. An ineligible initial
    policy may learn; a measured truncation stops immediately. No eligible model
    means no selection, never silently falling back to an unreliable checkpoint.
    """

    def __init__(self, config):
        config.validate()
        self.config = config
        self.best = None
        self.bad_streak = 0
        self.family_names = None

    def consider(self, result, checkpoint):
        c, families = self.config, family_metrics(result)
        if self.family_names is None:
            self.family_names = set(families)
        if set(families) != self.family_names:
            raise ValueError("validation families changed")
        eligible = (
            all(
                f["forcing"] <= c.max_forced_episode_rate
                and f["truncation"] == 0
                and f["no_purchase_loss"] <= c.max_no_purchase_loss_rate
                for f in families.values()
            )
            and np.mean([f["mean_wins"] for f in families.values()]) >= c.min_mean_wins
        )
        score = self.score(families)
        reasons = []
        if self.best is not None:
            if not eligible:
                reasons.append("reliability_or_basic_play_regression")
            if any(
                self.best["families"][name]["success"] - f["success"]
                > c.max_family_success_drop + 1e-12
                for name, f in families.items()
            ):
                reasons.append("family_strength_regression")
        self.bad_streak = self.bad_streak + 1 if reasons else 0
        selected = eligible and not reasons and (self.best is None or score > self.best["score"])
        if selected:
            self.best = {"checkpoint": checkpoint, "score": score, "families": families}
        truncated = any(f["truncation"] > 0 for f in families.values())
        return {
            "eligible": bool(eligible),
            "selected": bool(selected),
            "score": score,
            "families": families,
            "regression_reasons": reasons,
            "bad_streak": self.bad_streak,
            "stop": truncated
            or (c.stop_on_regression and self.bad_streak >= c.regression_patience),
            "stop_reason": "validation_truncation" if truncated else "validation_regression",
        }

    def score(self, families):
        return (
            float(np.mean([f["success"] for f in families.values()])),
            min(f["success"] for f in families.values()),
            -max(f["forcing"] for f in families.values()),
            float(np.mean([f["mean_return"] for f in families.values()])),
        )


class GuardStop(Exception):
    """An expected bounded stop; do not retry or erase the last candidate."""


class ObservedMaskablePPO(MaskablePPO):
    """Keep upstream PPO intact; measure actual optimizer steps and full-buffer KL."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["after_update", "update_writer"]

    def _logits(self, observations, masks):
        mode = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with torch.no_grad():
                dist = self.policy.get_distribution(
                    obs_as_tensor(observations, self.device), action_masks=masks
                )
                return dist.distribution.logits.detach().double().clone()
        finally:
            self.policy.set_training_mode(mode)

    def train(self):
        buffer = self.rollout_buffer
        if not buffer.full or buffer.generator_ready:
            raise ValueError("expected a fresh full rollout before each update")
        obs = {k: buffer.swap_and_flatten(v) for k, v in buffer.observations.items()}
        masks = buffer.swap_and_flatten(buffer.action_masks).astype(bool)
        actions = buffer.swap_and_flatten(buffer.actions).reshape(-1).astype(int)
        advantages = buffer.swap_and_flatten(buffer.advantages).reshape(-1).copy()
        returns = buffer.swap_and_flatten(buffer.returns).reshape(-1).copy()
        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise FloatingPointError("nonfinite GAE advantages or return targets")
        if not masks[np.arange(len(actions)), actions].all():
            raise ValueError("illegal rollout actions")
        before = self._logits(obs, masks)
        actual_steps = 0

        def count_step(*_):
            nonlocal actual_steps
            actual_steps += 1

        hook = self.policy.optimizer.register_step_post_hook(count_step)
        try:
            super().train()
        finally:
            hook.remove()
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
            raise FloatingPointError("nonfinite policy weights")
        after = self._logits(obs, masks)
        # Exact categorical KL on sampled rollout states, not a global bound.
        before = before - torch.logsumexp(before, dim=-1, keepdim=True)
        after = after - torch.logsumexp(after, dim=-1, keepdim=True)
        probs = before.exp()
        kl = (probs * (before - after)).sum(dim=-1).clamp_min(0)
        entropy = -(probs * before).sum(dim=-1)
        expected_steps = self.n_epochs * (len(actions) // self.batch_size)
        kinds = np.asarray(self.action_kinds)[actions]
        row = {
            "timesteps": self.num_timesteps,
            "rollout_decisions": len(actions),
            "optimizer_steps": actual_steps,
            "maximum_optimizer_steps": expected_steps,
            "kl_early_stopped": actual_steps < expected_steps,
            "target_kl": self.target_kl,
            "exact_rollout_kl_mean": float(kl.mean()),
            "exact_rollout_kl_max": float(kl.max()),
            "entropy_before_update": float(entropy.mean()),
            "greedy_action_change_fraction": float(
                (before.argmax(-1) != after.argmax(-1)).double().mean()
            ),
            "training": {
                k: float(v) if math.isfinite(float(v)) else None
                for k, v in self.logger.name_to_value.items()
                if k.startswith("train/")
            },
            "actions": {
                kind: {
                    "count": int((kinds == kind).sum()),
                    "raw_gae_advantage_mean": float(advantages[kinds == kind].mean()),
                    "positive_raw_gae_fraction": float((advantages[kinds == kind] > 0).mean()),
                }
                for kind in sorted(set(kinds))
            },
            "note": "Raw GAE is diagnostic; PPO normalizes advantages per minibatch. "
            "SB3 n_updates is not the actual optimizer-step count.",
        }
        if hasattr(self, "update_writer"):
            self.update_writer(row)
        if hasattr(self, "after_update"):
            self.after_update()


def train_guarded(
    config: TrainingConfig,
    guardrails: GuardrailConfig,
    *,
    evaluator=evaluate_suite,
    guard_factory=ValidationGuard,
):
    """Fresh CPU-only run with immutable checkpoints, including each evaluated weight."""
    config.validate()
    guardrails.validate()
    catalog, game = config.environment()
    if config.device != "cpu" or config.vector_backend != "dummy":
        raise ValueError("this bounded runner supports CPU DummyVecEnv only")
    if catalog.rules_version not in {5, 6} or game.shop_action_limit_mode != "force_battle":
        raise ValueError("this diagnostic runner requires the v5 force-battle contract")
    if not config.opponent_leagues or not config.validation_leagues:
        raise ValueError("explicit training and validation pools are required")
    size = config.environments * config.rollout_steps
    if config.timesteps % size or config.evaluation_interval % size:
        raise ValueError("budget and evaluation interval must align to full rollouts")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = {
        p: file_digest(p)
        for p in (
            *config.opponent_leagues,
            *config.validation_leagues.values(),
            *((config.initialize_from,) if config.initialize_from else ()),
        )
    }
    contract = {
        "schema_version": 1,
        "catalog_id": catalog.catalog_id,
        "game_config": asdict(game),
        "allow_development": True,
        "catalog_sha256": catalog_digest(catalog),
        "masking_revision": "cached-probs-clear-v1",
    }
    torch.set_num_threads(config.torch_threads)
    provider = OpponentMixture([(1.0, SnapshotLeague.load(p)) for p in config.opponent_leagues])

    def factory():
        return SapAutoBattlerEnv(
            catalog=catalog,
            config=game,
            allow_development=True,
            opponent_provider=provider,
            action_cost=config.action_cost,
            swap_cost=config.swap_cost,
            success_bonus_max=config.success_bonus_max,
            success_action_cost=config.success_action_cost,
            observe_episode_actions=config.observe_episode_actions,
            forfeit_on_limit=config.forfeit_on_limit,
        )

    env = VecMonitor(DummyVecEnv([factory for _ in range(config.environments)]))
    model = None
    try:
        env.seed(config.seed)
        model = ObservedMaskablePPO(
            StableMaskableMultiInputPolicy,
            env,
            seed=config.seed,
            device="cpu",
            learning_rate=lambda progress: (
                config.learning_rate
                * (guardrails.final_lr_fraction + (1 - guardrails.final_lr_fraction) * progress)
            ),
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.entropy_coefficient,
            target_kl=guardrails.target_kl,
            n_epochs=guardrails.n_epochs,
            clip_range=guardrails.clip_range,
            verbose=0,
        )
        model.sap_environment_contract = contract
        if config.initialize_from:
            initial = MaskablePPO.load(config.initialize_from, device="cpu")
            if getattr(initial, "sap_environment_contract", None) != contract:
                raise ValueError("initial environment contract differs")
            copy_initial_policy(model, initial)
            del initial
            model.set_random_seed(config.seed)
        initial_sha = policy_digest(model.policy)
        if (
            config.expected_initial_policy_sha256
            and initial_sha != config.expected_initial_policy_sha256
        ):
            raise ValueError("initial policy hash differs")
        assert not model.policy.optimizer.state
        model.action_kinds = [
            env.venv.envs[0].engine.codec.decode(i).kind.value for i in range(model.action_space.n)
        ]
        manifest = {
            "training": asdict(config),
            "guardrails": asdict(guardrails),
            "environment_contract": contract,
            "input_sha256": hashes,
            "source_files_sha256": source_archive(output),
            "initial_policy_sha256": initial_sha,
            "initialization": "weights_only_fresh_optimizer",
            "packages": {
                p: _installed_version(p) for p in ("torch", "sb3-contrib", "stable-baselines3")
            },
            "evaluation": "deterministic; raw game reward; validation only; post-update",
            "no_test_selection": True,
            "automatic_delivery": False,
        }
        write_new(output / "manifest.json", manifest)
        guard, evaluations = guard_factory(guardrails), []

        def evaluate():
            path = output / f"eval{len(evaluations):03d}-step{model.num_timesteps}.zip"
            model.save(path)
            with isolated_evaluation_rng(model.policy):
                result = evaluator(
                    model,
                    config.validation_leagues,
                    episodes=config.validation_episodes,
                    seed=config.validation_seed,
                )
            binding = {
                "path": str(path),
                "sha256": file_digest(path),
                "policy_sha256": policy_digest(model.policy),
                "timesteps": model.num_timesteps,
            }
            decision = guard.consider(result, binding)
            write_new(
                path.with_suffix(".json"),
                {"checkpoint": binding, "decision": decision, "evaluation": result},
            )
            evaluations.append(
                {
                    "checkpoint": binding,
                    "decision": decision,
                    "evaluation": compact_evaluation(result),
                }
            )
            if decision["stop"]:
                raise GuardStop(decision["stop_reason"])

        def after_update():
            if model.num_timesteps % config.evaluation_interval == 0:
                evaluate()
            if time.monotonic() - started > guardrails.max_seconds:
                raise GuardStop("wall_clock_limit")

        model.after_update = after_update
        reason = "budget_complete"
        with (output / "updates.jsonl").open("x") as updates:

            def record(row):
                updates.write(json.dumps(row, allow_nan=False) + "\n")
                updates.flush()

            model.update_writer = record
            try:
                evaluate()
                model.learn(config.timesteps)
                if evaluations[-1]["checkpoint"]["timesteps"] != model.num_timesteps:
                    evaluate()
            except GuardStop as exc:
                reason = str(exc)
        final = output / "last_model.zip"
        model.save(final)
        if any(file_digest(p) != digest for p, digest in hashes.items()):
            raise ValueError("input files changed during training")
        summary = {
            "stop_reason": reason,
            "actual_timesteps": model.num_timesteps,
            "elapsed_seconds": time.monotonic() - started,
            "selected": guard.best,
            "last_model": str(final),
            "last_sha256": file_digest(final),
            "evaluations": evaluations,
            "automatic_delivery": False,
        }
        write_new(output / "complete.json", summary)
        return summary
    except Exception as exc:
        write_new(
            output / "failed.json",
            {
                "error": repr(exc),
                "actual_timesteps": model.num_timesteps if model is not None else 0,
            },
        )
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON with training and guardrails; output must not exist")
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text())
    result = train_guarded(
        TrainingConfig(**payload["training"]), GuardrailConfig(**payload["guardrails"])
    )
    print(json.dumps({k: result[k] for k in ("stop_reason", "actual_timesteps", "selected")}))


if __name__ == "__main__":
    main()
