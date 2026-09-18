"""Unchanged BC/PPO recipe applied to explicit full-pack curriculum contracts."""

from statistics import fmean

from .ppo_guardrails import ValidationGuard
from .training import TrainingConfig

FAMILIES = ("full_stats", "full_summon", "full_tempo")


def configuration(tier=6, **kwargs):
    if tier not in (5, 6):
        raise ValueError("Full-pack training curriculum must be Tier5 or Tier6")
    return TrainingConfig(
        catalog_id="turtle-v0.46-tier5-v7" if tier == 5 else "turtle-v0.46-full-v8",
        allow_development=True,
        shop_action_limit_mode="force_battle",
        action_cost=0.005,
        environments=8,
        rollout_steps=256,
        batch_size=256,
        device="cpu",
        torch_threads=1,
        **kwargs,
    )


class LearnedFirstGuard(ValidationGuard):
    def score(self, families):
        learned = [v for k, v in families.items() if k.startswith("validation_")]
        if len(learned) != 2:
            raise ValueError("Exactly two learned validation families required")
        return (
            fmean(f["success"] for f in learned),
            min(f["success"] for f in learned),
            fmean(f["success"] for f in families.values()),
            -max(f["forcing"] for f in families.values()),
            fmean(f["mean_return"] for f in families.values()),
        )
