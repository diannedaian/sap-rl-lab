"""Local, opt-in backport of SB3-Contrib PR #326's stale-probability fix.

Upstream: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/326
Our Python 3.9 environment uses SB3-Contrib 2.7.1; upstream 2.8 drops Python 3.9.
Do not modify site-packages, disable validation, or alter frozen legacy policies.
"""

import torch
from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


class FreshMaskableCategorical(MaskableCategorical):
    def apply_masking(self, masks):
        if masks is not None:
            boolean_masks = torch.as_tensor(masks, dtype=torch.bool, device=self.logits.device)
            if not boolean_masks.reshape(self.logits.shape).any(dim=-1).all():
                raise ValueError("Action mask has no legal action")
        # Categorical validates every cached parameter on reinitialization.
        # A stale softmax can drift beyond the simplex tolerance in float32.
        # Remove it before the inherited reinitialization; retain validation.
        self.__dict__.pop("probs", None)
        super().apply_masking(masks)


class FreshCategoricalDistribution(MaskableCategoricalDistribution):
    def proba_distribution(self, action_logits):
        self.distribution = FreshMaskableCategorical(logits=action_logits.view(-1, self.action_dim))
        return self


class StableMaskableMultiInputPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, MaskableCategoricalDistribution):
            raise ValueError("This project backport only supports Discrete actions")
        self.action_dist = FreshCategoricalDistribution(self.action_space.n)
