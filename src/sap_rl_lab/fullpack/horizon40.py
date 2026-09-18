"""Explicit 40-turn computational horizon; frozen 30-turn experiments stay unchanged.

Forty is a sandbox safety limit, not a claimed official Arena rule. Shop action
limits, rewards, pet rules and battle exchange limits are independent of it.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .imitation import fresh_model
from .recipe import configuration as previous_configuration
from .training import TrainingConfig, copy_initial_policy, policy_digest


@dataclass(frozen=True)
class HorizonTrainingConfig(TrainingConfig):
    max_turns: int = 40

    def environment(self):
        catalog, game = super().environment()
        if self.max_turns != 40:
            raise ValueError("This experiment explicitly uses a 40-turn horizon")
        return catalog, replace(game, max_turns=self.max_turns)


def configuration(tier=6, **kwargs):
    return HorizonTrainingConfig(**asdict(previous_configuration(tier=tier, **kwargs)))


def transfer_model(source, config):
    """Preserve the time feature's pre-30-turn function despite changing t/30 to t/40.

    This rescales the time input column in BOTH actor and critic. It is weights-only
    initialization with a fresh optimizer, not exact trajectory/optimizer resumption.
    Beyond turn30 the time input extrapolates instead of staying clipped at turn30.
    """
    old = source.sap_environment_contract
    target = fresh_model(config)
    try:
        new = target.sap_environment_contract
        expected = {**old, "game_config": {**old["game_config"], "max_turns": 40}}
        if new != expected or old["game_config"]["max_turns"] not in (30, 40):
            raise ValueError("Only the 30-to-40 horizon change is allowed")
        if source.observation_space["global"].shape != (10,):
            raise ValueError("This transfer does not support episode-action bonus features")
        if copy_initial_policy(target, source) != "exact_policy_weights":
            raise ValueError("Observation/action schema changed")
        column = 0
        for name, space in source.observation_space.spaces.items():
            if name == "global":
                break
            column += int(np.prod(space.shape))
        scale = 40 / old["game_config"]["max_turns"]
        with torch.no_grad():
            target.policy.mlp_extractor.policy_net[0].weight[:, column].mul_(scale)
            target.policy.mlp_extractor.value_net[0].weight[:, column].mul_(scale)
        if target.policy.optimizer.state:
            raise ValueError("Transfer must use a fresh optimizer")
        return target, {
            "source_policy_sha256": policy_digest(source.policy),
            "policy_sha256": policy_digest(target.policy),
            "old_contract": old,
            "new_contract": new,
            "turn_column": column,
            "turn_column_multiplier": scale,
            "fresh_optimizer": True,
            "function_preserved_through_turn30_with_float_tolerance": True,
        }
    except Exception:
        target.get_env().close()
        raise
