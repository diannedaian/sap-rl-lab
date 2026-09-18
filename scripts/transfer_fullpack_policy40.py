"""Explicit 40-turn v7 -> v8 curriculum transfer, preserving weights.

This is initialization for a new training run, not evidence of Tier6 performance.
No Adam state, RNG trajectory, opponent pools, or prior evaluation scores transfer.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import obs_as_tensor

from sap_rl_lab.fullpack.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.horizon40 import configuration
from sap_rl_lab.fullpack.imitation import fresh_model, make_env
from sap_rl_lab.fullpack.ppo_guardrails import write_new
from sap_rl_lab.fullpack.training import copy_initial_policy, policy_digest


def transfer(source_path, output, seed):
    source_path = source_path.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    source_sha = file_digest(source_path)
    source = MaskablePPO.load(source_path, device="cpu")
    old_contract = source.sap_environment_contract
    if old_contract["game_config"]["max_turns"] != 40:
        raise ValueError("Source must already use the authorized 40-turn horizon")
    if old_contract["catalog_id"] != "turtle-v0.46-tier5-v7":
        raise ValueError("Curriculum source must be an explicit fullpack Tier5 checkpoint")
    old_catalog = load_catalog_by_id(old_contract["catalog_id"])
    if catalog_digest(old_catalog) != old_contract["catalog_sha256"]:
        raise ValueError("Source rules/catalog changed")
    config = configuration(tier=6, seed=seed)
    new_catalog, new_game = config.environment()
    for field in ("pets", "foods"):
        a, b = getattr(old_catalog, field), getattr(new_catalog, field)
        if list(a) != list(b) or any(asdict(a[k]) != asdict(b[k]) for k in a):
            raise ValueError("Feature vocabulary or pet/food meanings changed")
    expected_game = {**old_contract["game_config"], "shop_rules": "turtle_full", "max_shop_tier": 6}
    if expected_game != asdict(new_game):
        raise ValueError("Unexpected game changes beyond the Tier6 shop unlock")
    target, env = fresh_model(config), make_env(config)
    try:
        if old_contract["masking_revision"] != target.sap_environment_contract["masking_revision"]:
            raise ValueError("Action masking semantics changed")
        if copy_initial_policy(target, source) != "exact_policy_weights":
            raise ValueError("This transfer requires exact shapes and policy weights")
        digest = policy_digest(source.policy)
        if policy_digest(target.policy) != digest or target.policy.optimizer.state:
            raise ValueError("Weights changed or optimizer was inherited")
        # Exercise both policies on identical, legally reached Tier6 observations.
        # This checks inference identity, not whether the policy plays well.
        source.policy.set_training_mode(False)
        target.policy.set_training_mode(False)
        obs, _ = env.reset(seed=seed + 10_000_000)
        with torch.no_grad():
            for i in range(64):
                mask = env.action_masks()
                tensors = obs_as_tensor({k: np.expand_dims(v, 0) for k, v in obs.items()}, "cpu")
                a = source.policy.get_distribution(tensors, action_masks=mask).distribution.probs
                b = target.policy.get_distribution(tensors, action_masks=mask).distribution.probs
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(
                    source.policy.predict_values(tensors),
                    target.policy.predict_values(tensors),
                    rtol=0,
                    atol=0,
                )
                action, _ = target.predict(obs, action_masks=mask, deterministic=True)
                obs, _, done, cut, _ = env.step(int(action))
                if done or cut:
                    obs, _ = env.reset(seed=seed + 10_000_001 + i)
        if file_digest(source_path) != source_sha:
            raise ValueError("Source file changed during transfer")
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = output / "initial.zip"
        target.save(checkpoint)
        reloaded = MaskablePPO.load(checkpoint, device="cpu")
        if policy_digest(reloaded.policy) != digest:
            raise ValueError("Transferred checkpoint did not reload exactly")
        record = {
            "source": str(source_path),
            "source_sha256": source_sha,
            "path": str(checkpoint),
            "sha256": file_digest(checkpoint),
            "policy_sha256": digest,
            "old_contract": old_contract,
            "new_contract": target.sap_environment_contract,
            "exact_inference_states": 64,
            "fresh_optimizer": True,
            "initialization_only_not_tier6_performance": True,
            "script_sha256": file_digest(Path(__file__)),
        }
        write_new(output / "transfer.json", record)
        print(f"Exact policy/value transfer and reload passed: {checkpoint}", flush=True)
        return record
    finally:
        target.get_env().close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    transfer(args.source, args.output.resolve(), args.seed)
