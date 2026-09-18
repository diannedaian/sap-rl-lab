"""16,384 CPU decisions solely to verify real BC-to-PPO guardrail wiring.

Not a late-collapse experiment. Never use this tiny validation to tune or promote.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from sap_rl_lab.expanded_confirmation import read
from sap_rl_lab.imitation_experiment import checked
from sap_rl_lab.ppo_guardrails import GuardrailConfig, train_guarded, write_new
from sap_rl_lab.training import TrainingConfig


def run(parent, output):
    protocol = checked(parent)
    initial = read(parent / "clone-seed204101-complete.json")["selected"]
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "purpose.json",
        {
            "purpose": "Integration smoke test only; no strength/stability conclusions or tuning",
            "decisions_per_arm": 8192,
            "total_decisions": 16384,
            "initial": initial,
            "test_pools_read": False,
            "automatic_delivery": False,
        },
    )
    results = {}
    for arm, target_kl in (("control", None), ("kl_brake", 0.01)):
        config = TrainingConfig(
            **{
                **protocol["base_config"],
                "output_dir": str(output / arm),
                "seed": 94301,
                "timesteps": 8192,
                "evaluation_interval": 4096,
                "validation_episodes": 5,
                "validation_seed": 32_000_000,
                "initialize_from": initial["path"],
                "expected_initial_policy_sha256": initial["policy_sha256"],
            }
        )
        guard = GuardrailConfig(target_kl=target_kl, max_seconds=300)
        write_new(output / f"{arm}.json", {"training": asdict(config), "guardrails": asdict(guard)})
        result = train_guarded(config, guard)
        results[arm] = {
            k: result[k] for k in ("stop_reason", "actual_timesteps", "elapsed_seconds")
        }
        print(arm, results[arm], flush=True)
    checked(parent)
    write_new(output / "complete.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("runs/imitation-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output)
