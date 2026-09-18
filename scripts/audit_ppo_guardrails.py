"""Read-only historical validation/log audit. No test reads or model re-selection.

This is retrospective guard behavior, not a new experiment or proof of causality.
SB3 prints a rollout's training metrics alongside the NEXT rollout's step count.
"""

import argparse
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read
from sap_rl_lab.imitation_experiment import checked
from sap_rl_lab.ppo_guardrails import GuardrailConfig, ValidationGuard, write_new


def parse_log(text, rollout_size):
    records = []
    for block in re.split(r"(?m)^-+\n", text):
        values = dict(re.findall(r"\|\s*(\w+)\s*\|\s*([-+\deE.]+)\s*\|", block))
        if "approx_kl" not in values:
            continue
        records.append(
            {
                "logged_at_timesteps": int(values["total_timesteps"]),
                "updated_through_timesteps": int(values["total_timesteps"]) - rollout_size,
                **{
                    k: float(values[k])
                    for k in (
                        "approx_kl",
                        "clip_fraction",
                        "entropy_loss",
                        "explained_variance",
                        "value_loss",
                    )
                },
            }
        )
    return records


def audit(parent, output):
    protocol = checked(parent)
    output.mkdir(parents=True, exist_ok=False)
    config = GuardrailConfig()
    models, input_hashes = {}, {}
    size = protocol["base_config"]["environments"] * protocol["base_config"]["rollout_steps"]
    for name in protocol["arms"]:
        folder, log = parent / name, parent / f"{name}.log"
        records = parse_log(log.read_text(), size)
        assert len(records) == protocol["steps_per_model"] // size - 1
        assert [r["updated_through_timesteps"] for r in records] == list(
            range(size, protocol["steps_per_model"], size)
        )
        input_hashes[str(log)] = file_digest(log)
        guard, decisions, first_stop = ValidationGuard(config), [], None
        for index in range(protocol["validation_records_per_model"]):
            binding_path = folder / "validation_weights" / f"eval{index:03d}.json"
            binding = read(binding_path)
            result_path = folder / binding["evaluation_file"]
            input_hashes[str(binding_path)] = file_digest(binding_path)
            input_hashes[str(result_path)] = file_digest(result_path)
            assert file_digest(binding["path"]) == binding["sha256"]
            if first_stop is None:
                result = read(result_path)
                decision = guard.consider(result, binding)
                decisions.append({"timesteps": binding["timesteps"], **decision})
                if decision["stop"]:
                    first_stop = binding["timesteps"]
        models[name] = {
            "logged_updates": len(records),
            "training_statistics": {
                key: {
                    "median": float(np.median([r[key] for r in records])),
                    "p95": float(np.quantile([r[key] for r in records], 0.95)),
                    "maximum": float(max(r[key] for r in records)),
                }
                for key in ("approx_kl", "clip_fraction", "entropy_loss", "value_loss")
            },
            "largest_kl_record": max(records, key=lambda row: row["approx_kl"]),
            "retrospective_first_stop": first_stop,
            "retrospective_eligible_reference": guard.best,
            "decisions_through_stop": decisions,
        }
    checked(parent)
    assert all(file_digest(path) == digest for path, digest in input_hashes.items())
    result = {
        "parent": str(parent),
        "guardrails": asdict(config),
        "models": models,
        "input_sha256": input_hashes,
        "limitations": "Retrospective validation/log analysis, not a randomized experiment. "
        "No test data used, no old selections changed, no new model promoted. Old approx_kl "
        "is the last PPO epoch's minibatch mean, not exact full-buffer post-update KL. "
        "The final training update was not dumped in old stdout. The proposed 5% and "
        "10-percentage-point limits are pilot guardrails, not industry constants.",
    }
    write_new(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("runs/imitation-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parent, args.output)
    for name, model in result["models"].items():
        reference = model["retrospective_eligible_reference"]
        print(
            name,
            "median_kl",
            round(model["training_statistics"]["approx_kl"]["median"], 4),
            "would_stop",
            model["retrospective_first_stop"],
            "eligible_step",
            reference["checkpoint"]["timesteps"] if reference else None,
        )
