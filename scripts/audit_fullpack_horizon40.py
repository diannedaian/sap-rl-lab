"""Replay the stopped validation family under a declared 30 -> 40 horizon change."""

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from sap_rl_lab.fullpack.evaluation import evaluate_policy, file_digest
from sap_rl_lab.fullpack.horizon40 import configuration, transfer_model
from sap_rl_lab.fullpack.ppo_guardrails import write_new
from sap_rl_lab.fullpack.training import policy_digest
from sap_rl_lab.round3 import source_archive

ROOT = Path(__file__).resolve().parents[1]


def audit(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    parent = ROOT / "runs/fullpack-tier5-v1"
    source_path = parent / "candidate-84101/eval005-step1310720.zip"
    recorded = json.loads(source_path.with_suffix(".json").read_text())
    assert file_digest(source_path) == recorded["checkpoint"]["sha256"]
    source = MaskablePPO.load(source_path, device="cpu")
    target, transfer = transfer_model(source, configuration(5, seed=84101))
    try:
        checkpoint = output / "initial40.zip"
        target.save(checkpoint)
        reloaded = MaskablePPO.load(checkpoint, device="cpu")
        assert policy_digest(reloaded.policy) == transfer["policy_sha256"]
        family = "validation_stats_tempo"
        kwargs = dict(
            episodes=200,
            seed=99000000,
            opponent_league=str(parent / f"data/validation/{family}.json"),
        )
        before = evaluate_policy(source, **kwargs)
        expected = recorded["evaluation"]["families"][family]["episode_results"]
        assert before["episode_results"] == expected, "Original validation did not reload exactly"
        after = evaluate_policy(reloaded, **kwargs)
        old_case = next(r for r in before["episode_results"] if r["seed"] == 99000037)
        new_case = next(r for r in after["episode_results"] if r["seed"] == 99000037)
        assert old_case["truncated"] and old_case["reason"] == "turn_limit"
        assert old_case["battle_counts"]["draw"] == 17
        unchanged = sum(a == b for a, b in zip(before["episode_results"], after["episode_results"]))
        write_new(output / "before.json", before)
        write_new(output / "after.json", after)
        summary = {
            "source": str(source_path),
            "source_sha256": file_digest(source_path),
            "initial40": str(checkpoint),
            "initial40_sha256": file_digest(checkpoint),
            "transfer": transfer,
            "original_200_rows_exact": True,
            "identical_rows": unchanged,
            "old_case": old_case,
            "new_case": new_case,
            "old_truncations": sum(r["truncated"] for r in before["episode_results"]),
            "new_truncations": sum(r["truncated"] for r in after["episode_results"]),
            "source_files_sha256": source_archive(output),
            "script_sha256": file_digest(Path(__file__)),
            "test_scores_opened": False,
            "late_opponents": "unchanged frozen pool; nearest earlier turn if none saved",
        }
        write_new(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "original_200_rows_exact",
                        "identical_rows",
                        "old_truncations",
                        "new_truncations",
                        "old_case",
                        "new_case",
                    )
                }
            ),
            flush=True,
        )
    finally:
        target.get_env().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output.resolve())
