"""Post-pipeline review: correct a frozen report's freeze-action naming defect.

Never trains or evaluates new episodes, never edits frozen source/evidence, and
never selects a checkpoint from test results. See KL_80_METRIC_ADDENDUM.md.
"""

import argparse
from pathlib import Path

from sap_rl_lab.actions import ActionKind
from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read
from sap_rl_lab.imitation_experiment import require_reloads
from sap_rl_lab.kl_80_experiment import acceptance, frozen, result_metrics
from sap_rl_lab.midgame import ROOT
from sap_rl_lab.ppo_guardrails import write_new


def corrected_metrics(result):
    measured = result_metrics(result)
    rows = [r for f in result["families"].values() for r in f["episode_results"]]
    allowed = {kind.value for kind in ActionKind}
    if any(set(r["action_counts"]) - allowed for r in rows):
        raise ValueError("Unknown action-count field; do not silently omit it")
    battles = sum(r["battles"] for r in rows)
    movements = sum(
        r["action_counts"].get(k, 0)
        for r in rows
        for k in (ActionKind.SWAP.value, ActionKind.FREEZE.value)
    )
    measured["swap_freeze_actions_per_battle"] = movements / max(1, battles)
    return measured


def run(output):
    p, selected = frozen(output)
    require_reloads(output, selected)
    if (output / "pipeline_failed.json").exists():
        raise ValueError("Failed pipeline; no acceptance decision")
    if not read(output / "pipeline_complete.json")["training_evaluation_and_audit_complete"]:
        raise ValueError("Pipeline incomplete")
    audit = read(output / "audit/summary.json")
    if not all(
        audit[k]
        for k in (
            "all_curves_and_evaluations_recounted",
            "all_replays_exact",
            "objective_formula_verified",
            "all_reloads_precede_all_tests",
            "acceptance_recomputed",
        )
    ):
        raise ValueError("Incomplete evidence audit")
    summary, corrected, hashes = read(output / "summary.json"), {}, {}
    for name, item in selected["models"].items():
        path = output / f"test-{name}.json"
        result = read(path)
        if result["model_sha256"] != item["sha256"] or result["selection_sha256"] != file_digest(
            output / "selection.json"
        ):
            raise ValueError("Test binding differs")
        if result_metrics(result) != summary["results"][name]:
            raise ValueError("Original summary cannot be reproduced")
        corrected[name] = corrected_metrics(result)
        hashes[str(path)] = file_digest(path)
    for name in p["arms"]:
        done = read(output / name / "complete.json")
        if (
            done["actual_timesteps"] != p["steps_per_model"]
            or done["stop_reason"] != "budget_complete"
        ):
            raise ValueError("Not all exact budgets complete")
    for filename in (
        "summary.json",
        "selection.json",
        "audit/summary.json",
        "pipeline_complete.json",
    ):
        hashes[str(output / filename)] = file_digest(output / filename)
    for path in (
        Path(__file__),
        ROOT / "docs/KL_80_METRIC_ADDENDUM.md",
        ROOT / "tests/test_kl_80_finalize.py",
    ):
        hashes[str(path)] = file_digest(path)
    write_new(
        output / "final_review.json",
        {
            "results": corrected,
            "acceptance": acceptance(corrected),
            "uncorrected_acceptance": summary["acceptance"],
        "correction": "Count ActionKind.FREEZE.value (toggle_freeze), not nonexistent freeze key. "
            "No changes to training, model selection, threshold, or test episodes.",
            "evidence_sha256": hashes,
            "all_training_and_evidence_complete": True,
            "replay_review_required": True,
            "automatic_delivery": False,
            "new_training_steps": 0,
            "new_evaluation_episodes": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
