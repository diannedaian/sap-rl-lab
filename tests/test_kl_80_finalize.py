"""Exercise the real enum interface rather than duplicating a guessed key."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from sap_rl_lab.actions import ActionKind
from sap_rl_lab.kl_80_experiment import SEEDS, acceptance, result_metrics

spec = importlib.util.spec_from_file_location(
    "finalize_kl_80", Path(__file__).resolve().parents[1] / "scripts/finalize_kl_80.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
corrected_metrics = review.corrected_metrics


def result(freezes=15):
    row = {
        "success": True,
        "wins": 10,
        "forced_end_turns": 0,
        "truncated": False,
        "battles": 10,
        "actions": 30 + freezes,
        "action_counts": {
            ActionKind.SWAP.value: 10,
            ActionKind.FREEZE.value: freezes,
            ActionKind.BUY_PET.value: 20,
        },
    }
    return {
        "families": {n: {"episode_results": [deepcopy(row)]} for n in ("fresh_stats", "fresh_mix")}
    }


def test_real_freeze_key_counted():
    data = result()
    assert result_metrics(data)["swap_freeze_actions_per_battle"] == 1.0
    assert corrected_metrics(data)["swap_freeze_actions_per_battle"] == 2.5


def test_corrected_gate_blocks_hidden_freeze_regression():
    original, corrected = {}, {}
    for s in SEEDS:
        for kind, freezes in (("initial", 0), ("selected", 20)):
            name = f"kl-seed{s}-{kind}"
            original[name] = result_metrics(result(freezes))
            corrected[name] = corrected_metrics(result(freezes))
            for metrics in (original[name], corrected[name]):
                metrics.update(learned_successes=800, learned_episodes=1000)
    assert acceptance(original)["eligible_after_evidence_audit"]
    assert not acceptance(corrected)["eligible_after_evidence_audit"]


def test_other_metrics_unchanged():
    original, corrected = result_metrics(result()), corrected_metrics(result())
    original.pop("swap_freeze_actions_per_battle")
    corrected.pop("swap_freeze_actions_per_battle")
    assert original == corrected


def test_unknown_count_field_fails_closed():
    data = result()
    data["families"]["fresh_stats"]["episode_results"][0]["action_counts"]["freeze"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        corrected_metrics(data)
