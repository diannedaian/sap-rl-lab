"""Exact postprocessing boundaries, independent of training and model selection."""

import importlib.util
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from sap_rl_lab.expanded_gate import screen

spec = importlib.util.spec_from_file_location(
    "exact_confirmation_counts",
    Path(__file__).resolve().parents[1] / "scripts/exact_confirmation_counts.py",
)
exact_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact_module)
exact_screen = exact_module.exact_screen


def boundary_pairs(candidate_counts, control_counts):
    families = ("expanded_stats", "expanded_summon", "expanded_tempo")

    def suite(counts):
        return {
            "families": {
                family: {
                    "league_sha256": "boundary-fixture-pool",
                    "environment_contract": {
                        "game_config": {"shop_action_limit_mode": "force_battle"}
                    },
                    "episode_results": [
                        {
                            "seed": i,
                            "success": i < count,
                            "truncated": False,
                            "forced_end_turns": 0,
                            "battles": 10,
                        }
                        for i in range(1000)
                    ],
                }
                for family, count in zip(families, counts)
            }
        }

    pair = {"candidate": suite(candidate_counts), "control": suite(control_counts)}
    return [{**deepcopy(pair), "seed": seed} for seed in (1811, 1907, 2027)]


def test_exact_ninety_percent_macro_must_pass():
    counts = (813, 938, 949)
    assert Fraction(sum(counts), 3000) == Fraction(9, 10)
    result = screen(boundary_pairs(counts, counts))
    assert result["numerical_screen_passed"]


def test_exact_one_percentage_point_loss_must_pass():
    candidate, control = (900, 902, 900), (910, 912, 910)
    assert Fraction(sum(control) - sum(candidate), 3000) == Fraction(1, 100)
    result = screen(boundary_pairs(candidate, control))
    assert result["numerical_screen_passed"]


@pytest.mark.parametrize(
    "candidate,control,passed",
    [
        ((813, 938, 949), (813, 938, 949), True),
        ((900, 902, 900), (910, 912, 910), True),
        ((900, 902, 900), (910, 912, 911), False),
        ((812, 938, 949), (812, 938, 949), False),
    ],
)
def test_exact_reference_preserves_inclusive_boundaries_without_epsilon(candidate, control, passed):
    report = exact_screen(boundary_pairs(candidate, control))
    assert report["numerical_screen_passed"] is passed
    assert not report["goal_complete"]


@pytest.mark.parametrize("flag", ["forced_end_turns", "truncated"])
@pytest.mark.parametrize("count,passed", [(9, True), (10, False)])
def test_exact_reference_keeps_strict_reliability_boundary(flag, count, passed):
    pairs = boundary_pairs((900, 902, 900), (900, 902, 900))
    rows = pairs[0]["candidate"]["families"]["expanded_stats"]["episode_results"]
    for row in rows[:count]:
        row[flag] = 1 if flag == "forced_end_turns" else True
    assert exact_screen(pairs)["numerical_screen_passed"] is passed


def test_exact_reference_keeps_structural_rejections():
    pairs = boundary_pairs((900, 902, 900), (900, 902, 900))
    pairs[1]["seed"] = pairs[0]["seed"]
    with pytest.raises(ValueError, match="distinct"):
        exact_screen(pairs)


def test_exact_reference_keeps_macro_not_episode_weighted_success():
    pairs = boundary_pairs((700, 1000, 1000), (700, 1000, 1000))
    for pair in pairs:
        for role in ("candidate", "control"):
            rows = pair[role]["families"]["expanded_stats"]["episode_results"]
            rows.extend([{**row, "seed": row["seed"] + 1000} for row in rows[:1000]])
            rows.extend([{**row, "seed": row["seed"] + 2000} for row in rows[:1000]])
    report = exact_screen(pairs)
    # Macro (70% + 100% + 100%)/3 = 90%; pooled episode rate would be only 82%.
    assert report["numerical_screen_passed"]
    assert all(row["exact_macro_success"] == "9/10" for row in report["per_seed"])
