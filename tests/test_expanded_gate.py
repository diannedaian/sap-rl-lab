from copy import deepcopy

import pytest

from sap_rl_lab.expanded_gate import FAMILIES, screen


def pairs():
    family = {
        "league_sha256": "frozen-pool",
        "environment_contract": {"game_config": {"shop_action_limit_mode": "force_battle"}},
        "episode_results": [
            {
                "seed": i,
                "success": i < 900,
                "truncated": False,
                "forced_end_turns": 0,
                "battles": 10,
            }
            for i in range(1000)
        ],
    }
    suite = {"families": {name: deepcopy(family) for name in FAMILIES}}
    return [
        {"seed": s, "candidate": deepcopy(suite), "control": deepcopy(suite)} for s in (11, 23, 37)
    ]


def test_passing_numerics_does_not_certify_goal_or_parity():
    result = screen(pairs())
    assert result["numerical_screen_passed"]
    assert not result["goal_complete"]


@pytest.mark.parametrize("flag", ["forced_end_turns", "truncated"])
def test_one_family_one_seed_at_one_percent_fails_even_if_macro_is_small(flag):
    data = pairs()
    rows = data[0]["candidate"]["families"]["expanded_stats"]["episode_results"]
    for row in rows[:10]:
        row[flag] = 1 if flag == "forced_end_turns" else True
    assert not screen(data)["numerical_screen_passed"]


def test_nine_per_thousand_forced_episodes_passes_and_battle_rate_is_reported():
    data = pairs()
    rows = data[0]["candidate"]["families"]["expanded_stats"]["episode_results"]
    for row in rows[:9]:
        row["forced_end_turns"] = 2
    result = screen(data)
    assert result["numerical_screen_passed"]
    family = result["per_seed"][0]["families"]["expanded_stats"]
    assert family["forced_episode_rate"] == 0.009
    assert family["forced_battle_rate"] == 0.0018


def test_recompute_from_rows_ignores_falsely_perfect_headline():
    data = pairs()
    for family in data[0]["candidate"]["families"].values():
        family["success_rate"] = 1.0
        family["episode_results"][0]["success"] = False
    assert not screen(data)["numerical_screen_passed"]


def test_more_than_one_point_drop_vs_paired_control_fails():
    data = pairs()
    for family in data[0]["control"]["families"].values():
        for row in family["episode_results"][900:911]:
            row["success"] = True
    assert not screen(data)["numerical_screen_passed"]


@pytest.mark.parametrize(
    "problem", ["duplicate_seed", "too_small", "pool", "pairing", "contract", "count"]
)
def test_incomplete_or_unpaired_evidence_rejected(problem):
    data = pairs()
    family = data[0]["candidate"]["families"]["expanded_stats"]
    if problem == "duplicate_seed":
        data[0]["seed"] = data[1]["seed"]
    elif problem == "too_small":
        family["episode_results"].pop()
    elif problem == "pool":
        family["league_sha256"] = "other"
    elif problem == "pairing":
        family["episode_results"][0]["seed"] = 42
    elif problem == "contract":
        family["environment_contract"]["game_config"]["shop_action_limit_mode"] = "truncate"
    else:
        family["episode_results"][0]["forced_end_turns"] = 11
    with pytest.raises(ValueError):
        screen(data)
