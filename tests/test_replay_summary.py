"""Postprocessing facts must not overstate causality or truncated returns."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "summarize_policy_replays",
    Path(__file__).resolve().parents[1] / "scripts/summarize_policy_replays.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def replay():
    pet = {"spec_id": "fish", "attack": 2, "health": 3, "experience": 1}
    state = {"turn": 1, "gold": 0, "actions_this_turn": 0, "team": [pet, dict(pet)], "shop": []}
    steps = []
    for index, action in enumerate(("swap_adjacent:0,1", "swap_adjacent:0,1", "end_turn")):
        steps.append(
            {
                "state": {**deepcopy(state), "actions_this_turn": index},
                "action": action,
                "value": 2.0,
                "top_actions": [{"action": action, "probability": 0.9}],
                "objective_reward": (-0.005 if index < 2 else -1.005),
                "info": {},
                "terminated": index == 2,
                "truncated": False,
            }
        )
    return {"summary": {"seed": 12, "actions": 3, "success": False, "wins": 0}, "steps": steps}


def test_identical_swaps_and_repeats_exclude_only_action_counter():
    result = module.summarize_episode(replay(), 1)
    assert result["identical_pet_swaps"] == [0, 1]
    assert result["consecutive_same_slot_swaps"] == [1]
    assert result["repeated_state_steps"] == [1, 2]
    assert result["inspection_points"][0]["observed_discounted_remaining_reward"] == pytest.approx(
        -1.015
    )


def test_same_species_different_stats_are_not_identical_pets():
    record = replay()
    for step in record["steps"]:
        step["state"]["team"][1]["attack"] = 5
    assert module.summarize_episode(record, 1)["identical_pet_swaps"] == []


def test_truncation_does_not_claim_complete_value_error_or_bootstrap():
    record = replay()
    record["steps"][-1].update(terminated=False, truncated=True)
    points = module.summarize_episode(record, 0.9)["inspection_points"]
    assert all(not point["complete_episode_return"] for point in points)
    assert all(point["realized_return_minus_value"] is None for point in points)
    assert points[0]["observed_discounted_remaining_reward"] == pytest.approx(
        -0.005 - 0.9 * 0.005 - 0.9**2 * 1.005
    )


def test_forced_end_is_inspected_but_not_mislabeled_external_truncation():
    record = replay()
    record["steps"][1]["info"]["forced_end_turn"] = True
    point = module.summarize_episode(record, 1)["inspection_points"][1]
    assert point["forced_end_turn"] and not point["truncated"]
    assert point["complete_episode_return"]


def test_incomplete_or_miscounted_episodes_are_rejected():
    record = replay()
    record["steps"][-1]["terminated"] = False
    with pytest.raises(ValueError, match="completed"):
        module.summarize_episode(record, 1)
    record = replay()
    record["summary"]["actions"] += 1
    with pytest.raises(ValueError, match="Step count"):
        module.summarize_episode(record, 1)


def test_purchases_merges_and_sales_are_counted_from_recorded_slots():
    record = replay()
    for step, action in zip(record["steps"], ("buy_pet:0", "merge:0,0", "sell:0")):
        step["action"] = action
        step["state"]["shop"] = [{"kind": "pet", "item_id": "dodo"}]
    result = module.summarize_episode(record, 1)
    assert result["pet_purchase_counts_including_merges"] == {"dodo": 2}
    assert result["pet_sale_counts"] == {"fish": 1}
    assert result["species_seen_in_team"] == ["fish"]
