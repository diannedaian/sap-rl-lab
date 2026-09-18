"""Inspect existing expanded diagnostic replays; never train or generate episodes."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from sap_rl_lab.evaluation import file_digest


def summarize_episode(record, gamma):
    if not 0 <= gamma <= 1:
        raise ValueError("Discount must be between zero and one")
    steps, summary = record["steps"], record["summary"]
    if not steps or not (steps[-1]["terminated"] or steps[-1]["truncated"]):
        raise ValueError("Requires a completed recorded episode")
    if len(steps) != summary["actions"]:
        raise ValueError("Step count disagrees with episode summary")
    complete_return = steps[-1]["terminated"] and not steps[-1]["truncated"]
    remaining, total = [], 0.0
    for step in reversed(steps):
        reward = step["objective_reward"]
        if not math.isfinite(reward) or not math.isfinite(step["value"]):
            raise ValueError("Non-finite diagnostic value/reward")
        total = reward + gamma * total
        remaining.append(total)
    remaining.reverse()
    seen, species, purchases, sales = set(), set(), Counter(), Counter()
    identical_swaps, consecutive_swaps, repeat_states, examples = [], [], [], []
    for index, step in enumerate(steps):
        state = step["state"]
        key = json.dumps(
            {k: v for k, v in state.items() if k != "actions_this_turn"}, sort_keys=True
        )
        if key in seen:
            repeat_states.append(index)
        seen.add(key)
        species.update(pet["spec_id"] for pet in state["team"])
        kind, _, suffix = step["action"].partition(":")
        slots = [int(part) for part in suffix.split(",")] if suffix else []
        if kind == "swap_adjacent":
            a, b = slots
            if state["team"][a] == state["team"][b]:
                identical_swaps.append(index)
            if (
                index
                and steps[index - 1]["action"] == step["action"]
                and steps[index - 1]["state"]["turn"] == state["turn"]
            ):
                consecutive_swaps.append(index)
        elif kind in {"buy_pet", "merge"}:
            item = state["shop"][slots[0]]
            if not item or item["kind"] != "pet":
                raise ValueError("Pet purchase does not identify a pet in the recorded shop")
            purchases[item["item_id"]] += 1
        elif kind == "sell":
            sales[state["team"][slots[0]]["spec_id"]] += 1
        if (
            index in identical_swaps
            or step["info"].get("forced_end_turn")
            or index == len(steps) - 1
        ):
            examples.append(
                {
                    "step": index,
                    "turn": state["turn"],
                    "gold": state["gold"],
                    "action": step["action"],
                    "value": step["value"],
                    "top_actions": step["top_actions"],
                    "observed_discounted_remaining_reward": remaining[index],
                    "complete_episode_return": bool(complete_return),
                    "realized_return_minus_value": (
                        remaining[index] - step["value"] if complete_return else None
                    ),
                    "forced_end_turn": bool(step["info"].get("forced_end_turn")),
                    "terminated": step["terminated"],
                    "truncated": step["truncated"],
                }
            )
    return {
        "seed": summary["seed"],
        "success": summary["success"],
        "wins": summary["wins"],
        "actions": len(steps),
        "identical_pet_swaps": identical_swaps,
        "consecutive_same_slot_swaps": consecutive_swaps,
        "repeated_state_steps": repeat_states,
        "species_seen_in_team": sorted(species),
        "pet_purchase_counts_including_merges": dict(purchases),
        "pet_sale_counts": dict(sales),
        "inspection_points": examples,
    }


def audit(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    metadata_path = directory / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    model = Path(metadata["model"])
    if file_digest(model) != metadata["model_sha256"]:
        raise ValueError("Recorded checkpoint changed")
    manifest = model.parent / "run_manifest.json"
    gamma = json.loads(manifest.read_text())["config"]["gamma"]
    rows, inputs, seeds = [], {}, set()
    for expected in metadata["episodes"]:
        seed = expected["seed"]
        if type(seed) is not int or seed in seeds:
            raise ValueError("Requires unique integer episode seeds")
        seeds.add(seed)
        path = directory / f"episode-{seed}.json"
        record = json.loads(path.read_text())
        if record["summary"] != expected:
            raise ValueError("Episode metadata changed")
        rows.append(summarize_episode(record, gamma))
        inputs[path.name] = file_digest(path)
    result = {
        "gamma": gamma,
        "model_sha256": metadata["model_sha256"],
        "input_summary_sha256": file_digest(metadata_path),
        "manifest_sha256": file_digest(manifest),
        "episode_sha256": inputs,
        "script_sha256": file_digest(__file__),
        "episodes": rows,
        "limits": "Read-only analysis of previously generated trajectories, not a new score. "
        "Repeated states omit only the shop-action counter. Consecutive swaps may overlap "
        "and are not proven harmful. Species presence/purchases do not establish mastery. "
        "One realized return is not the expected value; truncated returns have no value-error "
        "claim or invented bootstrap. Values belong to the saved policy, "
        "not historical PPO updates.",
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                "identical_pet_swaps": sum(len(row["identical_pet_swaps"]) for row in rows),
                "pet_sales": sum(sum(row["pet_sale_counts"].values()) for row in rows),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.directory, args.output)
