"""Five development-only policy inspection exercises, selected without test scores."""

import argparse
import copy
import json
import random
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from sap_rl_lab.baselines import SpendGoldPolicy
from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.opponents import SnapshotLeague
from sap_rl_lab.replay import ReplayRecorder, verify_replay


def inspect_policy(model, env):
    observation, _ = model.policy.obs_to_tensor(env._observation())
    with torch.no_grad():
        distribution = model.policy.get_distribution(observation, action_masks=env.action_masks())
        probabilities = distribution.distribution.probs.cpu().numpy()[0]
        value = float(model.policy.predict_values(observation).item())
    ranked = sorted(env.engine.legal_action_ids(), key=lambda a: (-probabilities[a], a))
    return value, [
        {"id": a, "action": env.engine.codec.describe(a), "probability": float(probabilities[a])}
        for a in ranked
    ]


def alternatives(model, saved_env, action_ids, case, repetitions=32):
    """One forced legal action, then deterministic policy, on paired future RNG seeds.

    These estimate one-step deviations, not optimal Q values. Subsequent random
    draw consumption can diverge after different choices despite matched seeds.
    """
    results = []
    for action_id in action_ids:
        envs = [copy.deepcopy(saved_env) for _ in range(repetitions)]
        observations = []
        returns = [0.0] * repetitions
        active = []
        for i, env in enumerate(envs):
            env.engine.rng = random.Random(800000 + case * 10000 + i)
            observation, reward, terminated, truncated, _ = env.step(action_id)
            observations.append(observation)
            returns[i] += reward
            if not (terminated or truncated):
                active.append(i)
        while active:
            batch = {
                key: np.stack([observations[i][key] for i in active])
                for key in observations[active[0]]
            }
            masks = np.stack([envs[i].action_masks() for i in active])
            actions, _ = model.predict(batch, action_masks=masks, deterministic=True)
            remaining = []
            for i, action in zip(active, actions):
                observations[i], reward, terminated, truncated, _ = envs[i].step(int(action))
                returns[i] += reward
                if not (terminated or truncated):
                    remaining.append(i)
            active = remaining
        results.append(
            {
                "id": action_id,
                "action": saved_env.engine.codec.describe(action_id),
                "mean_future_return": fmean(returns),
                "standard_error": stdev(returns) / repetitions**0.5,
                "success_fraction": fmean(env.engine.state.wins >= 10 for env in envs),
                "cutoff_fraction": fmean(env.engine.state.truncated for env in envs),
                "future_rng_seed_start": 800000 + case * 10000,
                "returns": returns,
            }
        )
        for env in envs:
            env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    root, output = Path(args.run_directory), Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    # Declared before test results: validation success, return, then name for ties.
    name = sorted(
        selection,
        key=lambda n: (-selection[n]["validation_success"], -selection[n]["validation_return"], n),
    )[0]
    record = selection[name]
    if file_digest(record["path"]) != record["sha256"]:
        raise ValueError("selected model no longer matches the frozen checkpoint")
    torch.set_num_threads(1)
    model = MaskablePPO.load(record["path"], device="cpu")
    model.policy.set_training_mode(False)
    cases = []
    targets = [2, 5, 10, 15, 20]
    families = ["greedy", "stats", "summon", "greedy", "stats"]
    for case, (family, target) in enumerate(zip(families, targets)):
        provider = SnapshotLeague.load(protocol["paths"]["validation"][family])
        env = SapAutoBattlerEnv(opponent_provider=provider)
        recorder = ReplayRecorder(env.engine, 700000 + case)
        states, values = [], []
        while not (env.engine.state.terminated or env.engine.state.truncated):
            value, ranked = inspect_policy(model, env)
            states.append((copy.deepcopy(env), ranked))
            values.append(value)
            recorder.step(ranked[0]["id"])
        replay = recorder.finish()
        verify_replay(replay, AutoBattler(opponent_provider=provider))
        replay.save(output / f"episode-{case + 1}.json")
        index = min(target, len(states) - 1)
        saved, ranked = states[index]
        heuristic = saved.engine.codec.encode(
            SpendGoldPolicy().choose(saved.engine, random.Random(700000 + case))
        )
        action_ids = list(dict.fromkeys([item["id"] for item in ranked[:2]] + [heuristic]))
        cases.append(
            {
                "number": case + 1,
                "family": family,
                "episode_seed": 700000 + case,
                "decision_index": index,
                "state": saved.engine.state.to_dict(),
                "value_prediction": values[index],
                "actions": ranked,
                "actual_remaining_return": sum(step.reward for step in replay.steps[index:]),
                "realized_advantage_example": sum(step.reward for step in replay.steps[index:])
                - values[index],
                "episode_values": values,
                "episode_rewards": [s.reward for s in replay.steps],
                "alternatives": alternatives(model, saved, action_ids, case),
            }
        )
        print(f"EXERCISE {case + 1}: {family}, decision {index}, {ranked[0]['action']}", flush=True)
    payload = {
        "policy": name,
        "selection_rule": "validation only; test scores not read",
        "model_sha256": record["sha256"],
        "cases": cases,
    }
    (output / "exercises.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Five shop decisions to work through",
        "",
        f"Policy: `{name}`, selected using validation only. These five fixed development "
        "seeds are not benchmark results.",
        "",
        "Try choosing before opening each answer. Slot numbers start at zero, matching "
        "the action traces. Team slot 0 fights first. `merge:shop,team` buys a copy from "
        "the shop into an existing pet; `buy_food:shop,team` feeds that team slot.",
        "",
        "An action probability is the policy's preference, not the chance of winning. "
        "The critic predicts future net reward, not win probability. Both may be wrong.",
        "The family label below describes the scenario's opponent pool; it is not an "
        "input given to the policy, and the next opponent team is not revealed.",
        "",
        "Rules reminder: buying a pet appends it to the back; rolling costs one gold; "
        "swapping and freezing cost no gold but consume decisions. A turn allows 30 "
        "decisions. Shop moves earn zero immediate reward, battles earn +1/0/-1 for "
        "win/draw/loss, and reaching the shop limit cuts the episode short with -1. "
        "Temporary buffs add to the displayed base stats for the next battle only.",
        "",
        "These examples use the frozen Round 3 simulator, not verified live-game "
        "parity. In particular, Fish's level-up indexing needs a separate correction "
        "and parity test; see [the Round 3 report](../ROUND3.md).",
        "",
    ]
    for case in cases:
        state = case["state"]
        lines += [
            f"## Situation {case['number']}",
            "",
            f"Turn {state['turn']} · gold {state['gold']} · wins {state['wins']} · "
            f"lives {state['lives']} · shop decisions used {state['actions_this_turn']}/30 "
            f"· opponent family {case['family']}",
            "",
            "| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |",
            "|---|---|---|---|---|---|",
        ]
        for i, pet in enumerate(state["team"]):
            lines.append(
                f"| {i} | {pet['spec_id']} | {pet['attack']} / {pet['health']} | "
                f"{pet['experience']} | {pet['perk'] or '—'} | "
                f"+{pet['temporary_attack']} / +{pet['temporary_health']} |"
            )
        lines += ["", "| Shop slot | Item | Cost | Frozen |", "|---|---|---|---|"]
        for i, item in enumerate(state["shop"]):
            if item:
                lines.append(f"| {i} | {item['item_id']} | {item['cost']} | {item['frozen']} |")
        lines += [
            "",
            "Before revealing: What would you do? What later battle benefit do you "
            "expect? Which alternative would you test?",
            "",
            "<details>",
            "<summary>Policy answer and evidence</summary>",
            "",
            "| Action | Policy probability |",
            "|---|---:|",
        ]
        for action in case["actions"][:5]:
            lines.append(f"| `{action['action']}` | {action['probability']:.1%} |")
        lines += [
            "",
            f"Critic value: **{case['value_prediction']:.2f}**. The recorded "
            f"continuation earned **{case['actual_remaining_return']:.1f}**. Their "
            f"difference is **{case['realized_advantage_example']:+.2f}**.",
            "",
            "That difference illustrates a full-return advantage estimate. Actual PPO "
            "training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic "
            "policy—not exactly this deterministic full-episode calculation.",
            "",
            "For each action below, force that one move and then follow the policy over "
            "32 sampled futures. These are noisy one-step-deviation estimates, not "
            "optimal action values. Matched initial random seeds can diverge after "
            "different actions consume randomness.",
            "",
            "| Forced first action | Mean future reward | Standard error |",
            "|---|---:|---:|",
        ]
        for alternative in case["alternatives"]:
            lines.append(
                f"| `{alternative['action']}` | "
                f"{alternative['mean_future_return']:.2f} | "
                f"{alternative['standard_error']:.2f} |"
            )
        lines += [
            "",
            "Does the most probable action also look best in these rollouts? "
            "What would additional samples clarify?",
            "",
            f"[Complete episode replay](episode-{case['number']}.json)",
            "",
            "</details>",
            "",
        ]
    lines += [
        "## How to interpret your answers",
        "",
        "- A forced move is followed by the existing policy, which can undo that "
        "move. Similar returns do not prove that positioning never matters.",
        "- A high action probability can coexist with similar or better sampled "
        "returns for another action. PPO has learned a preference, not a proof of optimality.",
        "- The uncertainty shown is a standard error for each mean, not a 95% "
        "interval for the difference. With only 32 futures, small gaps need more evidence.",
        "- Try explaining one zero-reward shop move in terms of later rewards, "
        "then explain why the critic can be wrong about its continuation.",
        "",
    ]
    (output / "EXERCISES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
