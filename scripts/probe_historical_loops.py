"""Bounded frozen-policy diagnostic interventions, not training or deployment."""

import argparse
import copy
from pathlib import Path

import torch
from inspect_replay_alternatives import restore_prefix
from round3_exercises import inspect_policy
from sb3_contrib import MaskablePPO

from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.evaluation import compact_evaluation, evaluate_policy, policy_environment_options
from sap_rl_lab.expanded_confirmation import read, save_new
from sap_rl_lab.historical_confirmation import check_tests, checked


def strip_counter(state):
    return {k: v for k, v in state.items() if k != "actions_this_turn"}


def run(root, diagnosis):
    from sap_rl_lab.opponents import SnapshotLeague

    protocol = checked(root)
    check_tests(root)
    torch.set_num_threads(1)
    name = "historical_mix-seed2027"
    cfg = protocol["arms"][name]
    model_path = read(root / "selection.json")["models"][name]["path"]
    model = MaskablePPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)
    cases = read(diagnosis / "replay_audit.json")["cases"]
    plan = read(diagnosis / "plan.json")["cases"]
    probes = []
    kinds = set()
    for case in cases:
        if case["group"] != "forced" or case["model"] != name:
            continue
        record = read(diagnosis / case["directory"] / f"episode-{case['summary']['seed']}.json")
        steps = record["steps"]
        forced_turns = {s["turn"] for s in case["forced_turns"]}
        for i in range(len(steps) - 2):
            action = steps[i]["action"]
            kind = action.split(":")[0]
            if kind not in {"toggle_freeze", "swap_adjacent"} or kind in kinds:
                continue
            if (
                action != steps[i + 1]["action"]
                or steps[i]["state"]["turn"] not in forced_turns
                or steps[i]["state"]["actions_this_turn"] > 26
                or strip_counter(steps[i]["state"]) != strip_counter(steps[i + 2]["state"])
            ):
                continue
            original = next(
                c
                for c in plan
                if c["model"] == name
                and c["family"] == case["family"]
                and c["expected"]["seed"] == case["summary"]["seed"]
            )
            env = SapAutoBattlerEnv(
                opponent_provider=SnapshotLeague.load(original["league"]),
                **policy_environment_options(model),
                action_cost=cfg["action_cost"],
            )
            restore_prefix(env, record, case["summary"]["seed"], i)
            value, probabilities = inspect_policy(model, env)
            a, b = copy.deepcopy(env), copy.deepcopy(env)
            before = b.engine.rng.getstate()
            _, r0, t0, u0, _ = b.step(steps[i]["action_id"])
            _, r1, t1, u1, _ = b.step(steps[i]["action_id"])
            assert not (t0 or u0 or t1 or u1)
            assert before == b.engine.rng.getstate()
            assert strip_counter(a.engine.state.to_dict()) == strip_counter(
                b.engine.state.to_dict()
            )
            oa, ra, ta, ua, ia = a.step(0)
            ob, rb, tb, ub, ib = b.step(0)
            assert a.engine.state.to_dict() == b.engine.state.to_dict()
            assert a.engine.rng.getstate() == b.engine.rng.getstate()
            assert all((oa[k] == ob[k]).all() for k in oa)
            assert (ta, ua) == (tb, ub)
            probes.append(
                {
                    "directory": case["directory"],
                    "step": i,
                    "state": steps[i]["state"],
                    "cycle_action": action,
                    "value": value,
                    "probabilities": probabilities,
                    "end_now_reward": ra,
                    "two_reversible_actions_then_end_reward": r0 + r1 + rb,
                    "difference": r0 + r1 + rb - ra,
                    "game_rewards": [ia["game_reward"], ib["game_reward"]],
                    "post_battle_state_observation_and_rng_identical": True,
                }
            )
            for instance in (env, a, b):
                instance.close()
            kinds.add(kind)
            break
        if len(kinds) == 2:
            break
    if len(probes) != 2:
        raise ValueError("Could not find both bounded reversible-action probes")
    save_new(
        diagnosis / "reversible_cycle_probes.json",
        {
            "probes": probes,
            "limits": "Exactly two post-test diagnostic states. Two reversible actions "
            "then END versus END now give identical subsequent game/observation/RNG, "
            "but cost 0.01 extra. This proves local wasted reward, not the historical "
            "gradient cause of learning the loop. Parameters and engine untouched.",
        },
    )
    selected = read(root / "selection.json")["models"][name]["selected"]
    deterministic = read(root / name / selected["evaluation_file"])
    outcomes = []
    # Fixed existing validation families only. Do not add another test comparison.
    for family in ("control1907", "expanded_tempo"):
        for sampling_seed in (441, 442, 443):
            torch.manual_seed(sampling_seed)
            result = evaluate_policy(
                model,
                episodes=200,
                seed=cfg["validation_seed"],
                opponent_league=cfg["validation_leagues"][family],
                deterministic=False,
            )
            save_new(diagnosis / f"sampled-{family}-{sampling_seed}.json", result)
            outcomes.append(
                {
                    "family": family,
                    "sampling_seed": sampling_seed,
                    "result": compact_evaluation(result),
                }
            )
            print(
                f"Sampling probe {family}/{sampling_seed}: "
                f"success={result['success_rate']:.3f}, "
                f"forcing={result['forced_episode_rate']:.3f}",
                flush=True,
            )
    save_new(
        diagnosis / "sampling_probe_summary.json",
        {
            "model": name,
            "deterministic": {
                f: compact_evaluation(deterministic["families"][f])
                for f in ("control1907", "expanded_tempo")
            },
            "sampled": outcomes,
            "limits": "Same weights, 200 existing validation seeds per family, three "
            "action-sampling seeds. Diagnostic only: changing action selection is not "
            "a certified fix and changes playing strength too. No training/reselection.",
        },
    )
    checked(root)
    check_tests(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--diagnosis", required=True)
    args = parser.parse_args()
    run(Path(args.root).resolve(), Path(args.diagnosis).resolve())
