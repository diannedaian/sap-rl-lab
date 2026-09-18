"""Bounded legal-action branch audit on reachable curriculum states, not training."""

import argparse
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from sap_rl_lab.baselines import scripted_policy
from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig
from sap_rl_lab.engine import AutoBattler
from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.events import attack, health
from sap_rl_lab.experiments import save_json
from sap_rl_lab.round3 import source_archive


def check_branches(engine):
    before = (engine.state.to_dict(), engine.rng.getstate(), deepcopy(engine.last_battle))
    legal = engine.legal_action_ids()
    assert legal and len(legal) == len(set(legal))
    assert set(legal) == {i for i, allowed in enumerate(engine.action_mask()) if allowed}
    results, counts = [], Counter()
    for action in legal:
        branches = [deepcopy(engine), deepcopy(engine)]
        transitions = [branch.step_id(action) for branch in branches]
        states = [branch.state.to_dict() for branch in branches]
        assert transitions[0] == transitions[1] and states[0] == states[1]
        assert branches[0].rng.getstate() == branches[1].rng.getstate()
        state = branches[0].state
        assert len(state.team) <= 5 and len(state.shop) == 9
        assert state.gold >= 0 and branches[0].unlocked_tier <= 2
        assert all(0 < health(p) <= 50 and 0 <= attack(p) <= 50 for p in state.team)
        counts[engine.codec.decode(action).kind.value] += 1
        results.append((action, states[0]))
    assert before == (engine.state.to_dict(), engine.rng.getstate(), engine.last_battle)
    return counts, results


def audit(output, states_per_family):
    if not 1 <= states_per_family <= 100:
        raise ValueError("Requires 1–100 sampled states per family")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    sources = source_archive(output)
    results_digest = hashlib.sha256()
    counts, sampled, species = Counter(), Counter(), Counter()
    context = {}
    try:
        for index, family in enumerate(
            ("random", "expanded_stats", "expanded_summon", "expanded_tempo")
        ):
            policy = scripted_policy(family)
            for episode in range(20):
                seed = 4_300_000 + index * 1000 + episode
                engine = AutoBattler(catalog, GameConfig.turtle_curriculum())
                engine.reset(seed=seed)
                rng = random.Random(seed + 50_000)
                step = 0
                while not (engine.state.terminated or engine.state.truncated):
                    if step % 5 == 0 and sampled[family] < states_per_family:
                        context = {
                            "family": family,
                            "seed": seed,
                            "step": step,
                            "state": engine.state.to_dict(),
                            "rng": engine.rng.getstate(),
                        }
                        branch_counts, states = check_branches(engine)
                        counts.update(branch_counts)
                        species.update(p.spec_id for p in engine.state.team)
                        sampled[family] += 1
                        results_digest.update(json.dumps(states, sort_keys=True).encode())
                    if sampled[family] >= states_per_family:
                        break
                    action = engine.codec.encode(policy.choose(engine, rng))
                    engine.step_id(action)
                    step += 1
                    assert step <= 900
                if sampled[family] >= states_per_family:
                    break
            assert sampled[family] == states_per_family
    except Exception as error:
        save_json(output / "failed.json", {"error": repr(error), "context": context})
        raise
    summary = {
        "catalog_sha256": catalog_digest(catalog),
        "source_files_sha256": sources,
        "audit_script_sha256": file_digest(__file__),
        "sampled_states": dict(sampled),
        "legal_branches_twice_checked": sum(counts.values()),
        "action_counts": dict(counts),
        "observed_species": dict(species),
        "result_states_sha256": results_digest.hexdigest(),
        "originals_unchanged": True,
        "limits": "Reachable heuristic/random shop states; every advertised legal action is "
        "executed twice on independent copies. This checks mask soundness, invariants and "
        "determinism, not completeness of the official action set or full client parity. "
        "No training, model selection or held-out evaluation.",
    }
    save_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("sampled_states", "legal_branches_twice_checked")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states-per-family", type=int, default=50)
    args = parser.parse_args()
    audit(args.output, args.states_per_family)
