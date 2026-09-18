"""Recount all KL-brake evidence and verify <=18 final-policy diagnostic replays."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path

from audit_exploration_pilot import choose_cases
from audit_midgame_confirmation import check_environment_contract, exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import read
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.historical_confirmation import fallback_counts
from sap_rl_lab.imitation_experiment import require_reloads
from sap_rl_lab.kl_long_experiment import TEST_SEED, VAL_SEED, frozen
from sap_rl_lab.ppo_guardrails import write_new


def run(root, output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selected = frozen(root)
    require_reloads(root, selected)
    if not read(root / "summary.json")["all_training_complete"]:
        raise ValueError("Not a completed fixed-budget experiment")
    output.mkdir(parents=True, exist_ok=False)
    hashes, counts, cases = {}, {}, []

    def recount(label, result, split, contract, path):
        count, seed = (100, VAL_SEED) if split == "validation" else (200, TEST_SEED)
        if set(result["families"]) != set(p["paths"][split]):
            raise ValueError("Missing families")
        counts[label] = {}
        coverage = {}
        for family, data in result["families"].items():
            pool = p["paths"][split][family]
            if data["league_sha256"] != file_digest(pool) or not data["deterministic"]:
                raise ValueError("Wrong opponent pool or stochastic evaluation")
            check_environment_contract(data["environment_contract"], contract)
            counts[label][family] = exact_family_counts(data, count, seed)
            coverage[family] = fallback_counts(
                data["episode_results"], {r["turn"] for r in read(pool)["snapshots"]}
            )
        if (
            "opponent_fallback_coverage" in result
            and result["opponent_fallback_coverage"] != coverage
        ):
            raise ValueError("Fallback coverage accounting differs")
        hashes[str(path)] = file_digest(path)

    for name, payload in p["arms"].items():
        folder = root / name
        done, manifest = read(folder / "complete.json"), read(folder / "manifest.json")
        for index, row in enumerate(done["evaluations"]):
            path = Path(row["checkpoint"]["path"]).with_suffix(".json")
            result = read(path)["evaluation"]
            recount(
                f"curve/{name}/{index}",
                result,
                "validation",
                manifest["environment_contract"],
                path,
            )
        final = read(root / f"validation-{name}-final.json")
        for case in choose_cases(final["families"]):
            cases.append(
                {
                    "name": name,
                    "path": done["last_model"],
                    "sha256": done["last_sha256"],
                    "gamma": payload["training"]["gamma"],
                    **case,
                }
            )
    for name, item in selected["models"].items():
        contract = MaskablePPO.load(item["path"], device="cpu").sap_environment_contract
        for stage, split in (("reload", "validation"), ("test", "test")):
            path = root / f"{stage}-{name}.json"
            result = read(path)
            if result["model_sha256"] != item["sha256"] or result[
                "selection_sha256"
            ] != file_digest(root / "selection.json"):
                raise ValueError("Evaluation model binding differs")
            recount(f"{stage}/{name}", result, split, contract, path)
    write_new(output / "counts.json", counts)
    write_new(
        output / "plan.json",
        {
            "cases": cases,
            "evidence_sha256": hashes,
            "selection_sha256": file_digest(root / "selection.json"),
            "selection": "Earliest loss (prefer no-purchase zero-win), forced/truncated, success "
            "per final model; deduplicate, at most 18. Diagnostic, not population sampling.",
        },
    )
    records = []
    for case in cases:
        expected, family = case["expected"], case["family"]
        label = f"{case['name']}-{family}-{expected['seed']}"
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(
                case["path"], p["paths"]["validation"][family], output / label, 1, expected["seed"]
            )
        replay = read(output / label / f"episode-{expected['seed']}.json")
        compare_replay(replay["summary"], expected)
        if file_digest(case["path"]) != case["sha256"]:
            raise ValueError("Replay weights changed")
        for step in replay["steps"]:
            if abs(step["objective_reward"] - (step["info"]["game_reward"] - 0.005)) > 1e-10:
                raise ValueError("Replay objective differs from fixed reward")
        records.append(
            {
                "name": case["name"],
                "directory": label,
                "summary": replay["summary"],
                "behavior": summarize_episode(replay, case["gamma"]),
                "critic_trained": True,
                "values_are_final_policy_reconstruction": True,
            }
        )
        print(f"Verified replay: {label}", flush=True)
    frozen(root)
    if any(file_digest(path) != digest for path, digest in hashes.items()):
        raise ValueError("Evidence changed during audit")
    write_new(
        output / "summary.json",
        {
            "all_curves_and_evaluations_recounted": True,
            "rows_recounted": sum(f["episodes"] for row in counts.values() for f in row.values()),
            "all_replays_exact": True,
            "objective_formula_verified": True,
            "replays": records,
            "replay_actions": sum(r["summary"]["actions"] for r in records),
            "training_updates": 0,
            "model_reselection": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
