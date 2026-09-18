"""Recount every row and reconstruct <=18 selected/final-policy diagnostics."""

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
from sap_rl_lab.kl_80_experiment import (
    TEST_EPISODES,
    TEST_SEED,
    VAL_EPISODES,
    VAL_SEED,
    acceptance,
    frozen,
    result_metrics,
)
from sap_rl_lab.ppo_guardrails import write_new


def run(root, output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, selected = frozen(root)
    require_reloads(root, selected)
    summary = read(root / "summary.json")
    if not summary["all_training_complete"]:
        raise ValueError("Incomplete training")
    output.mkdir(parents=True, exist_ok=False)
    hashes, counts, cases, seen = {}, {}, [], set()

    def recount(label, result, split, contract, path):
        count, seed = (
            (VAL_EPISODES, VAL_SEED) if split == "validation" else (TEST_EPISODES, TEST_SEED)
        )
        if set(result["families"]) != set(p["paths"][split]):
            raise ValueError("Missing families")
        counts[label], coverage = {}, {}
        for family, data in result["families"].items():
            pool = p["paths"][split][family]
            if data["league_sha256"] != file_digest(pool) or not data["deterministic"]:
                raise ValueError("Wrong pool or stochastic evaluation")
            check_environment_contract(data["environment_contract"], contract)
            counts[label][family] = exact_family_counts(data, count, seed)
            coverage[family] = fallback_counts(
                data["episode_results"], {r["turn"] for r in read(pool)["snapshots"]}
            )
        if (
            "opponent_fallback_coverage" in result
            and result["opponent_fallback_coverage"] != coverage
        ):
            raise ValueError("Fallback accounting differs")
        hashes[str(path)] = file_digest(path)

    for name in p["arms"]:
        folder = root / name
        done, manifest = read(folder / "complete.json"), read(folder / "manifest.json")
        for index, row in enumerate(done["evaluations"]):
            path = Path(row["checkpoint"]["path"]).with_suffix(".json")
            recount(
                f"curve/{name}/{index}",
                read(path)["evaluation"],
                "validation",
                manifest["environment_contract"],
                path,
            )
    recomputed = {}
    for name, item in selected["models"].items():
        model = MaskablePPO.load(item["path"], device="cpu")
        for stage, split in (("reload", "validation"), ("test", "test")):
            path = root / f"{stage}-{name}.json"
            result = read(path)
            if result["model_sha256"] != item["sha256"] or result[
                "selection_sha256"
            ] != file_digest(root / "selection.json"):
                raise ValueError("Model/selection binding differs")
            recount(f"{stage}/{name}", result, split, model.sap_environment_contract, path)
            if stage == "test":
                recomputed[name] = result_metrics(result)
        if not name.endswith(("-selected", "-final")):
            continue
        for case in choose_cases(read(item["validation_path"])["families"]):
            key = (item["sha256"], case["family"], case["expected"]["seed"])
            if key not in seen:
                seen.add(key)
                cases.append(
                    {
                        "name": name,
                        "path": item["path"],
                        "sha256": item["sha256"],
                        "gamma": model.gamma,
                        **case,
                    }
                )
    if recomputed != summary["results"] or acceptance(recomputed) != summary["acceptance"]:
        raise ValueError("Summary/80% decision cannot be reproduced")
    if len(cases) > 18:
        raise ValueError("Replay budget exceeded")
    # Every frozen-model reload was written before any held-out test began.
    if max((root / f"reload-{n}.json").stat().st_mtime_ns for n in selected["models"]) >= min(
        (root / f"test-{n}-started.json").stat().st_mtime_ns for n in selected["models"]
    ):
        raise ValueError("Testing began before all reloads were complete")
    write_new(output / "counts.json", counts)
    write_new(
        output / "plan.json",
        {
            "cases": cases,
            "evidence_sha256": hashes,
            "selection": "Earliest failure, forced/truncated, success per final/selected; <=18, "
            "deduplicated by model file/family/seed. Diagnostic sample, not a loop-rate estimate.",
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
                raise ValueError("Replay objective differs")
        records.append(
            {
                "name": case["name"],
                "directory": label,
                "summary": replay["summary"],
                "behavior": summarize_episode(replay, case["gamma"]),
                "values_are_final_policy_reconstruction": True,
            }
        )
        print(f"Verified replay: {label}", flush=True)
    frozen(root)
    if any(file_digest(path) != digest for path, digest in hashes.items()):
        raise ValueError("Evidence changed")
    write_new(
        output / "summary.json",
        {
            "all_curves_and_evaluations_recounted": True,
            "rows_recounted": sum(f["episodes"] for row in counts.values() for f in row.values()),
            "all_replays_exact": True,
            "objective_formula_verified": True,
            "all_reloads_precede_all_tests": True,
            "acceptance_recomputed": True,
            "replays": records,
            "replay_actions": sum(r["summary"]["actions"] for r in records),
            "training_updates": 0,
            "model_reselection": False,
        },
    )
