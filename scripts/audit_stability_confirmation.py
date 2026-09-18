"""Recount all ten-model stability results and bounded selected/final replays.

Run only after all held-out testing is complete. Never trains or reselects.
"""

import argparse
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from math import isclose
from pathlib import Path
from statistics import fmean

from audit_exploration_pilot import choose_cases
from audit_midgame_confirmation import check_environment_contract, exact_family_counts
from inspect_confirmation_delivery import compare_replay
from summarize_policy_replays import summarize_episode

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import compare_reload, read, save_new
from sap_rl_lab.expanded_diagnostics import inspect
from sap_rl_lab.exploration_pilot import metrics
from sap_rl_lab.stability_confirmation import (
    ARMS,
    CANDIDATE_STEPS,
    GENERATOR_STEPS,
    SEEDS,
    generator_quality,
    require_all_reloads,
    verify,
)
from sap_rl_lab.training import TrainingConfig, midgame_checkpoint_score


def check_training_configuration(manifest, expected, model):
    normalized = json.loads(json.dumps(asdict(TrainingConfig(**expected))))
    if manifest["config"] != normalized:
        raise ValueError("Actual training configuration differs from frozen design")
    if manifest["initialization"] != "from_scratch" or manifest["initial_model_sha256"] is not None:
        raise ValueError("Confirmation must start from scratch")
    mappings = {
        "ent_coef": "entropy_coefficient",
        "seed": "seed",
        "n_envs": "environments",
        "n_steps": "rollout_steps",
        "batch_size": "batch_size",
        "gamma": "gamma",
        "gae_lambda": "gae_lambda",
        "learning_rate": "learning_rate",
    }
    actual = {attribute: getattr(model, attribute) for attribute in mappings}
    for attribute, setting in mappings.items():
        if actual[attribute] != normalized[setting]:
            raise ValueError(f"Saved PPO setting differs from design: {attribute}")
    if manifest["training_mixture_sha256"] != {
        path: file_digest(path) for path in normalized["opponent_leagues"]
    } or manifest["validation_suite_sha256"] != {
        name: file_digest(path) for name, path in normalized["validation_leagues"].items()
    }:
        raise ValueError("Training manifest pool digests differ")
    return actual


def recount_suite(data, paths, episodes, seed, contract):
    if set(data["families"]) != set(paths):
        raise ValueError("Missing evaluation family")
    counts = {}
    for family, result in data["families"].items():
        if file_digest(paths[family]) != result["league_sha256"]:
            raise ValueError("Evaluation pool changed")
        if not result["deterministic"]:
            raise ValueError("Unexpected stochastic evaluation")
        check_environment_contract(result["environment_contract"], contract)
        counts[family] = exact_family_counts(result, episodes, seed)
        counts[family]["no_purchase_zero_win_losses"] = sum(
            not r["success"] and r["wins"] == 0 and not r["action_counts"].get("buy_pet", 0)
            for r in result["episode_results"]
        )
    for metric in (
        "success_rate",
        "mean_return",
        "mean_wins",
        "truncation_rate",
        "mean_episode_actions",
        "forced_episode_rate",
        "success_without_forcing_rate",
    ):
        expected = fmean(f[metric] for f in counts.values())
        if not isclose(data[metric], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Suite macro average differs from rows: {metric}")
    if data["episodes_per_family"] != episodes or data["seed_start"] != seed:
        raise ValueError("Suite episode metadata differs")
    events = sum(f["forced_events"] for f in counts.values())
    battles = sum(f["battles"] for f in counts.values())
    if data["forced_end_turns"] != events or not isclose(
        data["forced_end_turn_rate"], events / max(1, battles), rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Suite forcing count or battle rate differs from rows")
    return counts


def history_selection(history):
    if not history:
        raise ValueError("Missing validation history")
    best = max(history, key=midgame_checkpoint_score)  # First exact tie wins.
    selected = [r for r in history if r["selected"]]
    if not selected or selected[-1] != best:
        raise ValueError("Selection violates earliest-best validation ranking")
    return best


def run(root, output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = verify(root)
    complete = read(root / "pipeline_complete.json")
    if not complete["training_and_evaluation_complete"] or (root / "pipeline_failed.json").exists():
        raise ValueError("Requires completed, non-failed confirmation")
    design, selection = read(root / "design.json"), read(root / "selection.json")
    expected = {f"{a}-seed{s}" for s in SEEDS for a in ARMS}
    if set(selection["models"]) != expected or set(protocol["arms"]) != expected:
        raise ValueError("Requires all ten paired models")
    if selection["protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("Selection protocol changed")
    require_all_reloads(root, selection)
    counts, curves, cases, hashes, budgets, final_metrics = {}, {}, [], {}, {}, {}
    actual_settings = {}

    def evidence(path):
        hashes[str(path)] = file_digest(path)
        return read(path)

    for name in sorted(expected):
        folder = root / name
        done = evidence(root / f"{name}_complete.json")
        item = selection["models"][name]
        if item["path"] != done["best_model"] or item["sha256"] != done["best_model_sha256"]:
            raise ValueError("Selection differs from training completion")
        for field in ("best_model", "final_model"):
            if file_digest(done[field]) != done[field + "_sha256"]:
                raise ValueError("Model changed")
        model = MaskablePPO.load(done["final_model"], device="cpu")
        manifest = evidence(folder / "run_manifest.json")
        expected_config = {
            **protocol["base_config"],
            **protocol["arms"][name],
            "output_dir": str(folder),
        }
        actual_settings[name] = check_training_configuration(manifest, expected_config, model)
        budgets[name] = model.num_timesteps
        if model.num_timesteps != CANDIDATE_STEPS or not done["training_complete"]:
            raise ValueError("Incomplete final budget")
        if model.sap_environment_contract != protocol["environment_contract"]:
            raise ValueError("Final model has a different environment")
        del model
        history = evidence(folder / "validation_history.json")
        if history_selection(history) != item["selected"]:
            raise ValueError("Frozen checkpoint differs from curve selection")
        curves[name] = []
        for point in history:
            data = evidence(folder / point["evaluation_file"])
            counted = recount_suite(
                data,
                protocol["paths"]["validation"],
                design["base_config"]["validation_episodes"],
                design["base_config"]["validation_seed"],
                protocol["environment_contract"],
            )
            if tuple(point["selection_score"]) != midgame_checkpoint_score(data):
                raise ValueError("Curve score differs from evaluation")
            curves[name].append({"checkpoint": point["evaluation_file"], "families": counted})
        compare_reload(
            evidence(root / f"reload-{name}.json"),
            evidence(folder / item["selected"]["evaluation_file"]),
        )
        started = evidence(root / f"test-{name}-started.json")
        if started != {
            "model_sha256": item["sha256"],
            "selection_sha256": file_digest(root / "selection.json"),
        }:
            raise ValueError("Test was bound to different selected weights")
        test = evidence(root / f"test-{name}.json")
        counts[name] = recount_suite(
            test,
            protocol["paths"]["test"],
            design["test_episodes"],
            design["test_seed"],
            protocol["environment_contract"],
        )
        final = evidence(folder / history[-1]["evaluation_file"])
        final_metrics[name] = metrics(final)
        for stage, data, field, split in (
            ("selected_test", test, "best_model", "test"),
            ("final_validation", final, "final_model", "validation"),
        ):
            for case in choose_cases(data["families"]):
                cases.append(
                    {
                        "model": name,
                        "stage": stage,
                        "split": split,
                        "path": done[field],
                        "model_sha256": done[field + "_sha256"],
                        **case,
                    }
                )
    quality = evidence(root / "generator_quality.json")
    for name, item in protocol["roles"]["test"].items():
        model = MaskablePPO.load(item["path"], device="cpu")
        actual_settings[f"generator-{name}"] = check_training_configuration(
            evidence(root / f"generator-{name}" / "run_manifest.json"),
            design["generator_configs"][name],
            model,
        )
        budgets[f"generator-{name}"] = model.num_timesteps
        if model.num_timesteps != GENERATOR_STEPS:
            raise ValueError("Incomplete generator budget")
        del model
        actual = generator_quality(
            evidence(root / "data" / name / "generation.json")["episode_results"]
        )
        if actual != quality[name] or not actual["usable"]:
            raise ValueError("Generator quality record differs or failed")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "counts.json", {"test": counts, "curves": curves})
    save_new(
        output / "plan.json",
        {
            "script_sha256": file_digest(__file__),
            "protocol_sha256": file_digest(root / "protocol.json"),
            "selection_sha256": file_digest(root / "selection.json"),
            "evidence_sha256": hashes,
            "cases": cases,
            "selection": "Per model and stage: earliest empty loss (otherwise loss), earliest "
            "forced "
            "or truncated episode, earliest success. At most 60 deduplicated-within-stage cases. "
            "Outcome-selected diagnostic sample, not population-rate evidence. "
            "No model reselection.",
        },
    )
    replays = []
    for case in cases:
        row, family = case["expected"], case["family"]
        label = f"{case['model']}-{case['stage']}-{family}-{row['seed']}"
        with (output / f"{label}.log").open("x") as log, redirect_stdout(log):
            inspect(
                case["path"],
                protocol["paths"][case["split"]][family],
                output / label,
                1,
                row["seed"],
            )
        record = read(output / label / f"episode-{row['seed']}.json")
        compare_replay(record["summary"], row)
        if file_digest(case["path"]) != case["model_sha256"]:
            raise ValueError("Replay model changed")
        replays.append(
            {
                "directory": label,
                "model": case["model"],
                "stage": case["stage"],
                "summary": record["summary"],
                "behavior": summarize_episode(record, design["base_config"]["gamma"]),
            }
        )
        print(f"Exact stability replay: {label}", flush=True)
    verify(root)
    for path, digest in hashes.items():
        if file_digest(path) != digest:
            raise ValueError("Evidence changed during audit")
    save_new(
        output / "summary.json",
        {
            "all_counts_exact": True,
            "all_replays_exact": True,
            "actual_training_steps": budgets,
            "actual_ppo_settings": actual_settings,
            "test_rows": sum(
                f["episodes"] for families in counts.values() for f in families.values()
            ),
            "curve_rows": sum(
                f["episodes"]
                for rows in curves.values()
                for r in rows
                for f in r["families"].values()
            ),
            "final_validation": final_metrics,
            "replays": replays,
            "training_updates": 0,
            "model_reselection": False,
            "recipe_confirmed": False,
            "limitations": "Exact reconstruction is an artifact check, not an independent "
            "game-client parity proof. Final validation differs from selected held-out tests. "
            "Final evidence review remains required.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
