"""Predeclared paired continuation experiment with validation/test separation."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from .evaluation import evaluate_policy, file_digest
from .opponents import build_spend_gold_league
from .training import TrainingConfig, train


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def compact(result):
    return {k: v for k, v in result.items() if k not in {"episode_results", "failure_examples"}}


def paired_comparison(candidate, control, *, resamples=5000):
    """Episode-seed paired bootstrap for this pair of fixed policies, not seeds."""
    import numpy as np

    a, b = candidate["episode_results"], control["episode_results"]
    if [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("paired comparisons require identical episode seed lists")
    if candidate["league_sha256"] != control["league_sha256"]:
        raise ValueError("paired comparisons require the same opponent league")
    rng = np.random.default_rng(905)
    result = {}
    for metric in ("success", "wins", "return", "actions"):
        differences = np.asarray([float(x[metric]) - float(y[metric]) for x, y in zip(a, b)])
        samples = rng.choice(differences, size=(resamples, len(differences)), replace=True).mean(1)
        result[metric] = {
            "mean_difference": float(differences.mean()),
            "episode_bootstrap_95_percent": list(map(float, np.quantile(samples, [0.025, 0.975]))),
        }
    return result


def run(args):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    base = Path(args.base_run).resolve()
    initial = base / "train/final_model.zip"
    source_league = base / "data/train.json"
    if not initial.is_file() or not source_league.is_file():
        raise FileNotFoundError("base-run must contain train/final_model.zip and data/train.json")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    shutil.copyfile(source_league, data / "train.json")
    # These exact pools/seeds are fixed before seeing any candidate result.
    build_spend_gold_league(100, seed=30000).save(data / "validation.json")
    build_spend_gold_league(100, seed=50000).save(data / "test.json")
    base_config = TrainingConfig(
        timesteps=args.timesteps,
        environments=8,
        seed=17,
        opponent_league=str(data / "train.json"),
        initialize_from=str(initial),
        vector_backend=args.vector_backend,
        device=args.device,
        validation_league=str(data / "validation.json"),
        validation_episodes=200,
        validation_seed=40000,
        evaluation_interval=100_000,
        source_commit=args.source_commit,
    )
    protocol = {
        "base_config": asdict(base_config),
        "arms": {
            "control": {"action_cost": 0.0, "forfeit_on_limit": False},
            "efficient": {"action_cost": 0.005, "forfeit_on_limit": True},
        },
        "continuation_seeds": args.seeds,
        "shared_initial_model_sha256": file_digest(str(initial)),
        "data_sha256": {p.name: file_digest(str(p)) for p in data.iterdir()},
        "test_episode_seed_start": 60000,
        "test_episodes": 1000,
        "selection": "maximum validation success, then return; earliest wins exact ties",
        "scope": "Continuation seeds share one trained initialization; not scratch runs.",
    }
    save_json(output / "protocol.json", protocol)
    candidates = {}
    for seed in args.seeds:
        for arm, settings in protocol["arms"].items():
            name = f"{arm}-seed{seed}"
            config = TrainingConfig(
                **{
                    **asdict(base_config),
                    **settings,
                    "seed": seed,
                    "output_dir": str(output / name),
                }
            )
            started = time.monotonic()
            print(f"START {name}", flush=True)
            with (output / f"{name}-training.log").open("x", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    train(config)
            candidates[name] = str(output / name / "best_model.zip")
            save_json(output / name / "runtime.json", {"seconds": time.monotonic() - started})
            print(f"FINISHED {name}", flush=True)
    if args.pilot:
        print("Pilot complete; final test pool has not been evaluated.", flush=True)
        return

    test_results = {}
    for name, path in {"original": str(initial), **candidates}.items():
        model = MaskablePPO.load(path, device=args.device)
        result = evaluate_policy(
            model, episodes=1000, seed=60000, opponent_league=str(data / "test.json")
        )
        result["model_sha256"] = file_digest(path)
        save_json(output / f"test-{name}.json", result)
        test_results[name] = result
        print(f"TEST {name}: {json.dumps(compact(result))}", flush=True)
        del model
    for name in ("greedy", "random"):
        result = evaluate_policy(
            name, episodes=1000, seed=60000, opponent_league=str(data / "test.json")
        )
        save_json(output / f"test-{name}.json", result)
        test_results[name] = result
    comparisons = {
        str(seed): paired_comparison(
            test_results[f"efficient-seed{seed}"], test_results[f"control-seed{seed}"]
        )
        for seed in args.seeds
    }
    summary = {
        "results": {k: compact(v) for k, v in test_results.items()},
        "paired_comparisons": comparisons,
        "mean_success_by_arm": {
            arm: fmean(test_results[f"{arm}-seed{s}"]["success_rate"] for s in args.seeds)
            for arm in protocol["arms"]
        },
        "mean_success_difference_across_continuation_seeds": fmean(
            comparisons[str(s)]["success"]["mean_difference"] for s in args.seeds
        ),
        "inference_limits": "Episode intervals condition on fixed policies. Across-seed means "
        "are descriptive; all continuations share one initial model and one test pool. "
        "The two shaping changes are tested together; this does not isolate their effects.",
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vector-backend", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    run(args)


if __name__ == "__main__":
    main()
