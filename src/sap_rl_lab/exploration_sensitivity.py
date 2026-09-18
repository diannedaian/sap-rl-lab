"""One bounded lower-entropy comparison; preserves the original pilot source."""

import argparse
import time
from pathlib import Path

from .evaluation import evaluate_suite, file_digest
from .expanded import verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .exploration_pilot import SEEDS, batch, metrics
from .historical_confirmation import copy_checked
from .midgame import ROOT
from .round3 import source_archive

STEPS = 2_097_152
ENTROPIES = {"entropy0003": 0.003, "entropy001": 0.01}


def prepare(output):
    prior_root = ROOT / "runs/exploration-pilot-v1"
    prior = read(prior_root / "protocol.json")
    verify_inputs(prior_root, prior)
    summary = read(prior_root / "pilot_summary.json")
    audit = read(ROOT / "runs/exploration-pilot-audit-v1/summary.json")
    if not (
        summary["all_six_budgets_complete"]
        and summary["all_selected_validation_reloads_exact"]
        and audit["all_curves_recounted"]
        and audit["all_replays_exact"]
    ):
        raise ValueError("First pilot and replay audit must finish")
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split in ("train", "validation"):
        paths[split] = {}
        for family, source in prior["paths"][split].items():
            destination = output / "data" / split / f"{family}.json"
            copy_checked(source, destination)
            paths[split][family] = str(destination)
    config = dict(prior["base_config"])
    config.update(
        timesteps=STEPS,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        evaluation_interval=524_288,
    )
    save_new(
        output / "protocol.json",
        {
            "base_config": config,
            "arms": {
                f"{arm}-seed{seed}": {"seed": seed, "entropy_coefficient": value}
                for seed in SEEDS
                for arm, value in ENTROPIES.items()
            },
            "paths": paths,
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for p in sorted((output / "data").rglob("*.json"))
            },
            "parent_protocol_sha256": file_digest(prior_root / "protocol.json"),
            "parent_summary_sha256": file_digest(prior_root / "pilot_summary.json"),
            "parent_replay_audit_sha256": file_digest(
                ROOT / "runs/exploration-pilot-audit-v1/summary.json"
            ),
            "training_steps_total": len(SEEDS) * len(ENTROPIES) * STEPS,
            "scope": "Fresh initializations, same diagnostic seeds and mixed pools. Compare "
            "constant entropy 0.003 versus 0.01 at an equal longer horizon; do not inherit pilot "
            "weights. Only entropy differs within pairs. Reuses validation, not independent proof.",
            "budget": "Combines the two previously reserved supplementary pilot budgets. "
            "No additional pilot search beyond this comparison is authorized by this protocol.",
            "review": "Consider the entire curve, final no-purchase losses, mean/ten wins, and "
            "forcing. A recipe still requires full-budget fresh-seed independent confirmation.",
        },
    )


def review(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = read(output / "protocol.json")
    verify_inputs(output, p)
    results = {}
    for name in p["arms"]:
        folder = output / name
        done = read(output / f"{name}_complete.json")
        if not done["training_complete"]:
            raise ValueError("Incomplete sensitivity arm")
        for field in ("best_model", "final_model"):
            if file_digest(done[field]) != done[field + "_sha256"]:
                raise ValueError("Sensitivity model changed")
        final = MaskablePPO.load(done["final_model"], device="cpu")
        if final.num_timesteps != STEPS:
            raise ValueError("Sensitivity budget mismatch")
        history = read(folder / "validation_history.json")
        selected = [r for r in history if r["selected"]][-1]
        model = MaskablePPO.load(done["best_model"], device="cpu")
        actual = evaluate_suite(
            model,
            p["paths"]["validation"],
            episodes=p["base_config"]["validation_episodes"],
            seed=p["base_config"]["validation_seed"],
        )
        compare_reload(actual, read(folder / selected["evaluation_file"]))
        save_new(output / f"reload-{name}.json", actual)
        results[name] = {
            "initial_policy_sha256": read(folder / "run_manifest.json")["initial_policy_sha256"],
            "actual_steps": final.num_timesteps,
            "selected_steps": selected["timesteps"],
            "selected_model_sha256": done["best_model_sha256"],
            "selected": metrics(actual),
            "final": metrics(read(folder / history[-1]["evaluation_file"])),
            "curves": [
                {
                    "timesteps": r["timesteps"],
                    "evaluation_file": r["evaluation_file"],
                    **metrics(read(folder / r["evaluation_file"])),
                }
                for r in history
            ],
            "elapsed_training_seconds": done["elapsed_seconds"],
        }
    for seed in SEEDS:
        names = [f"{arm}-seed{seed}" for arm in ENTROPIES]
        if len({results[name]["initial_policy_sha256"] for name in names}) != 1:
            raise ValueError("Paired initial weights differ")
        compare_reload(
            *(read(output / name / results[name]["curves"][0]["evaluation_file"]) for name in names)
        )
    if len({r["initial_policy_sha256"] for r in results.values()}) != len(SEEDS):
        raise ValueError("Different seeds share initial weights")
    verify_inputs(output, p)
    save_new(
        output / "pilot_summary.json",
        {
            "results": results,
            "all_six_budgets_complete": True,
            "all_selected_validation_reloads_exact": True,
            "paired_initial_weights_and_rows_exact": True,
            "held_out_tests_opened": False,
            "recipe_confirmed": False,
            "next_step": "Review final-weight replays, then freeze one candidate and the "
            "full-budget independent-confirmation protocol. No further coefficient search.",
        },
    )


def pipeline(output):
    started = time.monotonic()
    prepare(output)
    try:
        batch(output)
        review(output)
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise
    print(
        f"Sensitivity pilot complete after {time.monotonic() - started:.1f}s; "
        "independent recipe confirmation remains required",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "review", "pipeline"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "review": review, "pipeline": pipeline}[args.command](
        Path(args.output).resolve()
    )
