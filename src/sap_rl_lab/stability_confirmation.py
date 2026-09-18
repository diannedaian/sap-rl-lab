"""Five paired seeds and fresh held-out generators; no adaptive search or retries."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from statistics import fmean

from .evaluation import evaluate_suite, file_digest
from .expanded import run_arm, verify_inputs
from .expanded_confirmation import compare_reload, read, save_new
from .exploration_pilot import metrics
from .generalization import validate_roles
from .historical_confirmation import collect_pool, copy_checked, fallback_counts
from .midgame import FAMILIES, ROOT
from .midgame_confirmation import check_protocol as check_parent
from .midgame_confirmation import new_config
from .opponents import OpponentMixture, SnapshotLeague, build_scripted_league
from .round3 import source_archive
from .training import TrainingConfig, midgame_checkpoint_score, policy_digest, train

SEEDS = (201101, 201201, 201301, 201401, 201501)
ARMS = ("control", "candidate")
CANDIDATE_STEPS = 8_388_608
GENERATOR_STEPS = 4_194_304
GENERATORS = {"fresh_stats": 202101, "fresh_mix": 202201}
VALIDATION_SEED = 22_000_000
TEST_SEED = 23_000_000


def checked(output):
    design = read(output / "design.json")
    verify_inputs(output, design)
    if file_digest(output / "candidate_decision.json") != design["decision_sha256"]:
        raise ValueError("Candidate decision changed")
    return design


def verify(output):
    checked(output)
    p = read(output / "protocol.json")
    verify_inputs(output, p)
    if p["design_sha256"] != file_digest(output / "design.json"):
        raise ValueError("Design changed after opponent freeze")
    for entries in p["roles"].values():
        for item in entries.values():
            if file_digest(item["path"]) != item["sha256"]:
                raise ValueError("Opponent generator changed")
    return p


def generator_quality(rows):
    if len(rows) != 500:
        raise ValueError("Generator quality requires all 500 predeclared episodes")
    empty = fmean(r["wins"] == 0 and not r["action_counts"].get("buy_pet", 0) for r in rows)
    wins = fmean(r["wins"] for r in rows)
    truncations = sum(r["truncated"] for r in rows)
    return {
        "mean_wins": wins,
        "no_purchase_zero_win_loss_rate": empty,
        "truncations": truncations,
        "episodes": len(rows),
        "usable": wins >= 2 and empty <= 0.01 and truncations == 0,
    }


def prepare(output, decision_path):
    prior_root = ROOT / "runs/exploration-sensitivity-v1"
    prior = read(prior_root / "protocol.json")
    verify_inputs(prior_root, prior)
    summary = read(prior_root / "pilot_summary.json")
    decision = read(decision_path)
    if not (
        summary["all_six_budgets_complete"]
        and summary["all_selected_validation_reloads_exact"]
        and summary["paired_initial_weights_and_rows_exact"]
    ):
        raise ValueError("Sensitivity experiment is incomplete")
    entropy = decision["candidate_entropy"]
    if entropy not in (0.003, 0.01) or not decision["rationale"].strip():
        raise ValueError("Requires one reviewed candidate and its rationale")
    if decision["pilot_summary_sha256"] != file_digest(prior_root / "pilot_summary.json"):
        raise ValueError("Decision does not cover the actual sensitivity results")
    audit_path = Path(decision["replay_audit"])
    audit = read(audit_path)
    if (
        file_digest(audit_path) != decision["replay_audit_sha256"]
        or not audit["all_curves_recounted"]
        or not audit["all_replays_exact"]
    ):
        raise ValueError("Requires completed sensitivity replay audit")
    if read(audit_path.parent / "plan.json")["protocol_sha256"] != file_digest(
        prior_root / "protocol.json"
    ):
        raise ValueError("Replay audit belongs to a different pilot")
    parent = check_parent(ROOT / "runs/midgame-confirmation-v1")
    output.mkdir(parents=True, exist_ok=False)
    copy_checked(decision_path, output / "candidate_decision.json")
    paths = {split: {} for split in ("train", "validation", "test")}
    for split in ("train", "validation"):
        for family, source in prior["paths"][split].items():
            target = output / "data" / split / f"{family}.json"
            copy_checked(source, target)
            paths[split][family] = str(target)
    config = new_config(
        timesteps=CANDIDATE_STEPS,
        validation_leagues=paths["validation"],
        validation_episodes=200,
        validation_seed=VALIDATION_SEED,
        evaluation_interval=1_048_576,
    )
    catalog, game = TrainingConfig(**config).environment()
    (output / "data/test").mkdir()
    for i, family in enumerate(FAMILIES):
        target = output / "data/test" / f"{family}.json"
        build_scripted_league(
            family, 500, 20_000_000 + i * 10_000, catalog=catalog, config=game
        ).save(target)
        paths["test"][family] = str(target)
    generator_configs = {}
    for name, seed in GENERATORS.items():
        against = (
            [paths["train"][FAMILIES[0]]]
            if name == "fresh_stats"
            else list(paths["train"].values())
        )
        generator_configs[name] = new_config(
            timesteps=GENERATOR_STEPS,
            seed=seed,
            entropy_coefficient=0.01,
            opponent_leagues=tuple(against),
            output_dir=str(output / f"generator-{name}"),
        )
    save_new(
        output / "design.json",
        {
            "base_config": config,
            "paths": paths,
            "candidate_entropy": entropy,
            "candidate_steps": CANDIDATE_STEPS,
            "generator_configs": generator_configs,
            "test_episodes": 1000,
            "test_seed": TEST_SEED,
            "generator_episodes": 500,
            "prior_roles": {role: parent["roles"][role] for role in ("train", "validation")},
            "decision_sha256": file_digest(output / "candidate_decision.json"),
            "source_files_sha256": source_archive(output),
            "data_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for p in sorted((output / "data").rglob("*.json"))
            },
            "total_training_decisions": 10 * CANDIDATE_STEPS + 2 * GENERATOR_STEPS,
            "selection": "Shared v5 win-first, worst-family, forcing, raw-return rule; "
            "earliest checkpoint on exact ties. All selections freeze before any test.",
            "acceptance": "All five candidates: <=1% no-purchase zero-win loss per test family, "
            "<=2% forced episodes per test family, zero test truncations, no final-validation "
            "empty-team relapse (>1%). Mean learned-test difference >=0 and at least four "
            "nonnegative seed pairs. Engineering screen only; "
            "final evidence review and written interpretation required.",
            "generator_quality_gate": "500 reachable episodes against the mixed training pool; "
            "mean wins >=2, <=1% no-purchase zero-win losses, zero truncations. No candidate "
            "evaluation used for this gate. Failure stops before candidate training, no retry.",
            "scope": "Fresh five paired seeds, same mixed opponents. Only entropy differs "
            "within pairs. Two fresh test generators (stats versus mixed training), fixed entropy "
            "0.01, distinct seeds and final weights. "
            "No old test pool or score used as new evidence.",
        },
    )


def generator(output, name):
    design = checked(output)
    started = time.monotonic()
    with (output / f"generator-{name}.log").open("x") as log:
        with redirect_stdout(log), redirect_stderr(log):
            path = train(TrainingConfig(**design["generator_configs"][name]))
    checked(output)
    save_new(
        output / f"generator-{name}_complete.json",
        {
            "path": str(path),
            "sha256": file_digest(path),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def seal(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    design = checked(output)
    paths = {k: dict(v) for k, v in design["paths"].items()}
    roles = {**design["prior_roles"], "test": {}}
    against = OpponentMixture([(1, SnapshotLeague.load(p)) for p in paths["train"].values()])
    contract, quality = None, {}
    for i, name in enumerate(GENERATORS):
        item = read(output / f"generator-{name}_complete.json")
        if file_digest(item["path"]) != item["sha256"]:
            raise ValueError("Generator changed")
        model = MaskablePPO.load(item["path"], device="cpu")
        if model.num_timesteps != GENERATOR_STEPS:
            raise ValueError("Generator budget incomplete")
        if contract is None:
            contract = model.sap_environment_contract
        if model.sap_environment_contract != contract:
            raise ValueError("Generator environments differ")
        roles["test"][name] = {**item, "policy_sha256": policy_digest(model.policy)}
        paths["test"][name] = collect_pool(
            output,
            name,
            item["path"],
            against,
            design["generator_episodes"],
            21_000_000 + i * 10_000,
        )
        quality[name] = generator_quality(
            read(output / "data" / name / "generation.json")["episode_results"]
        )
    save_new(output / "generator_quality.json", quality)
    if not all(q["usable"] for q in quality.values()):
        raise ValueError("A fresh test generator fails the predeclared quality gate")
    validate_roles(roles, {})
    base = design["base_config"]
    if contract["catalog_id"] != base["catalog_id"]:
        raise ValueError("Unexpected generator environment")
    save_new(
        output / "protocol.json",
        {
            "base_config": base,
            "paths": paths,
            "roles": roles,
            "environment_contract": contract,
            "arms": {
                f"{arm}-seed{seed}": {
                    "seed": seed,
                    "opponent_leagues": list(paths["train"].values()),
                    "entropy_coefficient": 0.0 if arm == "control" else design["candidate_entropy"],
                }
                for seed in SEEDS
                for arm in ARMS
            },
            "source_files_sha256": design["source_files_sha256"],
            "data_sha256": {
                str(p.relative_to(output)): file_digest(p)
                for p in sorted((output / "data").rglob("*.json"))
            },
            "design_sha256": file_digest(output / "design.json"),
        },
    )


def freeze(output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p = verify(output)
    models = {}
    for name in p["arms"]:
        done = read(output / f"{name}_complete.json")
        history = read(output / name / "validation_history.json")
        final = MaskablePPO.load(done["final_model"], device="cpu")
        if not done["training_complete"] or final.num_timesteps != CANDIDATE_STEPS:
            raise ValueError("Incomplete candidate budget")
        if final.sap_environment_contract != p["environment_contract"]:
            raise ValueError("Candidate rules differ from generators")
        for field in ("best_model", "final_model"):
            if file_digest(done[field]) != done[field + "_sha256"]:
                raise ValueError("Candidate weights changed")
        selected = [r for r in history if r["selected"]][-1]
        if tuple(selected["selection_score"]) != max(map(midgame_checkpoint_score, history)):
            raise ValueError("Selected checkpoint violates the frozen ranking")
        models[name] = {
            "path": done["best_model"],
            "sha256": done["best_model_sha256"],
            "selected": selected,
            "final_validation": history[-1],
            "initial_policy_sha256": read(output / name / "run_manifest.json")[
                "initial_policy_sha256"
            ],
        }
    initializers = {}
    for seed in SEEDS:
        names = [f"{arm}-seed{seed}" for arm in ARMS]
        if len({models[n]["initial_policy_sha256"] for n in names}) != 1:
            raise ValueError("Paired initial weights differ")
        initializers[str(seed)] = {"policy_sha256": models[names[0]]["initial_policy_sha256"]}
        rows = []
        for name in names:
            first = read(output / name / "validation_history.json")[0]
            rows.append(read(output / name / first["evaluation_file"]))
        compare_reload(*rows)
    if len({v["policy_sha256"] for v in initializers.values()}) != 5:
        raise ValueError("New seed pairs share initial weights")
    validate_roles(p["roles"], initializers)
    delivery = max(
        (n for n in models if n.startswith("candidate-")),
        key=lambda n: (
            *midgame_checkpoint_score(models[n]["selected"]),
            -int(n.rsplit("seed", 1)[1]),
        ),
    )
    save_new(
        output / "selection.json",
        {
            "models": models,
            "delivery": delivery,
            "protocol_sha256": file_digest(output / "protocol.json"),
        },
    )


def require_all_reloads(output, selection):
    for name, item in selection["models"].items():
        record = read(output / f"reload-{name}.json")
        if not (
            record["all_validation_rows_exact"]
            and record["model_sha256"] == item["sha256"]
            and record["selection_sha256"] == file_digest(output / "selection.json")
        ):
            raise ValueError("All frozen model reloads must pass before any test")


def inference(output, name, stage):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    p, design = verify(output), checked(output)
    selection = read(output / "selection.json")
    if selection["protocol_sha256"] != file_digest(output / "protocol.json"):
        raise ValueError("Selection protocol changed")
    item = selection["models"][name]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Selected model changed")
    if stage == "test":
        require_all_reloads(output, selection)
        paths, episodes, seed = p["paths"]["test"], design["test_episodes"], TEST_SEED
    else:
        paths, episodes, seed = p["paths"]["validation"], 200, VALIDATION_SEED
    started_path = output / f"{stage}-{name}-started.json"
    save_new(
        started_path,
        {
            "model_sha256": item["sha256"],
            "selection_sha256": file_digest(output / "selection.json"),
        },
    )
    model = MaskablePPO.load(item["path"], device="cpu")
    actual = evaluate_suite(model, paths, episodes=episodes, seed=seed)
    if stage == "reload":
        compare_reload(actual, read(output / name / item["selected"]["evaluation_file"]))
        actual.update(
            all_validation_rows_exact=True,
            model_sha256=item["sha256"],
            selection_sha256=file_digest(output / "selection.json"),
        )
    actual["opponent_fallback_coverage"] = {
        family: fallback_counts(
            actual["families"][family]["episode_results"],
            {s["turn"] for s in read(path)["snapshots"]},
        )
        for family, path in paths.items()
    }
    verify(output)
    save_new(output / f"{stage}-{name}.json", actual)


def acceptance(pairs):
    candidate = [p["candidate"] for p in pairs.values()]
    differences = [p["learned_difference"] for p in pairs.values()]
    return {
        "five_pairs": len(pairs) == 5,
        "no_purchase_losses_controlled": all(
            c["worst_family_no_purchase_loss_rate"] <= 0.01 for c in candidate
        ),
        "forcing_controlled": all(c["worst_family_forced_episode_rate"] <= 0.02 for c in candidate),
        "zero_truncations": all(c["worst_family_truncation_rate"] == 0 for c in candidate),
        "no_final_validation_empty_relapse": all(
            p["final_no_purchase_loss_rate"] <= 0.01 for p in pairs.values()
        ),
        "mean_learned_performance_not_lower": bool(differences) and fmean(differences) >= 0,
        "at_least_four_nonnegative_pairs": sum(d >= 0 for d in differences) >= 4,
    }


def summarize(output):
    import numpy as np

    from .experiments import paired_comparison

    p = verify(output)
    pairs = {}
    for seed in SEEDS:
        results = {a: read(output / f"test-{a}-seed{seed}.json") for a in ARMS}
        row = {a: metrics(r) for a, r in results.items()}
        for arm in ARMS:
            row[arm]["learned_success"] = fmean(
                results[arm]["families"][f]["success_rate"] for f in GENERATORS
            )
            row[arm]["scripted_success"] = fmean(
                results[arm]["families"][f]["success_rate"] for f in FAMILIES
            )
        row["learned_difference"] = (
            row["candidate"]["learned_success"] - row["control"]["learned_success"]
        )
        folder = output / f"candidate-seed{seed}"
        history = read(folder / "validation_history.json")
        row["final_no_purchase_loss_rate"] = metrics(read(folder / history[-1]["evaluation_file"]))[
            "worst_family_no_purchase_loss_rate"
        ]
        row["family_comparisons"] = {
            family: paired_comparison(
                results["candidate"]["families"][family],
                results["control"]["families"][family],
                resamples=2000,
            )
            for family in p["paths"]["test"]
        }
        pairs[str(seed)] = row
    differences = np.array([v["learned_difference"] for v in pairs.values()])
    boot = np.random.default_rng(24_000_000).choice(differences, size=(10000, 5)).mean(axis=1)
    gates = acceptance(pairs)
    save_new(
        output / "confirmation_summary.json",
        {
            "pairs": pairs,
            "mean_learned_difference": float(differences.mean()),
            "seed_bootstrap_95_percent": np.quantile(boot, [0.025, 0.975]).tolist(),
            "gates": gates,
            "practical_screen_passed": all(gates.values()),
            "recipe_confirmed": False,
            "evidence_review_required": True,
            "delivery": read(output / "selection.json")["delivery"],
            "limitations": "Five seeds and two new learned generators, not universal stability. "
            "Seed bootstrap differs from within-family episode intervals. Reused "
            "training/validation ecology, finite historical lineups and late-turn coverage "
            "biases remain. No test-based reselection.",
        },
    )


def batch(output, stage, names):
    for start in range(0, len(names), 2):
        children = []
        try:
            for name in names[start : start + 2]:
                print(f"Starting stability {stage}: {name}", flush=True)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "sap_rl_lab.stability_confirmation",
                            stage,
                            "--output",
                            str(output),
                            "--name",
                            name,
                        ],
                        cwd=ROOT,
                    )
                )
            deadline = time.monotonic() + 7200
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError(f"{stage} worker failed; no retry")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{stage} pair exceeded two hours")
                time.sleep(1)
            if any(c.returncode != 0 for c in children):
                raise RuntimeError(f"{stage} worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)


def pipeline(output, decision):
    started = time.monotonic()
    prepare(output, decision)
    try:
        batch(output, "generator", list(GENERATORS))
        seal(output)
        names = list(verify(output)["arms"])
        batch(output, "arm", names)
        freeze(output)
        batch(output, "reload", names)
        batch(output, "test", names)
        summarize(output)
        save_new(
            output / "pipeline_complete.json",
            {
                "training_and_evaluation_complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "human_readable_report_pending": True,
                "deployment_performed": False,
            },
        )
    except Exception as error:
        save_new(output / "pipeline_failed.json", {"error": repr(error), "automatic_retry": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pipeline", "generator", "arm", "reload", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision")
    parser.add_argument("--name")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "pipeline":
        if not args.decision:
            parser.error("Pipeline requires a reviewed --decision file")
        pipeline(output, Path(args.decision).resolve())
    elif args.command == "generator":
        generator(output, args.name)
    elif args.command == "arm":
        run_arm(output, args.name)
    else:
        inference(output, args.name, args.command)
