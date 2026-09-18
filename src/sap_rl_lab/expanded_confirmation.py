"""Staged expanded confirmation: paired seeds, frozen selection, then held-out tests.

No stage automatically calls the sandbox stable or completes the active goal.
Run stages explicitly; at most two training workers are dispatched externally.
"""

import argparse
import json
import shutil
from pathlib import Path

from .catalog import catalog_digest, load_catalog_by_id
from .evaluation import compact_evaluation, evaluate_suite, file_digest
from .expanded import ARMS, CATALOG_ID, FAMILIES, run_arm, verify_inputs
from .expanded_gate import screen
from .opponents import OpponentMixture, SnapshotLeague, build_scripted_league
from .round3 import source_archive

SEEDS = (1811, 1907, 2027)


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def latest_selected(history):
    selected = [row for row in history if row["selected"]]
    if not selected:
        raise ValueError("No validation-selected checkpoint")
    return selected[-1]


def delivery_choice(selection, candidate_arm):
    eligible = [name for name, item in selection.items() if item["arm"] == candidate_arm]
    if len(eligible) != 3 or len({selection[n]["seed"] for n in eligible}) != 3:
        raise ValueError("Delivery choice requires three independent candidate runs")
    return max(
        eligible,
        key=lambda name: (
            selection[name]["validation_unassisted_success"],
            selection[name]["validation_return"],
            -selection[name]["seed"],
        ),
    )


def prepare(pilot, output, candidate_arm, split_base, timesteps=None):
    if candidate_arm not in {"action_cost", "swap_cost"}:
        raise ValueError("Candidate must be one of the two tested reward treatments")
    if split_base < 5_000_000:
        raise ValueError("Reserve a fresh expanded held-out seed region")
    prior = read(pilot / "protocol.json")
    budget = prior["base_config"]["timesteps"] if timesteps is None else timesteps
    rollout = prior["base_config"]["environments"] * prior["base_config"]["rollout_steps"]
    if budget < prior["base_config"]["timesteps"] or budget % rollout:
        raise ValueError("Confirmation budget must cover the pilot and use complete rollouts")
    verify_inputs(pilot, prior)
    for arm in ARMS:
        completed = read(pilot / f"{arm}_complete.json")
        if not completed["training_complete"]:
            raise ValueError("Finish the full equal-budget pilot before confirmation")
        if file_digest(completed["best_model"]) != completed["best_model_sha256"]:
            raise ValueError("Pilot model changed")
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog_by_id(CATALOG_ID)
    paths = {split: {} for split in ("train", "validation", "test", "challenge")}
    for split in paths:
        (output / "data" / split).mkdir(parents=True)
    for split in ("train", "validation"):
        for family, old in prior["paths"][split].items():
            new = output / "data" / split / f"{family}.json"
            shutil.copyfile(old, new)
            paths[split][family] = str(new)
    for index, family in enumerate(FAMILIES):
        path = output / "data/test" / f"{family}.json"
        build_scripted_league(family, 200, split_base + index * 10_000, catalog=catalog).save(path)
        paths["test"][family] = str(path)

    # The pilot control is never a candidate-training opponent. Generate its
    # reachable teams without evaluating any confirmation candidate against it.
    import torch
    from sb3_contrib import MaskablePPO

    from .learned_pool import collect

    torch.set_num_threads(1)
    challenge_parent = pilot / "control/best_model.zip"
    challenge_hash = file_digest(str(challenge_parent))
    model = MaskablePPO.load(challenge_parent, device="cpu")
    league, rows = collect(
        model,
        episodes=300,
        seed=split_base + 200_000,
        opponent_provider=OpponentMixture(
            [(1.0, SnapshotLeague.load(path)) for path in paths["train"].values()]
        ),
    )
    challenge_path = output / "data/challenge/learned_pilot_control.json"
    league.save(challenge_path)
    paths["challenge"]["learned_pilot_control"] = str(challenge_path)
    if file_digest(str(challenge_parent)) != challenge_hash:
        raise ValueError("Challenge parent changed during collection")
    save_new(
        output / "challenge_generation.json",
        {
            "parent": str(challenge_parent),
            "parent_sha256": challenge_hash,
            "episode_results": rows,
            "candidate_evaluation_performed": False,
        },
    )
    base = dict(prior["base_config"])
    base.update(
        timesteps=budget,
        opponent_leagues=tuple(paths["train"].values()),
        validation_leagues=paths["validation"],
        validation_episodes=200,
        seed=SEEDS[0],
    )
    arms = {
        f"{arm}-seed{seed}": {**ARMS[arm], "seed": seed}
        for seed in SEEDS
        for arm in ("control", candidate_arm)
    }
    protocol = {
        "status": "Independent-seed confirmation; no held-out scores opened",
        "candidate_arm": candidate_arm,
        "seeds": SEEDS,
        "catalog_id": CATALOG_ID,
        "catalog_sha256": catalog_digest(catalog),
        "base_config": base,
        "arms": arms,
        "paths": paths,
        "pilot": str(pilot),
        "pilot_protocol_sha256": file_digest(str(pilot / "protocol.json")),
        "source_files_sha256": source_archive(output),
        "data_sha256": {
            str(p.relative_to(output)): file_digest(str(p))
            for p in sorted((output / "data").rglob("*.json"))
        },
        "test_episodes_per_family": 1000,
        "test_episode_seed": split_base + 100_000,
        "challenge_episode_seed": split_base + 300_000,
        "selection": "Per run: highest validation unassisted success, then raw return, "
        "earliest tie. Across candidate seeds: same metrics, smaller seed ties. "
        "Freeze all six models and delivery choice before any held-out evaluation.",
        "limits": "Equal fresh training budget and paired initial weights. Reused train/validation "
        "pools, new test pools, disjoint episode seeds. No source edits during training. "
        "Passing the numerical screen alone does not certify the full goal.",
    }
    save_new(output / "protocol.json", protocol)
    print("Prepared six independent-seed runs and unopened scripted/learned tests", flush=True)


def freeze(output):
    protocol = read(output / "protocol.json")
    verify_inputs(output, protocol)
    selection = {}
    for name, options in protocol["arms"].items():
        complete = read(output / f"{name}_complete.json")
        if not complete["training_complete"]:
            raise ValueError("All six runs must finish before selection is frozen")
        path = output / name / "best_model.zip"
        if file_digest(str(path)) != complete["best_model_sha256"]:
            raise ValueError("Completed checkpoint changed")
        manifest = read(output / name / "run_manifest.json")
        if manifest["config"]["seed"] != options["seed"]:
            raise ValueError("Recorded training seed differs from the protocol")
        history = read(output / name / "validation_history.json")
        if history[-1]["timesteps"] != protocol["base_config"]["timesteps"]:
            raise ValueError("Unequal or unfinished training budget")
        selected = latest_selected(history)
        selection[name] = {
            "path": str(path),
            "sha256": complete["best_model_sha256"],
            "seed": options["seed"],
            "arm": name.rsplit("-seed", 1)[0],
            "initial_policy_sha256": manifest["initial_policy_sha256"],
            "environment_contract": manifest["environment_contract"],
            "trained_timesteps": history[-1]["timesteps"],
            "selected_timesteps": selected["timesteps"],
            "validation_unassisted_success": selected["success_without_forcing_rate"],
            "validation_return": selected["mean_return"],
            "validation": selected,
        }
    for seed in protocol["seeds"]:
        a = selection[f"control-seed{seed}"]
        b = selection[f"{protocol['candidate_arm']}-seed{seed}"]
        if a["initial_policy_sha256"] != b["initial_policy_sha256"]:
            raise ValueError("Paired initial weights differ")
        if a["environment_contract"] != b["environment_contract"]:
            raise ValueError("Paired environment contracts differ")
    if (
        len({selection[f"control-seed{s}"]["initial_policy_sha256"] for s in protocol["seeds"]})
        != 3
    ):
        raise ValueError("Independent fresh seeds unexpectedly share initial parameters")
    save_new(
        output / "selection.json",
        {
            "models": selection,
            "delivery": delivery_choice(selection, protocol["candidate_arm"]),
            "protocol_sha256": file_digest(str(output / "protocol.json")),
            "held_out_evaluation_started": False,
        },
    )
    print("Frozen all six model choices; held-out evaluation has not started", flush=True)


def compare_reload(actual, expected):
    if set(actual["families"]) != set(expected["families"]):
        raise ValueError("Reloaded validation families differ")
    for family in actual["families"]:
        if (
            actual["families"][family]["episode_results"]
            != expected["families"][family]["episode_results"]
        ):
            raise ValueError(f"Reload did not reproduce every validation row: {family}")


def verify_reload(output, name):
    import torch
    from sb3_contrib import MaskablePPO

    protocol, selection = read(output / "protocol.json"), read(output / "selection.json")
    verify_inputs(output, protocol)
    chosen = selection["models"][name]
    if file_digest(chosen["path"]) != chosen["sha256"]:
        raise ValueError("Selected model changed before reload check")
    torch.set_num_threads(1)
    model = MaskablePPO.load(chosen["path"], device="cpu")
    result = evaluate_suite(
        model,
        protocol["paths"]["validation"],
        episodes=protocol["base_config"]["validation_episodes"],
        seed=protocol["base_config"]["validation_seed"],
    )
    expected = read(output / name / chosen["validation"]["evaluation_file"])
    compare_reload(result, expected)
    save_new(
        output / f"reload-{name}.json",
        {
            "all_validation_rows_exact": True,
            "model_sha256": chosen["sha256"],
            "selection_sha256": file_digest(str(output / "selection.json")),
        },
    )
    print(f"Reload exactly reproduced validation rows for {name}", flush=True)


def evaluate(output, name):
    import torch
    from sb3_contrib import MaskablePPO

    protocol, selection = read(output / "protocol.json"), read(output / "selection.json")
    verify_inputs(output, protocol)
    if file_digest(str(output / "protocol.json")) != selection["protocol_sha256"]:
        raise ValueError("Protocol changed after selection")
    torch.set_num_threads(1)
    options = {"catalog": load_catalog_by_id(CATALOG_ID), "allow_development": True}
    if name in FAMILIES:
        policy, model_hash = name, None
    else:
        chosen = selection["models"][name]
        reload_check = read(output / f"reload-{name}.json")
        if not reload_check["all_validation_rows_exact"] or (
            reload_check["model_sha256"] != chosen["sha256"]
            or reload_check["selection_sha256"] != file_digest(str(output / "selection.json"))
        ):
            raise ValueError("A matching exact validation reload check is required before test")
        model_hash = file_digest(chosen["path"])
        if model_hash != chosen["sha256"]:
            raise ValueError("Frozen selected model changed")
        policy = MaskablePPO.load(chosen["path"], device="cpu")
    save_new(
        output / f"evaluation-{name}-started.json",
        {
            "selection_sha256": file_digest(str(output / "selection.json")),
            "model_sha256": model_hash,
        },
    )
    result = {}
    for split, seed_key in (("test", "test_episode_seed"), ("challenge", "challenge_episode_seed")):
        result[split] = evaluate_suite(
            policy,
            protocol["paths"][split],
            episodes=protocol["test_episodes_per_family"],
            seed=protocol[seed_key],
            env_kwargs=options,
        )
    verify_inputs(output, protocol)
    if model_hash is not None and file_digest(selection["models"][name]["path"]) != model_hash:
        raise ValueError("Selected model changed during evaluation")
    save_new(output / f"evaluation-{name}.json", result)
    print(f"Evaluated frozen {name}; numerical and full-goal audits remain separate", flush=True)


def summarize(output):
    protocol, selection = read(output / "protocol.json"), read(output / "selection.json")
    verify_inputs(output, protocol)
    if file_digest(str(output / "protocol.json")) != selection["protocol_sha256"]:
        raise ValueError("Protocol changed after selection")
    selection_hash = file_digest(str(output / "selection.json"))
    for name in (*selection["models"], *FAMILIES):
        started = read(output / f"evaluation-{name}-started.json")
        expected_model = (
            selection["models"][name]["sha256"] if name in selection["models"] else None
        )
        if (
            started["selection_sha256"] != selection_hash
            or started["model_sha256"] != expected_model
        ):
            raise ValueError("Evaluation does not belong to the frozen selection")
    results = {
        name: read(output / f"evaluation-{name}.json") for name in (*selection["models"], *FAMILIES)
    }
    pairs = [
        {
            "seed": seed,
            "candidate": results[f"{protocol['candidate_arm']}-seed{seed}"]["test"],
            "control": results[f"control-seed{seed}"]["test"],
        }
        for seed in protocol["seeds"]
    ]
    save_new(
        output / "summary.json",
        {
            **screen(pairs),
            "delivery": selection["delivery"],
            "results": {
                name: {split: compact_evaluation(data) for split, data in result.items()}
                for name, result in results.items()
            },
            "selection_sha256": file_digest(str(output / "selection.json")),
        },
    )
    print("Saved numerical screen and separate learned-challenge/baseline results", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "run", "freeze", "reload", "evaluate", "summarize")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot")
    parser.add_argument("--candidate-arm", choices=("action_cost", "swap_cost"))
    parser.add_argument("--name")
    parser.add_argument("--split-base", type=int, default=5_000_000)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.stage == "prepare":
        if not args.pilot or not args.candidate_arm:
            parser.error("prepare requires --pilot and --candidate-arm")
        prepare(
            Path(args.pilot).resolve(), output, args.candidate_arm, args.split_base, args.timesteps
        )
    elif args.stage == "run":
        if args.name not in read(output / "protocol.json")["arms"]:
            parser.error("run requires one prespecified --name")
        run_arm(output, args.name)
    elif args.stage == "freeze":
        freeze(output)
    elif args.stage == "reload":
        if not args.name:
            parser.error("reload requires --name")
        verify_reload(output, args.name)
    elif args.stage == "evaluate":
        if not args.name:
            parser.error("evaluate requires --name")
        evaluate(output, args.name)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
