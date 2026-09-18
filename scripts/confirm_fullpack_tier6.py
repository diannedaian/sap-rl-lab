"""One-shot fresh-episode confirmation of the frozen full Tier6 screened triple.

No model selection, training, new opponent pools, or retry with different seeds.
Passing this numerical check is not the full rule/recipe/delivery evidence audit.
"""

import argparse
import importlib.util
from pathlib import Path

from sap_rl_lab.fullpack.evaluation import evaluate_suite, file_digest, policy_environment_options
from sap_rl_lab.fullpack.ppo_guardrails import family_metrics, write_new
from sap_rl_lab.fullpack.training import policy_digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "confirmation_campaign", ROOT / "scripts/train_fullpack_tier6_h40.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
SEED = 171_000_000
EPISODES = 500
FAMILIES = {"full_stats", "full_summon", "full_tempo", "test_balanced", "test_summon_tempo"}


def summarize(results, seed):
    if set(results) != set(map(str, campaign.SEEDS)):
        raise ValueError("Need the predeclared three models, not the best seed")
    successes, reliable, models = 0, True, {}
    for name, result in results.items():
        if set(result["families"]) != FAMILIES:
            raise ValueError("Need all five frozen test families")
        metrics = family_metrics(result)
        learned = 0
        for family, evidence in result["families"].items():
            rows = evidence["episode_results"]
            if len(rows) != EPISODES or [r["seed"] for r in rows] != list(
                range(seed, seed + EPISODES)
            ):
                raise ValueError("Incomplete, duplicate, or unplanned episode seeds")
            if not evidence["deterministic"]:
                raise ValueError("Evaluation policy must be deterministic")
            m = metrics[family]
            reliable &= m["forcing"] <= 0.05 and m["truncation"] == 0
            reliable &= m["no_purchase_loss"] <= 0.05
            if family.startswith("test_"):
                learned += sum(r["success"] for r in rows)
        successes += learned
        models[name] = {"learned_ten_win_rate": learned / 1000, "families": metrics}
    return {
        "models": models,
        "learned_successes": successes,
        "learned_episodes": 3000,
        "learned_ten_win_rate": successes / 3000,
        "all_families_reliable": bool(reliable),
        "passed": bool(reliable and successes >= 2400),
    }


def confirm(root):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    campaign.checked(root)
    screen_path = root / "reached80_screen.json"
    screen = campaign.read(screen_path)
    if not screen["screen_passed"] or (
        screen["ordinary_shop_tier"],
        screen["pets"],
        screen["max_turns"],
    ) != (6, 60, 40):
        raise ValueError("Full Tier6 screening has not passed")
    folder = campaign.segment_dir(root, screen["segment"])
    selection_path = folder / "selection.json"
    if Path(screen["selection"]).resolve() != selection_path.resolve():
        raise ValueError("Screen points to a different selection")
    chosen = campaign.sealed(selection_path)["selected"]
    cp = campaign.sealed(root / "candidate_protocol.json")
    if set(cp["paths"]["test"]) != FAMILIES:
        raise ValueError("Unexpected opponent families")
    original = {}
    bindings = [screen_path, selection_path, root / "candidate_protocol.json", Path(__file__)]
    for name in map(str, campaign.SEEDS):
        evidence_path = folder / f"test-{name}.json"
        evidence = campaign.read(evidence_path)
        if evidence["checkpoint"] != chosen[name]["checkpoint"]:
            raise ValueError("Test belongs to different selected weights")
        original[name] = evidence["evaluation"]
        bindings.extend([evidence_path, Path(chosen[name]["checkpoint"]["path"])])
    recomputed = summarize(original, 120_000_000)
    if not recomputed["passed"] or recomputed["learned_successes"] != screen["learned_successes"]:
        raise ValueError("Screening claim does not match raw episode evidence")
    for path in cp["paths"]["test"].values():
        if file_digest(path) != cp["data_sha256"][path]:
            raise ValueError("Frozen test pool changed")
        bindings.append(Path(path))
    hashes = {str(p): file_digest(p) for p in bindings}
    output = root / "confirmation171-v1"
    # A fixed destination prevents silent fresh-seed rerolls or overwritten failures.
    output.mkdir(exist_ok=False)
    campaign.seal(
        output / "protocol.json",
        {
            "seed": SEED,
            "episodes_per_family": EPISODES,
            "models": chosen,
            "input_sha256": hashes,
            "ordinary_shop_tier": 6,
            "pets": 60,
            "max_turns": 40,
            "new_episodes_same_frozen_opponent_pools": True,
            "no_model_selection_or_training": True,
            "not_a_test_of_new_opponent_strategies_or_official_client_parity": True,
        },
    )
    results = {}
    try:
        for name in map(str, campaign.SEEDS):
            checkpoint = chosen[name]["checkpoint"]
            if file_digest(checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("Checkpoint changed after selection")
            model = MaskablePPO.load(checkpoint["path"], device="cpu")
            if policy_digest(model.policy) != checkpoint["policy_sha256"]:
                raise ValueError("Policy tensors changed")
            options = policy_environment_options(model)
            game, catalog = options["config"], options["catalog"]
            if (
                catalog.catalog_id != "turtle-v0.46-full-v8"
                or (game.max_shop_tier, game.max_turns, game.target_wins) != (6, 40, 10)
                or len(catalog.rollable_pet_ids) != 60
            ):
                raise ValueError("Wrong full-pack environment contract")
            print(f"Confirming frozen model {name}: 5 x {EPISODES} new episodes", flush=True)
            result = evaluate_suite(model, cp["paths"]["test"], episodes=EPISODES, seed=SEED)
            for family, evidence in result["families"].items():
                if evidence["league_sha256"] != hashes[cp["paths"]["test"][family]]:
                    raise ValueError("Evaluated pool differs from frozen input")
            write_new(
                output / f"model-{name}.json", {"checkpoint": checkpoint, "evaluation": result}
            )
            results[name] = result
            del model
        campaign.checked(root)
        if any(file_digest(path) != sha for path, sha in hashes.items()):
            raise ValueError("Inputs changed during confirmation")
        summary = summarize(results, SEED)
        write_new(
            output / "summary.json",
            {
                **summary,
                "scope_audit_pending": True,
                "goal_complete": False,
                "protocol_sha256": file_digest(output / "protocol.json"),
                "limitation": "Fresh episodes against existing frozen pools, "
                "not new opponent policies. "
                "The 80% criterion is a point estimate, not an 80% confidence lower bound.",
            },
        )
        print(f"Confirmation: {summary['learned_ten_win_rate']:.3%}; passed={summary['passed']}")
    except Exception as error:
        write_new(output / "failure.json", {"error": repr(error), "finished_models": list(results)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    confirm(parser.parse_args().campaign.resolve())
