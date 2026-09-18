"""Validation-only counterfactual check of the unresolved long-battle threshold.

This does not select a game rule or a checkpoint, train, or open held-out tests.
Run only after the requested pilot arm has finished and its best model is frozen.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from sap_rl_lab.catalog import load_catalog_by_id
from sap_rl_lab.evaluation import compact_evaluation, evaluate_suite, file_digest
from sap_rl_lab.expanded import verify_inputs
from sap_rl_lab.expanded_confirmation import compare_reload, latest_selected, save_new


def changed_episode_seeds(reference, counterfactual):
    if set(reference["families"]) != set(counterfactual["families"]):
        raise ValueError("Unpaired families")
    changed = {}
    for family, result in reference["families"].items():
        a = result["episode_results"]
        b = counterfactual["families"][family]["episode_results"]
        if [row["seed"] for row in a] != [row["seed"] for row in b]:
            raise ValueError("Unpaired episode seeds")
        changed[family] = [left["seed"] for left, right in zip(a, b) if left != right]
    return changed


def run(run_dir, arm, output):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if arm not in {"control", "action_cost", "swap_cost"}:
        raise ValueError("Use a named completed pilot arm")
    protocol = json.loads((run_dir / "protocol.json").read_text())
    verify_inputs(run_dir, protocol)
    complete = json.loads((run_dir / f"{arm}_complete.json").read_text())
    model_path = run_dir / arm / "best_model.zip"
    model_hash = file_digest(model_path)
    if not complete["training_complete"] or model_hash != complete["best_model_sha256"]:
        raise ValueError("Requires an unchanged, completed pilot model")
    selected = latest_selected(json.loads((run_dir / arm / "validation_history.json").read_text()))
    expected = json.loads((run_dir / arm / selected["evaluation_file"]).read_text())
    catalog = load_catalog_by_id(protocol["catalog_id"])
    if catalog.battle_attack_limit != 30:
        raise ValueError("This predeclared diagnostic is for the provisional 30-exchange rule")
    output.mkdir(parents=True, exist_ok=False)
    model = MaskablePPO.load(model_path, device="cpu")
    results = {}
    for limit in (30, 40, 50):
        results[limit] = evaluate_suite(
            model,
            protocol["paths"]["validation"],
            episodes=protocol["base_config"]["validation_episodes"],
            seed=protocol["base_config"]["validation_seed"],
            # Deliberate diagnostic intervention after the saved base contract
            # is verified, not an attempt to relabel a different training rule.
            env_kwargs={"catalog": replace(catalog, battle_attack_limit=limit)},
        )
        save_new(output / f"validation-limit-{limit}.json", results[limit])
        if limit == 30:
            compare_reload(results[limit], expected)
    verify_inputs(run_dir, protocol)
    if file_digest(model_path) != model_hash:
        raise ValueError("Model changed during diagnostic")
    summary = {
        "model_sha256": model_hash,
        "audit_script_sha256": file_digest(__file__),
        "protocol_sha256": file_digest(run_dir / "protocol.json"),
        "training_updates": 0,
        "held_out_tests_opened": False,
        "rule_or_checkpoint_selected": False,
        "selected_validation_reload_exact": True,
        "changed_episode_seeds": {
            str(limit): changed_episode_seeds(results[30], results[limit]) for limit in (40, 50)
        },
        "results": {str(limit): compact_evaluation(result) for limit, result in results.items()},
        "limitation": "Conditional sensitivity of one fixed policy on its reused validation "
        "suite. Equal rows do not prove identical traces or official-client parity. Different "
        "scores are not a reason to choose whichever rule gives the highest score.",
    }
    save_new(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.arm, args.output)
