"""Check frozen inputs, paired training, raw test accounting, and saved-policy replay."""

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from statistics import fmean

from sap_rl_lab.evaluation import evaluate_policy, file_digest


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    root = Path(args.run_directory).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert set(selection) == {"swap_cost", "success_bonus"}
    assert protocol["catalog_id"] == "turtle-v0.46-tier1-rules-v2"
    source = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root / "source.zip") as archive:
        for path, expected in protocol["source_files_sha256"].items():
            assert file_digest(str(source / path)) == expected, path
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected, path
    for path, expected in protocol["data_sha256"].items():
        assert file_digest(str(root / path)) == expected, path
    assert len(set(protocol["data_sha256"].values())) == 9
    assert file_digest(str(root / "initial_model.zip")) == protocol["initial_model_sha256"]
    assert file_digest(protocol["initial_model"]["path"]) == protocol["initial_model_sha256"]
    initial_hashes, steps, initial_scores = [], {}, []
    for name in selection:
        manifest = json.loads((root / name / "run_manifest.json").read_text())
        config = manifest["config"]
        expected_config = {**protocol["base_config"], **protocol["arms"][name]}
        for key, expected in expected_config.items():
            if key not in {"output_dir", "expected_initial_policy_sha256"}:
                assert config[key] == expected, (name, key)
        assert config["action_cost"] == 0 and not config["forfeit_on_limit"]
        assert config["observe_episode_actions"] and config["gamma"] == 1
        assert manifest["input_migration"] == "zero_weight_episode_action_input"
        initial_hashes.append(manifest["initial_policy_sha256"])
        history = json.loads((root / name / "validation_history.json").read_text())
        assert history[0]["timesteps"] == 0
        initial_scores.append((history[0]["success_rate"], history[0]["mean_return"]))
        winner = max(
            enumerate(history),
            key=lambda item: (item[1]["success_rate"], item[1]["mean_return"], -item[0]),
        )[1]
        assert winner["timesteps"] == selection[name]["selected_timesteps"]
        assert winner["success_rate"] == selection[name]["validation_success"]
        assert winner["truncation_rate"] == selection[name]["validation_cutoffs"]
        assert file_digest(selection[name]["path"]) == selection[name]["sha256"]
        rollout = config["environments"] * config["rollout_steps"]
        steps[name] = math.ceil(config["timesteps"] / rollout) * rollout
        assert history[-1]["timesteps"] == steps[name] == selection[name]["trained_timesteps"]
        final = MaskablePPO.load(root / name / "final_model.zip", device="cpu")
        assert final.num_timesteps == steps[name]
    assert len(set(initial_hashes)) == 1 and initial_scores[0] == initial_scores[1]
    assert summary["training_runs"] == 2
    assert summary["actual_training_steps"] == sum(steps.values())
    rows_checked, replayed = 0, 0
    for name, compact in summary["results"].items():
        suite = json.loads((root / f"test-{name}.json").read_text())
        model_path = (
            root / "initial_model.zip" if name == "frozen_parent" else Path(selection[name]["path"])
        )
        assert suite["model_sha256"] == file_digest(str(model_path))
        model = MaskablePPO.load(model_path, device="cpu")
        for family, result in suite["families"].items():
            path = protocol["paths"]["test"][family]
            assert result["league_sha256"] == file_digest(path)
            assert result["deterministic"]
            rows = result["episode_results"]
            seed = protocol["test_episode_seed"]
            count = protocol["test_episodes_per_family"]
            assert len(rows) == count
            assert [row["seed"] for row in rows] == list(range(seed, seed + count))
            counts = Counter()
            for row in rows:
                assert row["actions"] == sum(row["action_counts"].values())
                assert row["battles"] == sum(row["battle_counts"].values())
                assert row["wins"] == row["battle_counts"].get("win", 0)
                assert row["success"] == (row["wins"] >= 10)
                assert row["truncated"] == (row["reason"] in {"shop_action_limit", "turn_limit"})
                # Prove no swap cost or success bonus leaked into reported returns.
                raw = row["wins"] - row["battle_counts"].get("loss", 0)
                raw -= int(row["reason"] == "shop_action_limit")
                close(row["return"], raw)
                counts.update(row["action_counts"])
            assert dict(counts) == result["action_counts"]
            for metric, key in (
                ("success_rate", "success"),
                ("truncation_rate", "truncated"),
                ("mean_return", "return"),
                ("mean_episode_actions", "actions"),
            ):
                close(result[metric], fmean(row[key] for row in rows))
                close(result[metric], compact["families"][family][metric])
            repeat = evaluate_policy(model, episodes=5, seed=seed, opponent_league=path)
            assert repeat["episode_results"] == rows[:5], (name, family)
            rows_checked += len(rows)
            replayed += 5
        for metric in ("success_rate", "truncation_rate", "mean_return", "mean_episode_actions"):
            close(compact[metric], fmean(value[metric] for value in suite["families"].values()))
    baseline = summary["results"]["frozen_parent"]
    report = {
        "all_checks_passed": True,
        "test_rows_checked": rows_checked,
        "saved_policy_episodes_replayed": replayed,
        "training_steps": steps,
        "shared_initial_policy_sha256": initial_hashes[0],
        "raw_evaluation_reward_accounting_verified": True,
        "screening_targets": {
            name: {
                "cutoffs_below_one_percent": summary["results"][name]["truncation_rate"] < 0.01,
                "success_within_one_point_of_parent": summary["results"][name]["success_rate"]
                >= baseline["success_rate"] - 0.01,
            }
            for name in selection
        },
    }
    with (root / "audit.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
