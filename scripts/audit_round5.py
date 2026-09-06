"""Independently check the six-run confirmation and frozen delivery artifact."""

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from statistics import fmean

from sap_rl_lab.evaluation import evaluate_policy, file_digest
from sap_rl_lab.round5 import ARMS, SEEDS, aggregate, select_delivery


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def audit(root, write_report=True):
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert set(selection) == {f"{arm}-seed{s}" for arm in ARMS for s in SEEDS}
    assert protocol["catalog_id"] == "turtle-v0.46-tier1-rules-v2"
    assert protocol["seeds"] == list(SEEDS) and protocol["arms"] == ARMS
    source = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root / "source.zip") as archive:
        for path, expected in protocol["source_files_sha256"].items():
            assert file_digest(str(source / path)) == expected, path
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected, path
    for path, expected in protocol["data_sha256"].items():
        assert file_digest(str(root / path)) == expected, path
    assert len(set(protocol["data_sha256"].values())) == 9
    previous = Path(protocol["initial_model"]["parent_run"])
    prior = json.loads((previous / "protocol.json").read_text())
    for path, expected in protocol["data_sha256"].items():
        if path.startswith("data/test/"):
            assert expected not in prior["data_sha256"].values()
        else:
            assert expected == prior["data_sha256"][path]
    assert protocol["test_episode_seed"] > prior["test_episode_seed"] + 1000
    assert file_digest(str(root / "initial_model.zip")) == protocol["initial_model_sha256"]
    assert file_digest(protocol["initial_model"]["path"]) == protocol["initial_model_sha256"]
    initial_hashes, steps, initial_scores = [], {}, []
    for name, selected in selection.items():
        manifest = json.loads((root / name / "run_manifest.json").read_text())
        config = manifest["config"]
        expected_config = {
            **protocol["base_config"],
            **ARMS[selected["arm"]],
            "seed": selected["seed"],
        }
        for key, expected in expected_config.items():
            if key not in {"output_dir", "expected_initial_policy_sha256"}:
                assert config[key] == expected, (name, key)
        assert config["action_cost"] == config["success_bonus_max"] == 0
        assert not config["forfeit_on_limit"] and config["observe_episode_actions"]
        assert manifest["input_migration"] == "zero_weight_episode_action_input"
        initial_hashes.append(manifest["initial_policy_sha256"])
        history = json.loads((root / name / "validation_history.json").read_text())
        assert history[0]["timesteps"] == 0
        initial_scores.append((history[0]["success_rate"], history[0]["mean_return"]))
        winner = max(
            enumerate(history), key=lambda x: (x[1]["success_rate"], x[1]["mean_return"], -x[0])
        )[1]
        for field, key in (
            ("selected_timesteps", "timesteps"),
            ("validation_success", "success_rate"),
            ("validation_return", "mean_return"),
            ("validation_cutoffs", "truncation_rate"),
        ):
            assert selected[field] == winner[key]
        assert file_digest(selected["path"]) == selected["sha256"]
        rollout = config["environments"] * config["rollout_steps"]
        steps[name] = math.ceil(config["timesteps"] / rollout) * rollout
        assert history[-1]["timesteps"] == steps[name] == selected["trained_timesteps"]
        final = MaskablePPO.load(root / name / "final_model.zip", device="cpu")
        assert final.num_timesteps == steps[name]
    assert len(set(initial_hashes)) == len(set(initial_scores)) == 1
    assert summary["training_runs"] == 6
    assert summary["actual_training_steps"] == sum(steps.values())
    assert summary["delivery_model"] == select_delivery(selection)
    release = json.loads((root / "release" / "manifest.json").read_text())
    assert release["model"] == summary["delivery_model"]
    assert release["sha256"] == selection[release["model"]]["sha256"]
    assert file_digest(str(root / "release" / "model.zip")) == release["sha256"]
    assert file_digest(str(root / "protocol.json")) == release["protocol_sha256"]
    assert file_digest(str(root / "source.zip")) == release["source_archive_sha256"]
    # Selection and delivery artifacts must precede the first test output.
    first_test = min(p.stat().st_mtime_ns for p in root.glob("test-*.json"))
    assert (root / "selection.json").stat().st_mtime_ns < first_test
    assert (root / "release" / "manifest.json").stat().st_mtime_ns < first_test
    checked = replayed = 0
    for name, compact in summary["results"].items():
        suite = json.loads((root / f"test-{name}.json").read_text())
        assert suite["model_sha256"] == selection[name]["sha256"]
        model = MaskablePPO.load(selection[name]["path"], device="cpu")
        for family, result in suite["families"].items():
            path = protocol["paths"]["test"][family]
            assert result["league_sha256"] == file_digest(path)
            assert result["deterministic"]
            rows = result["episode_results"]
            seed = protocol["test_episode_seed"]
            count = protocol["test_episodes_per_family"]
            assert len(rows) == count
            assert [r["seed"] for r in rows] == list(range(seed, seed + count))
            counts = Counter()
            for row in rows:
                assert row["actions"] == sum(row["action_counts"].values())
                assert row["battles"] == sum(row["battle_counts"].values())
                assert row["wins"] == row["battle_counts"].get("win", 0)
                assert row["success"] == (row["wins"] >= 10)
                assert row["truncated"] == (row["reason"] in {"shop_action_limit", "turn_limit"})
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
            checked += len(rows)
            replayed += 5
        for metric in ("success_rate", "truncation_rate", "mean_return", "mean_episode_actions"):
            close(compact[metric], fmean(v[metric] for v in suite["families"].values()))
    for key, value in aggregate(summary["results"]).items():
        assert summary[key] == value
    report = {
        "all_checks_passed": True,
        "test_rows_checked": checked,
        "saved_policy_episodes_replayed": replayed,
        "training_steps": steps,
        "shared_initial_policy_sha256": initial_hashes[0],
        "raw_evaluation_reward_accounting_verified": True,
        "delivery_selection_frozen_before_test": True,
        "predeclared_checks": summary["predeclared_checks"],
    }
    if write_report:
        with (root / "audit.json").open("x") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--check-only", action="store_true", help="Do not write a new audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.run_directory).resolve(), not args.check_only), indent=2))
