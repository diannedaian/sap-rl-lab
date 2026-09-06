"""Independently audit completed Round 3 budgets, selections, pools and test rows."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from sap_rl_lab.evaluation import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    root = Path(args.run_directory).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    source_root = Path(__file__).resolve().parents[1]
    for relative, digest in protocol["source_files_sha256"].items():
        assert file_digest(str(source_root / relative)) == digest
    for path, digest in protocol["data_sha256"].items():
        assert file_digest(str(root / path)) == digest
    initial_hashes, runtimes, decisions = {}, {}, {}
    for seed in protocol["seeds"]:
        pair = []
        for arm in protocol["arms"]:
            name = f"{arm}-seed{seed}"
            directory = root / name
            manifest = json.loads((directory / "run_manifest.json").read_text())
            config = manifest["config"]
            assert manifest["initialization"] == "from_scratch"
            assert not config["initialize_from"]
            assert config["seed"] == seed
            assert config["action_cost"] == 0.0 and not config["forfeit_on_limit"]
            assert config["opponent_leagues"] == protocol["arms"][arm]
            assert config["validation_leagues"] == protocol["paths"]["validation"]
            for key, value in protocol["base_config"].items():
                if key not in {
                    "seed",
                    "output_dir",
                    "opponent_leagues",
                    "expected_initial_policy_sha256",
                }:
                    assert config[key] == value, (name, key)
            pair.append(manifest["initial_policy_sha256"])
            history = json.loads((directory / "validation_history.json").read_text())
            for entry in history:
                saved = json.loads((directory / entry["evaluation_file"]).read_text())
                assert saved["success_rate"] == entry["success_rate"]
                assert set(saved["families"]) == {"greedy", "stats", "summon"}
                assert math.isclose(
                    saved["success_rate"],
                    fmean(item["success_rate"] for item in saved["families"].values()),
                )
                assert math.isclose(
                    saved["mean_return"],
                    fmean(item["mean_return"] for item in saved["families"].values()),
                )
            winner = max(
                enumerate(history),
                key=lambda row: (row[1]["success_rate"], row[1]["mean_return"], -row[0]),
            )[1]
            assert winner["selected"]
            record = selection[name]
            assert winner["timesteps"] == record["timesteps"]
            assert file_digest(record["path"]) == record["sha256"]
            decisions[name] = history[-1]["timesteps"]
            rollout = config["environments"] * config["rollout_steps"]
            assert decisions[name] == math.ceil(config["timesteps"] / rollout) * rollout
            runtimes[name] = json.loads((directory / "runtime.json").read_text())["seconds"]
        assert pair[0] == pair[1]
        initial_hashes[str(seed)] = pair[0]
    assert len(set(initial_hashes.values())) == len(protocol["seeds"])
    rows_checked = 0
    for name, result in summary["results"].items():
        for family, compact in result["families"].items():
            report = json.loads((root / f"test-{name}-{family}.json").read_text())
            expected_model = (
                selection[name]["sha256"]
                if name in selection
                else protocol["champion_sha256"]
                if name == "round2"
                else None
            )
            league = (
                protocol["paths"]["challenge"]
                if family == "round2_challenge"
                else protocol["paths"]["test"]
            )[family]
            assert report["model_sha256"] == expected_model
            assert report["league_sha256"] == file_digest(league)
            rows = report["episode_results"]
            seed = protocol[
                "challenge_episode_seed" if family == "round2_challenge" else "test_episode_seed"
            ]
            assert [row["seed"] for row in rows] == list(range(seed, seed + len(rows)))
            assert len(rows) == protocol["test_episodes_per_family"]
            assert all(row["success"] == (row["wins"] >= 10) for row in rows)
            assert math.isclose(fmean(row["success"] for row in rows), compact["success_rate"])
            assert math.isclose(fmean(row["return"] for row in rows), compact["mean_return"])
            assert math.isclose(fmean(row["truncated"] for row in rows), compact["truncation_rate"])
            rows_checked += len(rows)
        assert math.isclose(
            result["success_rate"],
            fmean(result["families"][f]["success_rate"] for f in ("greedy", "stats", "summon")),
        )
    report = {
        "all_checks_passed": True,
        "test_rows_checked": rows_checked,
        "initial_policy_sha256_by_seed": initial_hashes,
        "decisions": decisions,
        "training_validation_seconds": runtimes,
        "total_training_validation_seconds": sum(runtimes.values()),
        "total_training_decisions": sum(decisions.values()),
        "aspirational_target": {
            "every_mixed_seed_beats_greedy_primary": all(
                summary["results"][f"mixed-seed{seed}"]["success_rate"]
                > summary["results"]["greedy"]["success_rate"]
                for seed in protocol["seeds"]
            ),
            "every_mixed_seed_below_one_percent_primary_cutoffs": all(
                summary["results"][f"mixed-seed{seed}"]["truncation_rate"] < 0.01
                for seed in protocol["seeds"]
            ),
        },
    }
    target = root / "audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
