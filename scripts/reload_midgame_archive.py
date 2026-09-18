"""Reload the validation-selected v5 delivery using its trusted frozen source.

Reuses the existing checked archive extractor and isolated inference worker.
Only local project-produced archives/models are trusted. Never trains or tests.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from reload_archived_pilot import WORKER, extract_verified

from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.expanded_confirmation import compare_reload, read, save_new
from sap_rl_lab.midgame_confirmation import check_protocol


def validation_paths(root, protocol):
    paths = {}
    for family, original in protocol["paths"]["validation"].items():
        path = Path(original).resolve()
        relative = path.relative_to(root)
        if relative.parts[0] != "data":
            raise ValueError("Validation pool must belong to the experiment's data directory")
        if file_digest(path) != protocol["data_sha256"][str(relative)]:
            raise ValueError("Validation pool changed")
        paths[family] = str(path)
    return paths


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = check_protocol(root)
    selection = read(root / "selection.json")
    if selection["protocol_sha256"] != file_digest(root / "protocol.json"):
        raise ValueError("Protocol changed after selection")
    name = selection["delivery"]
    item = selection["models"][name]
    if file_digest(item["path"]) != item["sha256"]:
        raise ValueError("Selected model changed")
    expected_path = root / name / item["selected"]["evaluation_file"]
    expected = read(expected_path)
    pools = validation_paths(root, protocol)
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="sap-midgame-archived-reload-") as temporary:
        source = Path(temporary) / "source"
        extract_verified(root / "source.zip", protocol["source_files_sha256"], source)
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1")
        with (output / "reload.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    WORKER,
                    str(source),
                    item["path"],
                    json.dumps(pools),
                    str(protocol["base_config"]["validation_episodes"]),
                    str(protocol["base_config"]["validation_seed"]),
                    str(output / "validation.json"),
                ],
                cwd=temporary,
                env=environment,
                stdout=log,
                stderr=log,
                check=True,
                timeout=1200,
            )
        actual = read(output / "validation.json")
        compare_reload(actual, expected)
    check_protocol(root)
    save_new(
        output / "summary.json",
        {
            "all_validation_rows_exact": True,
            "model": name,
            "model_sha256": item["sha256"],
            "source_archive_sha256": file_digest(root / "source.zip"),
            "selection_sha256": file_digest(root / "selection.json"),
            "protocol_sha256": file_digest(root / "protocol.json"),
            "expected_validation_sha256": file_digest(expected_path),
            "validation_pool_sha256": {f: file_digest(p) for f, p in pools.items()},
            "helper_sha256": file_digest(__file__),
            "rows": sum(len(f["episode_results"]) for f in actual["families"].values()),
            "training_updates": 0,
            "held_out_tests_opened": False,
            "note": "Exact archived-source validation with current installed dependencies.",
        },
    )
    print(f"Archived source exactly reproduced delivery validation: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.output)
