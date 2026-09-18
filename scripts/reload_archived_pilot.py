"""Reproduce a trusted local pilot's selected validation using its archived rules.

Never load third-party source archives or model pickles with this command. Hashes
detect accidental drift; they are not authentication of an untrusted producer.
No training, test-set evaluation, in-place source replacement, or overwrites.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe relative archive/artifact path")
    return path


def extract_verified(archive_path, expected, destination):
    """Verify the complete manifest and reject unsafe members before any extraction."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Archive members do not exactly match the frozen source manifest")
        payloads = {}
        for member in members:
            path = relative_path(member.filename)
            if not (path.parts[0] == "src" or str(path) == "pyproject.toml"):
                raise ValueError("Unexpected source archive member")
            if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive directories and symlinks are not supported")
            if member.file_size > 20_000_000:
                raise ValueError("Unexpectedly large source member")
            content = archive.read(member)
            if hashlib.sha256(content).hexdigest() != expected[member.filename]:
                raise ValueError("Archived source digest mismatch: " + member.filename)
            payloads[path] = content
        destination.mkdir(parents=True, exist_ok=False)
        for path, content in payloads.items():
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)


WORKER = """
import json, sys
from pathlib import Path
import sap_rl_lab, torch
from sb3_contrib import MaskablePPO
from sap_rl_lab.evaluation import evaluate_suite
source, model_path, pools, count, seed, output = sys.argv[1:]
assert Path(sap_rl_lab.__file__).resolve().is_relative_to(Path(source).resolve())
torch.set_num_threads(1)
model = MaskablePPO.load(model_path, device="cpu")
result = evaluate_suite(model, json.loads(pools), episodes=int(count), seed=int(seed))
with open(output, "x") as handle:
    json.dump(result, handle, indent=2)
"""


def run(run_dir, arm, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    arm_path = relative_path(arm)
    if len(arm_path.parts) != 1:
        raise ValueError("Arm must be a single directory name")
    protocol = read(run_dir / "protocol.json")
    complete = read(run_dir / f"{arm}_complete.json")
    if not complete["training_complete"]:
        raise ValueError("Only completed pilot arms can be reproduced")
    model = run_dir / arm / "best_model.zip"
    if digest(model) != complete["best_model_sha256"]:
        raise ValueError("Completed best model changed")
    history = read(run_dir / arm / "validation_history.json")
    selected = [row for row in history if row["selected"]][-1]
    expected_path = run_dir / arm / relative_path(selected["evaluation_file"])
    expected = read(expected_path)
    pools = {}
    for family in protocol["paths"]["validation"]:
        relative = relative_path(f"data/validation/{family}.json")
        path = run_dir / relative
        if digest(path) != protocol["data_sha256"][str(relative)]:
            raise ValueError("Validation pool changed: " + family)
        pools[family] = str(path)
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="sap-archived-reload-") as temporary:
        source = Path(temporary) / "source"
        extract_verified(run_dir / "source.zip", protocol["source_files_sha256"], source)
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1")
        result_path = output / "validation.json"
        with (output / "reload.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    WORKER,
                    str(source),
                    str(model),
                    json.dumps(pools),
                    str(protocol["base_config"]["validation_episodes"]),
                    str(protocol["base_config"]["validation_seed"]),
                    str(result_path),
                ],
                cwd=temporary,
                env=environment,
                stdout=log,
                stderr=log,
                check=True,
            )
        actual = read(result_path)
        if set(actual["families"]) != set(expected["families"]):
            raise ValueError("Reloaded families differ")
        for family in actual["families"]:
            if (
                actual["families"][family]["episode_results"]
                != expected["families"][family]["episode_results"]
            ):
                raise ValueError("Archived-rule replay differs: " + family)
    summary = {
        "all_validation_rows_exact": True,
        "reproduction_helper_sha256": digest(__file__),
        "source_archive_sha256": digest(run_dir / "source.zip"),
        "protocol_sha256": digest(run_dir / "protocol.json"),
        "model_sha256": digest(model),
        "expected_validation_sha256": digest(expected_path),
        "validation_pool_sha256": {family: digest(path) for family, path in pools.items()},
        "rows": sum(len(item["episode_results"]) for item in actual["families"].values()),
        "training_updates": 0,
        "held_out_tests_opened": False,
        "note": "Executed the exact archived source using the current installed dependencies; "
        "not an evaluation under current rules and not a new benchmark.",
    }
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.arm, args.output)
