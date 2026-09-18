"""Record a real full test-suite run with exact test/source hashes, never synthesize a pass."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from sap_rl_lab.fullpack.evaluation import file_digest
from sap_rl_lab.fullpack.ppo_guardrails import write_new
from sap_rl_lab.round3 import source_archive

ROOT = Path(__file__).resolve().parents[1]


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    before = {
        str(p.relative_to(ROOT)): file_digest(p)
        for p in (ROOT / "tests").rglob("*")
        if p.is_file() and p.suffix in (".py", ".json")
    }
    sources = source_archive(output)
    started = time.monotonic()
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-o", "addopts="]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    with (output / "pytest.log").open("x") as handle:
        handle.write(result.stdout + result.stderr)
    if any(file_digest(ROOT / name) != sha for name, sha in {**before, **sources}.items()):
        raise ValueError("Inputs changed during tests")
    summary = {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "test_files_sha256": before,
        "source_files_sha256": sources,
        "log_sha256": file_digest(output / "pytest.log"),
    }
    write_new(output / "summary.json", summary)
    print(result.stdout[-2000:] + result.stderr[-500:], flush=True)
    print(json.dumps({k: v for k, v in summary.items() if "sha256" not in k}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output.resolve())
