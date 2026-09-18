"""Safe archive extraction for the trusted-local historical reproduction helper."""

import hashlib
import runpy
import stat
import zipfile
from pathlib import Path

import pytest

HELPER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/reload_archived_pilot.py")
)


def test_verified_extraction_preserves_contents_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "source.zip"
    contents = b"# preserved source\n"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("src/sap_rl_lab/example.py", contents)
    expected = {"src/sap_rl_lab/example.py": hashlib.sha256(contents).hexdigest()}
    destination = tmp_path / "extracted"
    HELPER["extract_verified"](archive, expected, destination)
    assert (destination / "src/sap_rl_lab/example.py").read_bytes() == contents
    with pytest.raises(FileExistsError):
        HELPER["extract_verified"](archive, expected, destination)


@pytest.mark.parametrize(
    "problem", ["traversal", "absolute", "backslash", "symlink", "hash", "extra"]
)
def test_bad_archive_rejected_before_extracting_anything(tmp_path, problem):
    name = {
        "traversal": "src/../../outside.py",
        "absolute": "/src/example.py",
        "backslash": "src\\example.py",
    }.get(problem, "src/example.py")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        member = zipfile.ZipInfo(name)
        if problem == "symlink":
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(member, b"data")
        if problem == "extra":
            handle.writestr("src/unlisted.py", b"data")
    expected = {name: "wrong" if problem == "hash" else hashlib.sha256(b"data").hexdigest()}
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError):
        HELPER["extract_verified"](archive, expected, destination)
    assert not destination.exists()
