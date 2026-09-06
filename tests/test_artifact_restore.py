import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "restore_artifacts", Path(__file__).resolve().parents[1] / "scripts/restore_artifacts.py"
)
restore_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore_module)


def bundle(folder, member_name="runs/example/result.json", symlink=False):
    data = b'{"wins": 10}\n'
    record = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    path = folder / "example.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size, info.mtime = len(data), 1700000000.125
        if symlink:
            info.type, info.linkname, info.size = tarfile.SYMTYPE, "/tmp", 0
        archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("public-export/example.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    manifest = {
        "bundles": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "files": {member_name: record},
            }
        }
    }
    (folder / "artifact-manifest.json").write_text(json.dumps(manifest))
    return path


def test_restore_verifies_and_preserves_mtime(tmp_path):
    bundle(tmp_path)
    destination = tmp_path / "restored"
    assert restore_module.restore(tmp_path, destination) == {
        "bundles_verified": 1,
        "files_restored": 2,
    }
    result = destination / "runs/example/result.json"
    assert json.loads(result.read_text()) == {"wins": 10}
    assert result.stat().st_mtime == 1700000000.125
    with pytest.raises(FileExistsError):
        restore_module.restore(tmp_path, destination)


def test_restore_rejects_corruption_before_writing(tmp_path):
    path = bundle(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        restore_module.restore(tmp_path, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "unlisted/result"])
def test_restore_rejects_unsafe_names(tmp_path, name):
    bundle(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        restore_module.restore(tmp_path, tmp_path / "restored")


def test_restore_rejects_symlink_member(tmp_path):
    bundle(tmp_path, symlink=True)
    with pytest.raises(ValueError, match="non-file"):
        restore_module.restore(tmp_path, tmp_path / "restored")


def test_restore_rejects_destination_symlink_escape(tmp_path):
    bundle(tmp_path)
    destination, outside = tmp_path / "restored", tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        restore_module.restore(tmp_path, destination)
    assert not list(outside.iterdir())
