"""Verify public release bundles, then restore without overwriting existing files."""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or not path.parts
        or path.parts[0] not in {"runs", "cluster_results", "public-export"}
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def restore(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = json.loads((source / "artifact-manifest.json").read_text())
    plans, seen = [], set()
    # Preflight every downloaded bundle and every collision before writing anything.
    for name, bundle in manifest["bundles"].items():
        if Path(name).name != name or not name.endswith(".tar.gz"):
            raise ValueError(f"Unsafe bundle name: {name}")
        path = source / name
        if not path.exists():
            continue  # Restore only the bundles the user chose to download.
        if path.stat().st_size != bundle["size"] or sha(path.read_bytes()) != bundle["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
        expected = set(bundle["files"]) | {f"public-export/{name[:-7]}.json"}
        members = set()
        with tarfile.open(path) as archive:
            for member in archive:
                relative = checked_path(member.name)
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise ValueError(f"Unexpected, duplicate or non-file member: {member.name}")
                seen.add(member.name)
                members.add(member.name)
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError(f"Destination escapes through a symlink: {member.name}")
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"Refuse overwrite: {target}")
                data = archive.extractfile(member).read()
                record = bundle["files"].get(member.name)
                if record and (len(data) != record["size"] or sha(data) != record["sha256"]):
                    raise ValueError(f"Member checksum mismatch: {member.name}")
            if members != expected:
                raise ValueError(f"Missing members: {name}")
        plans.append(path)
    if not plans:
        raise ValueError("No manifest-listed bundles found")
    count = 0
    for path in plans:
        # Recheck after preflight in case a downloaded file changed.
        if sha(path.read_bytes()) != manifest["bundles"][path.name]["sha256"]:
            raise ValueError(f"Bundle changed: {path.name}")
        with tarfile.open(path) as archive:
            for member in archive:
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError("Destination changed during restoration")
                with target.open("xb") as handle:
                    handle.write(archive.extractfile(member).read())
                os.utime(target, (member.mtime, member.mtime))
                count += 1
    return {"bundles_verified": len(plans), "files_restored": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Downloaded bundles and artifact-manifest.json"
    )
    parser.add_argument(
        "--destination", required=True, help="Fresh checkout or new restoration folder"
    )
    args = parser.parse_args()
    print(json.dumps(restore(args.source, args.destination), indent=2))
