"""Create privacy-checked, checksummed public experiment bundles; never alter runs."""

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

FORMAL = (
    "round2-full-seeded-v2",
    "round3-full-v1",
    "round4-two-objectives-v1",
    "round5-confirmation-v1",
)
CACHES = {"matplotlib-cache", "__pycache__", ".pytest_cache", ".ruff_cache", "cache"}
PRIVATE = re.compile(rb"/Users/[^/\s]+/|/data/vision/|[\w.-]+\.csail\.mit\.edu|[\w.+-]+@mit\.edu")
SECRET = re.compile(
    rb"gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|"
    rb"sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_bytes(data, label, depth=0):
    """Inspect ZIP members and SB3's base64 metadata without unpickling anything."""
    if PRIVATE.search(data) or SECRET.search(data):
        raise ValueError(f"Private data requires review: {label}")
    if depth < 4 and data.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                check_bytes(archive.read(member), f"{label}:{member}", depth + 1)
    if label.endswith(":data"):
        metadata = json.loads(data)
        for key, value in metadata.items():
            if isinstance(value, dict) and ":serialized:" in value:
                check_bytes(base64.b64decode(value[":serialized:"]), f"{label}:{key}", depth + 1)


def normalize(data, root, label):
    if data.startswith(b"PK\x03\x04"):
        check_bytes(data, label)
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        check_bytes(data, label)
        return data
    text = text.replace(str(root) + "/", "").replace(str(root), ".")
    # Cluster records remain useful without personal mount points or login hosts.
    text = re.sub(r"/data/vision/[^\s\"']*/projects/[^/\s\"']+/?", "remote-project/", text)
    text = re.sub(r"[\w.-]+\.csail\.mit\.edu", "cluster-host", text)
    text = re.sub(r"[\w.+-]+@mit\.edu", "researcher@example.invalid", text)
    result = text.encode()
    check_bytes(result, label)
    return result


def package(root, output):
    output.mkdir(parents=True, exist_ok=False)
    groups = {name: [root / "runs" / name] for name in FORMAL}
    groups["engineering-history"] = [
        p for p in sorted((root / "runs").iterdir()) if p.is_dir() and p.name not in FORMAL
    ]
    groups["initial-baseline"] = [root / "cluster_results" / "job1697192"]
    inventory = {
        "schema_version": 1,
        "privacy": "Personal project prefixes become repository-relative paths. Cluster mount "
        "prefixes/hostnames are redacted. Binary checkpoints, source ZIPs, numerical results "
        "and opponent-pool contents are not changed. Original raw metadata remains local.",
        "excluded": "Runtime caches and connection_diagnostics (never uploaded).",
        "bundles": {},
    }
    for name, folders in groups.items():
        files = {}
        for folder in folders:
            for path in sorted(folder.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Refuse symlink: {path.relative_to(root)}")
                if not path.is_file() or CACHES.intersection(path.parts):
                    continue
                relative = str(path.relative_to(root))
                original = path.read_bytes()
                files[relative] = (path, original, normalize(original, root, relative))
        # Preserve the release's link to its exported (path-redacted) protocol.
        if name == "round5-confirmation-v1":
            prefix = f"runs/{name}/"
            key = prefix + "release/manifest.json"
            path, original, data = files[key]
            metadata = json.loads(data)
            metadata["protocol_sha256"] = digest(files[prefix + "protocol.json"][2])
            data = (json.dumps(metadata, indent=2) + "\n").encode()
            files[key] = (path, original, data)
        entries = {
            relative: {
                "original_sha256": digest(original),
                "sha256": digest(data),
                "size": len(data),
                "metadata_redacted": original != data,
            }
            for relative, (_, original, data) in files.items()
        }
        export = {"bundle": name, "privacy": inventory["privacy"], "files": entries}
        path = output / f"{name}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for relative, (source, _, data) in files.items():
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = source.stat().st_mtime
                archive.addfile(info, io.BytesIO(data))
            data = (json.dumps(export, indent=2) + "\n").encode()
            info = tarfile.TarInfo(f"public-export/{name}.json")
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        record = {
            "sha256": digest(path.read_bytes()),
            "size": path.stat().st_size,
            "files": entries,
        }
        inventory["bundles"][path.name] = record
        print(
            f"PACKED {name}: {len(files)} files, {record['size']} bytes, "
            f"{sum(e['metadata_redacted'] for e in entries.values())} redacted metadata files",
            flush=True,
        )
    with (output / "artifact-manifest.json").open("x") as handle:
        json.dump(inventory, handle, indent=2)
        handle.write("\n")
    with (output / "SHA256SUMS").open("x") as handle:
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS":
                handle.write(f"{digest(path.read_bytes())}  {path.name}\n")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], Path(args.output).resolve())
