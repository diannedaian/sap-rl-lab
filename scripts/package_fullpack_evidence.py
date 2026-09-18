"""Publish compact, path-redacted final evidence without editing frozen runs."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fullpack-tier6-h40-v1"
OUTPUT = ROOT / "docs/evidence/fullpack-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [
        "delivery_audit.json",
        "reached80_screen.json",
        "segments/segment005/selection.json",
        "confirmation171-v1/summary.json",
        "confirmation171-v1/protocol.json",
        *[f"segments/segment005/test-{seed}.json" for seed in (85101, 85201, 85301)],
        *[f"confirmation171-v1/model-{seed}.json" for seed in (85101, 85201, 85301)],
    ]
    manifest = []
    for relative in paths:
        original = (RUN / relative).read_bytes()
        public = original.decode().replace(str(ROOT) + "/", "").encode()
        assert b"/Users/" not in public and b"/data/vision/" not in public
        json.loads(public)
        name = relative.replace("/", "--")
        compressed = len(public) > 200_000
        if compressed:
            name += ".gz"
        payload = gzip.compress(public, mtime=0) if compressed else public
        target = OUTPUT / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different evidence: {target}")
        target.write_bytes(payload)
        manifest.append(
            {
                "source": f"runs/fullpack-tier6-h40-v1/{relative}",
                "public_file": name,
                "original_sha256": sha(original),
                "redacted_json_sha256": sha(public),
                "public_file_sha256": sha(payload),
                "compressed": compressed,
            }
        )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "note": "Personal workspace prefixes removed from public JSON only. "
                "Numerical rows unchanged. "
                "Recorded historical hashes refer to original files, not redacted copies. "
                "This is final-result evidence, not the complete training archive.",
                "files": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    figures = ROOT / "docs/figures/fullpack-final"
    figures.mkdir(exist_ok=True)
    for source, name in [
        ("runs/fullpack-tier6-curves-final-v1/validation.png", "validation.png"),
        ("runs/fullpack-tier6-bc-curves-final-v1/bc.png", "bc.png"),
    ]:
        (figures / name).write_bytes((ROOT / source).read_bytes())
    print(f"Published {len(manifest)} evidence files and two plots; original runs untouched")


if __name__ == "__main__":
    main()
