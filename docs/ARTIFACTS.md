# Public experiment artifacts

[Download the eight-pet milestone release](https://github.com/diannedaian/sap-rl-lab/releases/tag/v0.1.0-eight-pet).
Code, reports, viewer source and all 38 selected replays live in Git. Training
artifacts are release attachments, keeping intermediate weights out of Git history.
The hosted viewer's access settings are separate; anyone can run the public viewer
locally using [these instructions](../viewer/README.md#develop).

## What to download

| Attachment | Contents |
| --- | --- |
| `round5-confirmation-v1.tar.gz` | Six confirmation runs, frozen release model, 18,000 test rows, pools, archived source and audit |
| `round4-two-objectives-v1.tar.gz` | Reward experiments and Round 5's protocol dependency |
| `round3-full-v1.tar.gz` | Earlier study and the parent checkpoint for continuation |
| `round2-full-seeded-v2.tar.gz` | Formal second study |
| `initial-baseline.tar.gz` | Initial CPU/GPU comparison and baseline outputs |
| `engineering-history.tar.gz` | Separately labeled preliminary, smoke and superseded runs; not additional benchmark evidence |
| `artifact-manifest.json`, `SHA256SUMS` | Bundle hashes plus original/published file hashes |

For **inference only**, download Round 5 and the two manifest files. For the
Round 5 independent audit or six-run reproduction, also download Rounds 3 and 4.
Download all six bundles to recover the full experiment history.

## Verify and restore

Use a fresh clone. Place downloaded assets in `artifact_staging/downloads/`.
From that download directory, verify `artifact-manifest.json` and each downloaded
bundle against its entry in `SHA256SUMS` (macOS: `shasum -a 256 FILE`; Linux:
`sha256sum FILE`). These are integrity checks, not a separate digital signature.
Then, from the repository root:

```sh
python scripts/restore_artifacts.py --source artifact_staging/downloads --destination .
python -m venv .venv
.venv/bin/python -m pip install -e ".[rl,dev]"
.venv/bin/python scripts/audit_round5.py runs/round5-confirmation-v1 --check-only
```

The restore command verifies every downloaded bundle and every experiment file,
rejects unsafe paths and links, and **refuses to overwrite existing files**. It
preserves recorded modification times. Missing bundles are deliberately skipped.
Audit checks are read-only with `--check-only`; preserved timestamps are consistency
checks, not cryptographic proof of when model selection occurred. Current source
must match the archived hashes; use the release tag for this historical audit.
Use trusted model archives only: Python model loaders may deserialize executable objects.
See [delivery instructions](DELIVERY.md) for frozen-policy inference and retraining.

## Public metadata and local cleanup

Public copies replace personal project prefixes with repository-relative paths and
remove cluster mount prefixes, login hostnames and personal email addresses. They
exclude connection diagnostics and runtime caches. Model ZIPs, source ZIPs,
opponent pools and numerical outcomes are unchanged. Path-bearing metadata hashes
can differ: the manifest records both hashes, and the Round 5 release manifest
references its exported protocol hash. Raw local metadata is not rewritten.

Only byte-identical, uploaded-and-download-verified experiment files are eligible
for local removal. Original metadata that required redaction stays local. Formal
selected/final models, provenance, reports, pools and the working environment stay
available; intermediate checkpoints and old trial outputs can be recovered from
the attachments. Build outputs and caches are regenerable. No shared cluster data
or files outside this project are part of this cleanup.

This is the completed **eight-pet milestone**, not a completed full-pack agent.
The [roadmap](ROADMAP.md) still targets all 60 Turtle Pack pets.
