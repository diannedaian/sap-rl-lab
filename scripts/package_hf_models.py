"""Export public final checkpoints without changing any tensor/optimizer bytes."""

import hashlib
import importlib.metadata
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from package_public_artifacts import check_bytes
from sb3_contrib import MaskablePPO

from sap_rl_lab.fullpack.env import SapAutoBattlerEnv
from sap_rl_lab.fullpack.evaluation import policy_environment_options


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    selection = json.loads(
        (root / "runs/fullpack-tier6-h40-v1/segments/segment005/selection.json").read_text()
    )["selected"]
    output = root / "artifact_staging/huggingface-fullpack-v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": "DianneDaian/sap-rl-turtle-pack",
        "metadata_change": "learning_rate becomes constant 0.0003; derived lr_schedule omitted. "
        "No other archive member changes. Original local checkpoints are preserved.",
        "models": {},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "stable-baselines3", "sb3-contrib", "gymnasium")
        },
    }
    for letter, (seed, record) in zip("ABC", selection.items()):
        source = Path(record["checkpoint"]["path"])
        original = source.read_bytes()
        assert digest(original) == record["checkpoint"]["sha256"]
        original_model = MaskablePPO.load(source, device="cpu")
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert original_model.lr_schedule(progress) == 0.0003
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as old:
            metadata = json.loads(old.read("data"))
            metadata["learning_rate"] = 0.0003
            metadata.pop("lr_schedule", None)
            with zipfile.ZipFile(buffer, "w") as new:
                for member in old.infolist():
                    data = old.read(member.filename)
                    if member.filename == "data":
                        data = json.dumps(metadata, indent=2).encode()
                    new.writestr(member, data)
        exported = buffer.getvalue()
        name = f"model-{letter}-{seed}.zip"
        check_bytes(exported, name)
        destination = output / name
        if destination.exists() and destination.read_bytes() != exported:
            raise ValueError(f"Refusing to overwrite different export: {name}")
        destination.write_bytes(exported)
        model = MaskablePPO.load(destination, device="cpu")
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination) as new:
            unchanged = {n: digest(old.read(n)) for n in old.namelist() if n != "data"}
            assert all(digest(new.read(n)) == sha for n, sha in unchanged.items())
        env = SapAutoBattlerEnv(**policy_environment_options(model))
        obs, _ = env.reset(seed=42)
        for _ in range(128):
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            reference, _ = original_model.predict(obs, action_masks=masks, deterministic=True)
            assert np.array_equal(action, reference) and masks[int(action)]
            obs, _, done, truncated, _ = env.step(int(action))
            if done or truncated:
                obs, _ = env.reset()
        manifest["models"][name] = {
            "original_sha256": digest(original),
            "public_sha256": digest(exported),
            "bytes": len(exported),
            "unchanged_members": unchanged,
            "legal_prediction_parity_steps": 128,
            "validation_learned_success": record["score"][0],
        }
        print(name, "privacy, tensor integrity and 128-step prediction parity passed", flush=True)
    for name, data in {
        "README.md": (root / "docs/MODEL_CARD.md").read_bytes(),
        "LICENSE": (root / "LICENSE").read_bytes(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
    }.items():
        check_bytes(data, name)
        (output / name).write_bytes(data)
    checksums = "".join(
        f"{digest(path.read_bytes())}  {path.name}\n"
        for path in sorted(output.iterdir())
        if path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(checksums)


if __name__ == "__main__":
    main()
