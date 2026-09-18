"""Export the frozen full-pack champion for NumPy/Pyodide inference, never train.

Only public inference weights and a byte-for-byte copy of the rules are shipped.
The observation and settlement methods are extracted from the frozen engine,
not independently reimplemented. Run tests/test_browser_duel.py after export.
"""

import ast
import hashlib
import io
import json
import textwrap
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from sap_rl_lab.fullpack.catalog import catalog_digest
from sap_rl_lab.fullpack.env import SapAutoBattlerEnv
from sap_rl_lab.fullpack.evaluation import policy_environment_options

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "viewer/duel"
CHECKPOINT = ROOT / (
    "runs/fullpack-tier6-h40-v1/segments/segment004/candidate-85101/eval004-step1048576.zip"
)
EXPECTED_SHA = "0afbbec0ae31911f22f6efc85b4249f3d60915a46acfeefdcc7f58c697c21a0b"
RULE_MODULES = (
    "actions",
    "catalog",
    "domain",
    "engine",
    "shop",
    "events",
    "midgame_events",
    "tier4_events",
    "late_events",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def method(source, class_name, name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def generated_adapter():
    source_dir = ROOT / "src/sap_rl_lab/fullpack"
    observation = method((source_dir / "env.py").read_text(), "SapAutoBattlerEnv", "_observation")
    end = method((source_dir / "engine.py").read_text(), "AutoBattler", "_end_turn")
    # The original settlement starts after end-turn effects/opponent/battle resolution.
    start = next(
        i
        for i, n in enumerate(end.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "outcome"
    )
    settlement = "\n".join(ast.unparse(n) for n in end.body[start:])
    return (
        '"""Generated from frozen fullpack methods; do not hand edit."""\n'
        "from __future__ import annotations\nimport numpy as np\n"
        "from .domain import BattleOutcome\nfrom .engine import Transition\n\n"
        "class Observation:\n" + textwrap.indent(ast.unparse(observation), "    ") + "\n\n"
        "def finish_battle(self, result):\n"
        "    self.last_battle = result\n    opponent = []\n"
        + textwrap.indent(settlement, "    ")
        + "\n"
    )


def main():
    assert digest(CHECKPOINT.read_bytes()) == EXPECTED_SHA, "Frozen checkpoint changed"
    torch.set_num_threads(1)
    model = MaskablePPO.load(CHECKPOINT, device="cpu")
    options = policy_environment_options(model)
    catalog, config = options["catalog"], options["config"]
    assert catalog.catalog_id == "turtle-v0.46-full-v8" and config.max_shop_tier == 6
    assert list(model.observation_space.spaces) == ["global", "shop", "team"]
    assert str(model.policy.activation_fn) == "<class 'torch.nn.modules.activation.Tanh'>"
    weights = {
        key: value.detach().cpu().numpy() for key, value in model.policy.state_dict().items()
    }
    expected_keys = {
        f"mlp_extractor.{net}.{layer}.{part}"
        for net in ("policy_net", "value_net")
        for layer in (0, 2)
        for part in ("weight", "bias")
    } | {f"{net}.{part}" for net in ("action_net", "value_net") for part in ("weight", "bias")}
    assert set(weights) == expected_keys
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **weights)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "model.npz").write_bytes(buffer.getvalue())
    fixtures = {
        k: [] for k in ("global", "shop", "team", "mask", "action", "probabilities", "value")
    }
    env = SapAutoBattlerEnv(**options)
    obs, _ = env.reset(seed=172000000)
    for i in range(408):
        mask = env.action_masks()
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            distribution = model.policy.get_distribution(tensor, action_masks=mask)
            probabilities = distribution.distribution.probs.cpu().numpy()[0]
            value = float(model.policy.predict_values(tensor).item())
        action = int(np.argmax(probabilities))
        if i % 17 == 0:
            for key in obs:
                fixtures[key].append(obs[key].copy())
            for key, data in (
                ("mask", mask),
                ("action", action),
                ("probabilities", probabilities),
                ("value", value),
            ):
                fixtures[key].append(data)
        obs, _, done, cutoff, _ = env.step(action)
        if done or cutoff:
            obs, _ = env.reset(seed=172000001 + i)
    np.savez_compressed(OUTPUT / "parity.npz", **{k: np.array(v) for k, v in fixtures.items()})
    raw = asdict(catalog)
    raw["pets"] = [
        {
            **asdict(p),
            "abilities": [
                {"trigger": a.trigger, "effect": a.effect, **a.params} for a in p.abilities
            ],
        }
        for p in catalog.pets.values()
    ]
    raw["foods"] = [
        {**{k: v for k, v in asdict(f).items() if k != "params"}, **f.params}
        for f in catalog.foods.values()
    ]
    (OUTPUT / "catalog.json").write_text(json.dumps(raw, separators=(",", ":")))
    files = {"sap_web/__init__.py": b""}
    hashes = {}
    for name in RULE_MODULES:
        data = (ROOT / f"src/sap_rl_lab/fullpack/{name}.py").read_bytes()
        files[f"sap_web/{name}.py"] = data
        hashes[name] = digest(data)
    files["sap_web/generated.py"] = generated_adapter().encode()
    files["sap_web/duel.py"] = (OUTPUT / "duel_runtime.py").read_bytes()
    with zipfile.ZipFile(OUTPUT / "runtime.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    manifest = {
        "schema_version": 1,
        "model": "A · 85101",
        "checkpoint_sha256": EXPECTED_SHA,
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog_digest(catalog),
        "config": asdict(config),
        "observation_order": list(model.observation_space.spaces),
        "observation_shapes": {k: list(v.shape) for k, v in model.observation_space.spaces.items()},
        "rule_sha256": hashes,
        "parameters": sum(x.size for x in weights.values()),
        "inference": "float32 NumPy, Tanh MLP, legal-action mask, deterministic argmax",
        "validation_ten_win_rate": 0.8375,
        "notice": "Frozen 0.46 sandbox, not official-client parity. Duel is not Arena evaluation.",
        "files": {
            name: digest((OUTPUT / name).read_bytes())
            for name in ("model.npz", "catalog.json", "runtime.zip", "parity.npz")
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {manifest['parameters']:,} parameters; checkpoint SHA verified")


if __name__ == "__main__":
    main()
