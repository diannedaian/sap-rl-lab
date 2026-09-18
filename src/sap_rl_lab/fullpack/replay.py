"""Save, inspect, and deterministically verify complete episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .catalog import catalog_digest, load_catalog_by_id
from .domain import GameConfig
from .engine import AutoBattler, Transition


@dataclass(frozen=True)
class ReplayStep:
    action_id: int
    action: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass(frozen=True)
class EpisodeReplay:
    schema_version: int
    catalog_id: str
    seed: Optional[int]
    config: Dict[str, Any]
    steps: List[ReplayStep]
    final_state: Dict[str, Any]
    catalog_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "seed": self.seed,
            "config": self.config,
            "steps": [asdict(step) for step in self.steps],
            "final_state": self.final_state,
        }
        if self.catalog_sha256 is not None:
            result["catalog_sha256"] = self.catalog_sha256
        return result

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> EpisodeReplay:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=int(data["schema_version"]),
            catalog_id=str(data["catalog_id"]),
            seed=data["seed"],
            config=dict(data["config"]),
            steps=[ReplayStep(**step) for step in data["steps"]],
            final_state=dict(data["final_state"]),
            catalog_sha256=data.get("catalog_sha256"),
        )


class ReplayRecorder:
    """Thin recorder around an engine; it does not alter transition behavior."""

    def __init__(self, engine: AutoBattler, seed: Optional[int]) -> None:
        self.engine = engine
        self.seed = seed
        self.steps: List[ReplayStep] = []
        self.engine.reset(seed=seed)

    def step(self, action_id: int) -> Transition:
        description = self.engine.codec.describe(action_id)
        transition = self.engine.step_id(action_id)
        self.steps.append(
            ReplayStep(
                action_id=action_id,
                action=description,
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=transition.info,
            )
        )
        return transition

    def finish(self) -> EpisodeReplay:
        return EpisodeReplay(
            schema_version=1,
            catalog_id=self.engine.catalog.catalog_id,
            seed=self.seed,
            config=asdict(self.engine.config),
            steps=list(self.steps),
            final_state=self.engine.state.to_dict(),
            catalog_sha256=catalog_digest(self.engine.catalog)
            if self.engine.expanded_rules
            else None,
        )


def verify_replay(replay: EpisodeReplay, engine: Optional[AutoBattler] = None) -> None:
    """Raise a useful error if rerunning a replay produces different output."""

    recorded_config = GameConfig(**replay.config)
    candidate = engine or AutoBattler(
        catalog=load_catalog_by_id(replay.catalog_id), config=recorded_config
    )
    if replay.schema_version != 1:
        raise ValueError(f"Unsupported replay schema: {replay.schema_version}")
    if candidate.catalog.catalog_id != replay.catalog_id:
        raise ValueError(
            f"Replay needs catalog {replay.catalog_id}, got {candidate.catalog.catalog_id}"
        )
    if asdict(candidate.config) != asdict(recorded_config):
        raise ValueError("Replay engine configuration does not match the recorded configuration")
    if replay.catalog_sha256 and replay.catalog_sha256 != catalog_digest(candidate.catalog):
        raise ValueError("Replay catalog content changed; use the archived source/catalog")

    recorder = ReplayRecorder(candidate, replay.seed)
    for expected in replay.steps:
        recorder.step(expected.action_id)
    actual = recorder.finish()
    if replay.catalog_sha256 is None:
        # Legacy payloads predate fingerprints. Compare their original schema,
        # without claiming their absent content fingerprint was verified.
        from dataclasses import replace

        actual = replace(actual, catalog_sha256=None)
    expected = replay.to_dict()
    # Missing fields in legacy recordings have their historical default values.
    expected["config"] = asdict(recorded_config)
    if actual.to_dict() != expected:
        raise AssertionError("Replay output differs from the recorded episode")
