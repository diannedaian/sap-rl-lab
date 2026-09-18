from dataclasses import asdict

import numpy as np
import pytest

from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig
from sap_rl_lab.engine import default_opponent
from sap_rl_lab.learned_pool import collect


class LegalModel:
    """Test policy choosing the first legal non-END action; does not fake game state."""

    def __init__(self):
        catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
        self.sap_environment_contract = {
            "schema_version": 1,
            "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog_digest(catalog),
            "allow_development": True,
            "game_config": asdict(
                GameConfig.turtle_curriculum(
                    shop_action_limit_mode="force_battle", max_actions_per_turn=1
                )
            ),
        }

    def predict(self, observation, action_masks, deterministic):
        assert deterministic
        # Prefer a purchase over roll (id 1), so snapshots are nonempty.
        candidates = np.flatnonzero(action_masks)
        buy_candidates = candidates[(candidates >= 2) & (candidates < 11)]
        return int(buy_candidates[0] if len(buy_candidates) else candidates[-1]), None


def test_learned_pool_captures_every_forced_battle_and_reproduces():
    model = LegalModel()
    first, rows = collect(model, episodes=3, seed=51000, opponent_provider=default_opponent)
    second, other = collect(model, episodes=3, seed=51000, opponent_provider=default_opponent)
    assert rows == other and first._by_turn == second._by_turn
    assert len(first) == sum(r["battles"] for r in rows)
    assert sum(r["forced_battles"] for r in rows) == len(first)
    assert all(s.pets for bucket in first._by_turn.values() for s in bucket)


def test_learned_pool_rejects_missing_contract_and_nonpositive_count():
    with pytest.raises(ValueError, match="contract"):
        collect(object(), episodes=1, seed=1, opponent_provider=default_opponent)
    with pytest.raises(ValueError, match="positive"):
        collect(LegalModel(), episodes=0, seed=1, opponent_provider=default_opponent)


def test_optional_battle_metadata_does_not_change_trajectories():
    model = LegalModel()
    original, rows = collect(model, episodes=2, seed=58000, opponent_provider=default_opponent)
    metadata = []
    instrumented, detailed = collect(
        model,
        episodes=2,
        seed=58000,
        opponent_provider=default_opponent,
        snapshot_metadata=metadata,
    )
    assert original._by_turn == instrumented._by_turn
    assert len(metadata) == len(original)
    for row in detailed:
        assert sum(row.pop("action_counts").values()) == row["actions"]
    assert rows == detailed
    assert {(s["label"], s["turn"]) for s in metadata} == {
        (s.label, s.turn) for bucket in original._by_turn.values() for s in bucket
    }
    assert all(0 <= s["wins_before_battle"] < 10 for s in metadata)
    assert all(s["lives_before_battle"] > 0 for s in metadata)
