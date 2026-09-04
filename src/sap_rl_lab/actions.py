"""Stable action IDs with a much smaller space than full-team permutations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class ActionKind(str, Enum):
    END_TURN = "end_turn"
    ROLL = "roll"
    BUY_PET = "buy_pet"
    BUY_FOOD = "buy_food"
    MERGE = "merge"
    SELL = "sell"
    SWAP = "swap_adjacent"
    FREEZE = "toggle_freeze"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    source: int = -1
    target: int = -1


class ActionCodec:
    """Bidirectional mapping whose size is constant across all game states."""

    def __init__(self, max_team_size: int = 5, max_shop_size: int = 5):
        self.max_team_size = max_team_size
        self.max_shop_size = max_shop_size
        self._id_to_action: Tuple[Action, ...] = tuple(self._build_actions())
        self._action_to_id: Dict[Action, int] = {
            action: action_id for action_id, action in enumerate(self._id_to_action)
        }

    def _build_actions(self) -> Iterator[Action]:
        yield Action(ActionKind.END_TURN)
        yield Action(ActionKind.ROLL)
        for shop_index in range(self.max_shop_size):
            yield Action(ActionKind.BUY_PET, source=shop_index)
        for shop_index in range(self.max_shop_size):
            for team_index in range(self.max_team_size):
                yield Action(ActionKind.BUY_FOOD, source=shop_index, target=team_index)
        for shop_index in range(self.max_shop_size):
            for team_index in range(self.max_team_size):
                yield Action(ActionKind.MERGE, source=shop_index, target=team_index)
        for team_index in range(self.max_team_size):
            yield Action(ActionKind.SELL, source=team_index)
        for team_index in range(self.max_team_size - 1):
            yield Action(ActionKind.SWAP, source=team_index, target=team_index + 1)
        for shop_index in range(self.max_shop_size):
            yield Action(ActionKind.FREEZE, source=shop_index)

    @property
    def size(self) -> int:
        return len(self._id_to_action)

    def decode(self, action_id: int) -> Action:
        if not 0 <= int(action_id) < self.size:
            raise ValueError(f"Action ID out of range: {action_id}")
        return self._id_to_action[int(action_id)]

    def encode(self, action: Action) -> int:
        try:
            return self._action_to_id[action]
        except KeyError as exc:
            raise ValueError(f"Action is outside this codec: {action}") from exc

    def describe(self, action_id: int) -> str:
        action = self.decode(action_id)
        arguments = [value for value in (action.source, action.target) if value >= 0]
        suffix = "" if not arguments else ":" + ",".join(map(str, arguments))
        return f"{action.kind.value}{suffix}"
