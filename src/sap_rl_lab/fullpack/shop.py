"""Development shop primitives, isolated from the frozen eight-pet rules.

See docs/TIER12_SOURCES.md for evidence and outstanding game-parity checks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .domain import Pet, ShopItem


class ContentNotReady(RuntimeError):
    """Required content is missing; do not silently substitute a lower tier."""


def shop_slots(turn: int) -> Tuple[int, int, int]:
    """Normal pet slots, normal food slots, total including two stock spaces."""
    if turn < 1:
        raise ValueError("turn must be positive")
    pets = 3 if turn < 5 else 4 if turn < 9 else 5
    foods = 1 if turn < 5 else 2
    return pets, foods, pets + foods + 2


def merged_pet(target: Pet, incoming: Pet, *, refresh_triggers: bool = False) -> Pet:
    """Return a new pet; modern XP is highest + 1, not the sum of copies."""
    if target.spec_id != incoming.spec_id or target.experience >= 6:
        raise ValueError("Merge needs matching pets and a non-max-level target")
    result = target.clone()
    result.attack = min(50, max(target.attack, incoming.attack) + 1)
    result.health = min(50, max(target.health, incoming.health) + 1)
    result.experience = min(6, max(target.experience, incoming.experience) + 1)
    result.temporary_attack = max(target.temporary_attack, incoming.temporary_attack)
    result.temporary_health = max(target.temporary_health, incoming.temporary_health)
    result.perk = target.perk or incoming.perk
    result.ability_uses = max(target.ability_uses, incoming.ability_uses)
    if refresh_triggers:
        # Drag direction matters: retain the destination's spent triggers.
        # A genuine level increase instead refreshes the full new-level quota.
        result.ability_uses = (
            0 if result.level > max(target.level, incoming.level) else target.ability_uses
        )
    return result


def stock_items(
    shop: List[Optional[ShopItem]], items: Sequence[ShopItem], active_slots: int, kind: str
) -> None:
    """Stock from the relevant edge, using blanks before unfrozen replacements.

    Reserve destinations as a batch so the second choice cannot overwrite the
    first. Frozen stock is never overwritten; insufficient space drops excess.
    """
    order = list(range(active_slots))
    if kind == "food":
        order.reverse()
    destinations = [i for i in order if shop[i] is None]
    destinations += [i for i in order if shop[i] is not None and not shop[i].frozen]
    for index, item in zip(destinations, items):
        shop[index] = item
