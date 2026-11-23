"""TurnQueue — clockwise turn scheduler with extra turns, 2023 consecutive-turn cap, and Fleet round."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dominion.Player import Player


@dataclass
class TurnEntry:
    """One scheduled or completed turn."""

    player_uuid: str
    is_extra: bool = False
    is_fleet: bool = False
    source: str = ""
    modifiers: dict[str, Any] = field(default_factory=dict)
    enforce_consecutive_cap: bool = True


class TurnQueue:
    """Manages Dominion turn order, extra turns, Fleet post-game round, and turn history.

    Clockwise order follows `player_order`:
      - `player_to_left(p)` is the NEXT player in turn order (`index + 1`).
      - `player_to_right(p)` is the PREVIOUS player in turn order (`index - 1`).
    """

    def __init__(self, player_uuids: Optional[list[str]] = None, random_start: bool = False) -> None:
        self.player_order: list[str] = list(player_uuids) if player_uuids else []
        self.next_regular_index: int = 0
        if self.player_order and random_start:
            self.next_regular_index = random.randrange(len(self.player_order))
        self.extra_queue: list[TurnEntry] = []
        self.fleet_queue: list[TurnEntry] = []
        self.fleet_started: bool = False
        self.history: list[TurnEntry] = []
        self.current_entry: Optional[TurnEntry] = None

    def initialize(self, player_uuids: list[str], random_start: bool = False) -> None:
        """Initialize turn order from player UUIDs."""
        self.player_order = list(player_uuids)
        self.next_regular_index = (
            random.randrange(len(self.player_order)) if (self.player_order and random_start) else 0
        )
        self.extra_queue = []
        self.fleet_queue = []
        self.fleet_started = False
        self.history = []
        self.current_entry = None

    def index_of(self, player_uuid: str) -> int:
        return self.player_order.index(player_uuid)

    def left_uuid(self, player_uuid: str) -> str:
        """Next player clockwise (to the left)."""
        idx = (self.player_order.index(player_uuid) + 1) % len(self.player_order)
        return self.player_order[idx]

    def right_uuid(self, player_uuid: str) -> str:
        """Previous player clockwise (to the right)."""
        idx = (self.player_order.index(player_uuid) - 1) % len(self.player_order)
        return self.player_order[idx]

    def turn_order_from(self, start_uuid: str) -> list[str]:
        """Return all player UUIDs in clockwise turn order starting with `start_uuid`."""
        if not self.player_order:
            return []
        idx = self.player_order.index(start_uuid)
        return self.player_order[idx:] + self.player_order[:idx]

    def consecutive_turns_ending_with(self, player_uuid: str) -> int:
        """Count how many immediately preceding turns (including current if active) belonged to `player_uuid`."""
        count = 0
        if self.current_entry is not None and self.current_entry.player_uuid == player_uuid:
            count += 1
        elif self.current_entry is not None:
            return 0
        for entry in reversed(self.history):
            if entry.player_uuid == player_uuid:
                count += 1
            else:
                break
        return count

    def had_previous_turn(self, player_uuid: str) -> bool:
        """True if the most recently completed turn (before the current turn) belonged to `player_uuid`."""
        return bool(self.history and self.history[-1].player_uuid == player_uuid)

    def can_take_extra_turn(self, player_uuid: str, max_consecutive: int = 2) -> bool:
        """2023 errata: a player cannot take more than `max_consecutive` (2) turns in a row."""
        return self.consecutive_turns_ending_with(player_uuid) < max_consecutive

    def schedule_extra_turn(
        self,
        player_uuid: str,
        source: str = "",
        modifiers: Optional[dict[str, Any]] = None,
        enforce_consecutive_cap: bool = True,
    ) -> bool:
        """Queue an extra turn after the current turn. Returns False if blocked by the 2-turn cap."""
        if enforce_consecutive_cap and not self.can_take_extra_turn(player_uuid):
            return False
        self.extra_queue.append(
            TurnEntry(
                player_uuid=player_uuid,
                is_extra=True,
                is_fleet=False,
                source=source,
                modifiers=dict(modifiers or {}),
                enforce_consecutive_cap=enforce_consecutive_cap,
            )
        )
        return True

    def start_fleet_round(self, last_player_uuid: str, fleet_player_uuids: set[str]) -> bool:
        """Queue one Fleet turn per player with Fleet, in turn order starting after `last_player_uuid`."""
        if self.fleet_started or not fleet_player_uuids:
            return False
        self.fleet_started = True
        order = self.turn_order_from(self.left_uuid(last_player_uuid))
        for uid in order:
            if uid in fleet_player_uuids:
                self.fleet_queue.append(TurnEntry(player_uuid=uid, is_extra=True, is_fleet=True, source="Fleet"))
        return bool(self.fleet_queue)

    def has_pending_fleet_turns(self) -> bool:
        return bool(self.fleet_queue)

    def has_pending_post_game_turns(self) -> bool:
        """Check for Fleet turns or extra turns that can still be taken."""
        self.extra_queue[:] = [
            entry for entry in self.extra_queue
            if not entry.enforce_consecutive_cap or self.can_take_extra_turn(entry.player_uuid)
        ]
        return bool(self.fleet_queue or self.extra_queue)

    def pop_next_turn(self) -> TurnEntry:
        """Take an eligible extra turn, a Fleet turn, or a regular turn."""
        while self.extra_queue:
            entry = self.extra_queue.pop(0)
            if entry.enforce_consecutive_cap and not self.can_take_extra_turn(entry.player_uuid):
                continue
            self.current_entry = entry
            return entry
        if self.fleet_queue:
            entry = self.fleet_queue.pop(0)
        else:
            uid = self.player_order[self.next_regular_index]
            self.next_regular_index = (self.next_regular_index + 1) % len(self.player_order)
            entry = TurnEntry(player_uuid=uid, is_extra=False, is_fleet=False)
        self.current_entry = entry
        return entry

    def finish_current_turn(self) -> None:
        """Record the active turn into history."""
        if self.current_entry is not None:
            self.history.append(self.current_entry)
            self.current_entry = None

    def regular_turn_count(self, player_uuid: str) -> int:
        """Number of regular (non-extra, non-Fleet) turns completed by `player_uuid`."""
        return sum(1 for e in self.history if e.player_uuid == player_uuid and not e.is_extra and not e.is_fleet)


# EOF
