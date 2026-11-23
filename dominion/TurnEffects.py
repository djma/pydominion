"""Turn-scoped ongoing effects and Duration/Throne-room stay-in-play tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dominion.Card import Card


###############################################################################
@dataclass
class CostModifier:
    """Turn-scoped coin cost reduction (e.g. Bridge, Highway, Quarry, Inventor, Bridge Troll).

    Registered when a card's play ability resolves so replaying with Throne Room stacks
    cumulatively and trashing the source card (e.g. Counterfeit + Quarry) preserves the effect.
    """

    source_name: str
    delta: int  # Negative for cost reduction (e.g. -1, -2)
    card_filter: Optional[str] = None  # None = all cards, "action" = Action cards only, etc.
    persists_next_turn: bool = False  # e.g. Bridge Troll

    def applies_to(self, card: "Card") -> bool:
        if self.card_filter is None:
            return True
        if self.card_filter == "action":
            return card.isAction()
        if self.card_filter == "treasure":
            return card.isTreasure()
        if self.card_filter == "victory":
            return card.isVictory()
        return True


###############################################################################
@dataclass
class CardTrack:
    """Implements Dominion's Lose-Track / Stop-Moving rule (Ch. III p. 26; 2021 & 2025 errata).

    An ability expects a card to remain in `expected_location` without having moved away
    (even if moved back). Every zone transition or top-of-deck cover/shuffle increments
    `card.move_epoch`.
    """

    card: "Card"
    expected_location: Any
    expected_epoch: int

    @classmethod
    def capture(cls, card: "Card", expected_location: Any = None) -> "CardTrack":
        loc = card.location if expected_location is None else expected_location
        return cls(card=card, expected_location=loc, expected_epoch=card.move_epoch)

    def is_valid(self) -> bool:
        """True if the card has never left its expected location since capture."""
        return self.card.location == self.expected_location and self.card.move_epoch == self.expected_epoch


###############################################################################
@dataclass
class DurationStay:
    """Tracks an in-play Duration card (and any Throne-Room that directly played it)."""

    card_uuid: str
    card_epoch: int
    triggers_remaining: int = 1  # Number of times duration() triggers at start of next turn
    permanent: bool = False  # Stays in play for rest of game / until condition clears
    stays_until_condition: bool = False  # e.g. next-time Durations (Abundance, Cage, Search, etc.)
    attached_throne_uuids: list[tuple[str, int]] = field(default_factory=list)  # (throne_uuid, epoch)
    active: bool = True  # Set False if card loses track (2025 Duration-stopping rule)


###############################################################################
class TurnEffects:
    """Container for turn-scoped ongoing effects that survive card movement/trashing."""

    def __init__(self) -> None:
        self.cost_modifiers: list[CostModifier] = []
        self.next_turn_cost_modifiers: list[CostModifier] = []
        self.extra_effects: dict[str, Any] = {}

    def add_cost_modifier(
        self,
        source_name: str,
        delta: int,
        card_filter: Optional[str] = None,
        persists_next_turn: bool = False,
    ) -> None:
        mod = CostModifier(
            source_name=source_name,
            delta=delta,
            card_filter=card_filter,
            persists_next_turn=persists_next_turn,
        )
        self.cost_modifiers.append(mod)
        if persists_next_turn:
            self.next_turn_cost_modifiers.append(
                CostModifier(source_name=source_name, delta=delta, card_filter=card_filter, persists_next_turn=False)
            )

    def total_cost_delta(self, card: "Card") -> int:
        return sum(mod.delta for mod in self.cost_modifiers if mod.applies_to(card))

    def start_turn(self) -> None:
        """Promote any next-turn modifiers (e.g. Bridge Troll) into this turn."""
        self.cost_modifiers = list(self.next_turn_cost_modifiers)
        self.next_turn_cost_modifiers = []
        self.extra_effects = {}

    def end_turn(self) -> None:
        """Clear end-of-turn modifiers while retaining queued next-turn modifiers."""
        self.cost_modifiers = []
        self.extra_effects = {}


# EOF
