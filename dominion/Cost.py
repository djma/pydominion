"""Multi-dimensional Dominion Cost value object (Coins, Potions, Debt).

Implements Dominion's exact component-wise partial order (Complete Rules Ch. IV & VII):
  - Cost(coins, potions, debt)
  - "up to $X" / "costing at most C": component-wise <= (excludes Potion/Debt if C has 0)
  - "less than C": component-wise <= and !=
  - "exactly C" / "same cost": identical (coins, potions, debt)
  - "$X or more": coins >= X (any potions/debt allowed)
  - Coin cost reductions affect coins only (clamped >= 0), never potions or debt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Cost:
    """Immutable Dominion card/event/project cost."""

    coins: int = 0
    potions: int = 0
    debt: int = 0

    @classmethod
    def from_card(cls, card: Any) -> "Cost":
        """Extract printed base cost from a Card, Event, or Project."""
        coins = max(0, int(getattr(card, "cost", 0)))
        potions = int(bool(getattr(card, "potcost", False)))
        debt = max(0, int(getattr(card, "debtcost", 0)))
        return cls(coins=coins, potions=potions, debt=debt)

    def with_coin_delta(self, delta: int) -> "Cost":
        """Apply a coin cost modifier (clamped at 0; potions & debt untouched)."""
        return Cost(coins=max(0, self.coins + delta), potions=self.potions, debt=self.debt)

    plus_coins = with_coin_delta

    def is_up_to(self, limit: "Cost") -> bool:
        """True if self costs at most `limit` in every dimension (component-wise <=)."""
        return self.coins <= limit.coins and self.potions <= limit.potions and self.debt <= limit.debt

    def is_less_than(self, other: "Cost") -> bool:
        """True if self costs strictly less than `other` in Dominion's partial order."""
        return self.is_up_to(other) and self != other

    def is_more_than(self, other: "Cost") -> bool:
        """True if self costs strictly more than `other` in Dominion's partial order."""
        return other.is_less_than(self)

    def is_at_least_coins(self, min_coins: int) -> bool:
        """True for '$X or more' rules: coins >= min_coins regardless of potions/debt."""
        return self.coins >= min_coins

    def is_in_coin_range(self, min_coins: int, max_coins: int) -> bool:
        """True for 'from $X to $Y' rules: min_coins <= coins <= max_coins and no potions/debt."""
        return min_coins <= self.coins <= max_coins and self.potions == 0 and self.debt == 0

    def can_be_paid_by(self, coins: int, potions: int = 0) -> bool:
        """True if available coins & potions cover this cost (debt is taken on buy)."""
        return self.coins <= coins and self.potions <= potions

    def format_str(self, overpay: bool = False) -> str:
        """Human-readable cost description."""
        parts = [f"{self.coins} Coins"]
        if self.debt:
            parts.append(f"{self.debt} Debt")
        if self.potions:
            parts.append("Potion" if self.potions == 1 else f"{self.potions} Potions")
        if overpay:
            parts.append("Overpay")
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.format_str()


# EOF
