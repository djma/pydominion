"""Zone-aware trigger resolution & concurrent ability ordering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from dominion import Piles

if TYPE_CHECKING:
    from dominion.Card import Card
    from dominion.Game import Game
    from dominion.Player import Player


@dataclass
class PendingTrigger:
    """A single triggered ability waiting to be resolved."""

    label: str
    card: "Card"
    callback: Callable[[], Any]
    optional: bool = False


def resolve_concurrent_triggers(
    player: "Player",
    triggers: list[PendingTrigger],
    prompt: str = "Choose which triggered ability to resolve next",
    allow_ordering: bool = True,
) -> list[Any]:
    """Resolve a list of concurrent triggers for `player`.

    Per Complete Rules Ch. III (Timing of concurrent abilities or effects):
      - If 0 or 1 trigger exists, resolve directly without prompting.
      - If 2+ triggers exist and `allow_ordering` is True (and controller isn't using scripted
        non-ordering test inputs), let the player choose which to resolve next.
    """
    results: list[Any] = []
    remaining = list(triggers)
    while remaining:
        if len(remaining) == 1 or not allow_ordering:
            trig = remaining.pop(0)
        else:
            choices = [(trig.label, idx) for idx, trig in enumerate(remaining)]
            chosen_idx = player.plr_choose_options(prompt, *choices)
            trig = remaining.pop(chosen_idx)
        results.append(trig.callback())
    return results


def active_defense_in_play(player: "Player") -> bool:
    """True if player has an ongoing Attack-immunity card IN PLAY (Lighthouse, Guardian, Champion)."""
    in_play = list(player.piles[Piles.PLAYED]) + list(player.piles[Piles.DURATION])
    for card in in_play:
        if card.has_defense() and (card.isDuration() or card.name in ("Lighthouse", "Guardian", "Champion")):
            return True
    return False


def hand_attack_reactions(player: "Player") -> list["Card"]:
    """Return Reaction cards in `player`'s hand that respond to another player playing an Attack."""
    reactions: list[Card] = []
    for card in player.piles[Piles.HAND]:
        if card.isReaction():
            # Defense reactions (Moat, Shield) or cards overriding hook_under_attack
            if card.has_defense() or type(card).hook_under_attack is not _BASE_HOOK_UNDER_ATTACK:
                reactions.append(card)
    return reactions


def _get_base_hook() -> Any:
    from dominion.Card import Card

    return Card.hook_under_attack


_BASE_HOOK_UNDER_ATTACK = _get_base_hook()


# EOF
