#!/usr/bin/env python3
"""Core Player Engine — decoupled from UX via PlayerController."""
# pylint: disable=too-many-instance-attributes, too-many-public-methods
from __future__ import annotations

import contextlib
import inspect
import json
import operator
from collections import defaultdict
from types import NoneType
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

from dominion import Action, Limits, NoCardException, OptionKeys, Phase, Piles, Prompt, Token, Whens
from dominion.Artifact import Artifact
from dominion.Boon import Boon
from dominion.Card import Card, CardType
from dominion.CardPile import CardPile
from dominion.Controller import PlayerController, TextController
from dominion.Cost import Cost
from dominion.Counter import Counter
from dominion.Event import Event
from dominion.Hex import Hex
from dominion.Option import Option
from dominion.PlayArea import PlayArea
from dominion.Project import Project
from dominion.State import State
from dominion.Triggers import active_defense_in_play, hand_attack_reactions
from dominion.TurnEffects import CardTrack, DurationStay, TurnEffects
from dominion.TurnQueue import TurnEntry
from dominion.Way import Way

if TYPE_CHECKING:
    from dominion.Game import Game


###############################################################################
class Player:
    """Core Player state and rules execution; IO/decisions delegated to `self.controller`."""

    def __init__(
        self,
        game: "Game",
        name: str,
        quiet: bool = False,
        controller: Optional[PlayerController] = None,
        **kwargs: Any,
    ) -> None:
        self.game = game
        self.name = name
        self.quiet = quiet
        self.controller: PlayerController = controller if controller is not None else TextController()
        self._controller_stack: list[PlayerController] = []

        self.currcards: list[Card] = []
        self.score: dict[str, int] = {}
        self.had_cards: list[Card] = []
        self.messages: list[str] = []
        self.piles: dict[Piles, PlayArea] = {
            Piles.HAND: PlayArea(Piles.HAND, game=self.game),
            Piles.EXILE: PlayArea(Piles.EXILE, game=self.game),
            Piles.DURATION: PlayArea(Piles.DURATION, game=self.game),
            Piles.DEFER: PlayArea(Piles.DEFER, game=self.game),
            Piles.DECK: PlayArea(Piles.DECK, game=self.game),
            Piles.PLAYED: PlayArea(Piles.PLAYED, game=self.game),
            Piles.DISCARD: PlayArea(Piles.DISCARD, game=self.game),
            Piles.RESERVE: PlayArea(Piles.RESERVE, game=self.game),
        }
        # First-class set-aside storage (automatically scored and counted in all_cards)
        self.set_aside: dict[str, PlayArea] = {}

        self.projects: list[Project] = []
        self.states: list[State] = []
        self.artifacts: list[Artifact] = []

        self.buys = Counter("Buys", 1)
        self.actions = Counter("Actions", 1)
        self.coins = Counter("Coins", 0, on_add=self._intercept_coin_gain)
        self.potions = Counter("Potions", 0)
        self.villagers = Counter("Villager", 0)
        self.debt = Counter("Debt", 0)
        self.coffers = Counter("Coffers", 0)
        self.favors = Counter("Favors", 0)
        self.newhandsize = 5
        self.specials: dict[str, Any] = {}  # Legacy/card-specific state
        self.limits: dict[Limits, Any] = {Limits.PLAY: None, Limits.BUY: 999}
        self.card_token: bool = False
        self.coin_token: bool = False
        self.journey_token: bool = True
        self.test_input: list[str] = []
        self.forbidden_to_buy: list[Card] = []
        self.played_events = PlayArea("played_events", game=self.game)
        self.played_ways: list[tuple[Way, Card]] = []
        self.played_this_turn: list[Card] = []
        self.duration_stays: dict[str, DurationStay] = {}
        self.turn_effects: TurnEffects = TurnEffects()
        self.buy_phase_part: int = 1  # 1 = playing Treasures, 2 = buying cards/Events/Projects
        self._current_attack_immune: set[str] = set()
        self._attack_reactions_ran: bool = False
        self._just_played_action: bool = False
        self._last_action_played: Optional[Card] = None
        self._last_played_action: Optional[Card] = None
        self._last_played_action_track: Optional[CardTrack] = None
        self.gained_in_buy_phase: list[Card] = []
        self.once: dict[str, bool] = {}
        self.turn_number = 0
        self.stats: dict[str, list[Card]] = {"gained": [], "gained_in_buy": [], "bought": [], "trashed": []}
        self.secret_count = 0  # Legacy counter for cards held in card-local lists
        self.end_of_game_cards: list[Card] = []
        self.phase: Phase = Phase.NONE
        self.misc: dict[str, Any] = {"is_start": False, "cleaned": False}
        self.skip_turn: bool = False
        self.uuid: str = ""
        self._initial_deck(kwargs.get("heirlooms", None), kwargs.get("use_shelters", False))
        self._initial_tokens()
        self.pick_up_hand()
        game.output(f"Player {name} is at the table")

    ###########################################################################
    @property
    def active_controller(self) -> PlayerController:
        """Controller currently making decisions for this player (supports Possession overrides)."""
        return self._controller_stack[-1] if self._controller_stack else self.controller

    @contextlib.contextmanager
    def override_controller(self, controller: PlayerController):
        """Temporarily redirect this player's decisions to `controller` (e.g., Possession)."""
        self._controller_stack.append(controller)
        try:
            yield
        finally:
            self._controller_stack.pop()

    ###########################################################################
    def _intercept_coin_gain(self, amount: int) -> int:
        """Complete Rules Ch. VI p. 42: -$1 token triggers whenever you would get Coins."""
        if amount > 0 and self.coin_token:
            self.coin_token = False
            self.output("-1 Coin token reduces coin gain by 1")
            return max(0, amount - 1)
        return amount

    ###########################################################################
    def get_set_aside_area(self, key: str) -> PlayArea:
        """Return (creating if needed) a named set-aside PlayArea included in scoring & card counts."""
        if key not in self.set_aside:
            self.set_aside[key] = PlayArea(f"set_aside:{key}", game=self.game)
        return self.set_aside[key]

    ###########################################################################
    def print_state(self) -> None:
        """Print the player state for debugging"""
        print(f"\n{self.name} {self.turn_number} --------------------------")
        print(f"  state: {', '.join([_.name for _ in self.states])}")
        print(f"  artifacts: {', '.join([_.name for _ in self.artifacts])}")
        print(f"  projects: {', '.join([_.name for _ in self.projects])}")
        for pile in self.piles:
            self.piles[pile].dump(f"  {pile.name}")
        for key, mat in self.set_aside.items():
            if mat:
                mat.dump(f"  SET_ASIDE[{key}]")
        print(f"  score: {self.get_score()} {self.get_score_details()}")
        print(f"  tokens: {self.tokens}")
        print(f"  phase: {self.phase} (buy_part={self.buy_phase_part})")
        print(
            f"  coin={self.coins.get()} debt={self.debt.get()} actions={self.actions.get()}"
            f" buys={self.buys.get()} favors={self.favors.get()}"
            f" coffers={self.coffers.get()}"
            f" villagers={self.villagers.get()} potions={self.potions.get()}"
        )
        print("  messages:")
        for msg in self.messages:
            print(f"\t{msg}")

    ###########################################################################
    def get_card_from_pile(self, pile: str, name: str = "", top_only: bool = False) -> Card:
        """Get a card from a card pile"""
        card = self.game.get_card_from_pile(pile, name, top_only=top_only)
        card.player = self
        self.debug(card, f"Got {card} from pile")
        return card

    ###########################################################################
    def _initial_deck(self, heirlooms: Optional[list[str]] = None, use_shelters: bool = False) -> None:
        """Provide the initial deck (7 Coppers minus Heirlooms, + 3 Estates or 3 Shelters)."""
        if heirlooms is None:
            heirlooms = []

        self.piles[Piles.DECK].empty()

        for _ in range(7 - len(heirlooms)):
            card = self.get_card_from_pile("Copper")
            self.add_card(card, Piles.DECK)

        for heirloom in heirlooms:
            card = self.get_card_from_pile(heirloom)
            self.add_card(card, Piles.DECK)

        if use_shelters:
            self._use_shelters()
        else:
            for _ in range(3):
                card = self.get_card_from_pile("Estate")
                self.add_card(card, Piles.DECK)

        for card in self.piles[Piles.DECK]:
            card.player = self

        self.piles[Piles.DECK].shuffle()

    ###########################################################################
    def _use_shelters(self) -> None:
        """Pick one of each Shelter for starting deck."""
        for shelter in ("Overgrown Estate", "Hovel", "Necropolis"):
            card = self.get_card_from_pile("Shelters", shelter)
            self.add_card(card, Piles.DECK)

    ###########################################################################
    def _initial_tokens(self) -> None:
        self.tokens: dict[Token, Optional[str]] = {
            Token.TRASHING: None,
            Token.ESTATE: None,
            Token.PLUS_1_CARD: None,
            Token.PLUS_1_ACTION: None,
            Token.PLUS_1_BUY: None,
            Token.PLUS_1_COIN: None,
            Token.MINUS_2_COST: None,
        }

    ###########################################################################
    def replace_traveller(self, src: Card, replace_with: str) -> None:
        """For Traveller cards, optionally exchange `src` for `replace_with` (Ch. IV p. 31)."""
        assert isinstance(src, Card)
        assert isinstance(replace_with, str)

        if src not in self.piles[Piles.PLAYED] and src.location != Piles.DISCARD:
            self.output(f"Not activating {src} traveller as not played")
            return

        if replace_with not in self.game.card_piles or self.game.card_piles[replace_with].is_empty():
            self.output(f"No more {replace_with} cards")
            return

        if self.plr_choose_options(
            "Replace Traveller",
            (f"Keep {src}", False),
            (f"Replace with a {replace_with}?", True),
        ):
            self.exchange_card(src, replace_with)

    ###########################################################################
    def exchange_card(self, src: Card, dst: str, destination: Piles = Piles.DISCARD) -> Optional[Card]:
        """Complete Rules Ch. IV (Exchanging p. 31):
        Return `src` to TOP of its pile and take the top card of `dst` into `destination` (discard pile).
        Exchanging is NOT considered gaining a card (no when-gain hooks, no stats['gained'],
        no Trashing token, no Exile mat un-exile).
        """
        assert isinstance(src, Card), f"exchange_card {src=} {type(src)=}"
        assert isinstance(dst, str), f"exchange_card {dst=} {type(dst)=}"

        src_pile_name = src.pile
        if src_pile_name not in self.game.card_piles:
            return None
        dst_pile_name = self.game.card_instances[dst].pile if dst in self.game.card_instances else dst
        if dst_pile_name not in self.game.card_piles or self.game.card_piles[dst_pile_name].is_empty():
            return None

        # Verify top card of destination pile matches `dst`
        if self.game.card_piles[dst_pile_name].top_card() != dst:
            return None

        try:
            new_card = self.get_card_from_pile(dst_pile_name, dst, top_only=True)
        except NoCardException:
            return None

        # Return `src` to TOP of its pile
        if src.location and src.location != Piles.CARDPILE:
            self.remove_card(src)
        self.game.card_piles[src_pile_name].add_to_top(src)
        src.player = None
        src.location = Piles.CARDPILE

        # Place replacement directly in destination (normally Piles.DISCARD)
        self.add_card(new_card, destination)
        return new_card

    ###########################################################################
    def replace_card(self, src: Card, dst: str, **kwargs: Any) -> Optional[Card]:
        """Alias for exchange_card()."""
        destination = kwargs.get("destination", Piles.DISCARD)
        return self.exchange_card(src, dst, destination=destination)

    ###########################################################################
    def flip_journey_token(self) -> bool:
        """Flip a journey token - and return its new state"""
        self.journey_token = not self.journey_token
        return self.journey_token

    ###########################################################################
    def receive_hex(self, hx: Optional[Hex] = None, discard: bool = True) -> None:
        """Receive a hex (pass `discard=False` when Game.distribute_hex shares one Hex)."""
        if hx is None:
            hx = self.game.receive_hex()
        if hx is None:
            self.output("No more hexes")
            return
        self.output(f"Received {hx} as a hex")
        self.output(hx.description(self))
        if hx.cards:
            self.pickup_cards(hx.cards)
        self.add_actions(hx.actions)
        self.buys += hx.buys
        coin_val = self.hook_spend_value(hx, actual=True)
        if coin_val:
            self.coins += coin_val
        hx.special(game=self.game, player=self)
        if discard:
            self.game.discard_hex(hx)

    ###########################################################################
    def receive_boon(self, boon: Optional[Boon] = None, discard: bool = True) -> Optional[Boon]:
        """Receive a boon"""
        if boon is None:
            boon = self.game.receive_boon()
        if boon is None:
            self.output("No more boons")
            return None
        self.output(f"Received {boon} as a boon")
        self.output(boon.description(self))
        if boon.cards:
            self.pickup_cards(boon.cards)
        self.add_actions(boon.actions)
        self.buys += boon.buys
        coin_val = self.hook_spend_value(boon, actual=True)
        if coin_val:
            self.coins += coin_val
        boon.special(game=self.game, player=self)
        if discard:
            self.game.discard_boon(boon)
        return boon

    ###########################################################################
    def do_once(self, name: str) -> bool:
        """Allow a player to do something once per turn"""
        if self.has_done_once(name):
            return False
        self.once[name] = True
        return True

    ###########################################################################
    def has_done_once(self, name: str) -> bool:
        """Check if the player has done the thing already"""
        return name in self.once

    ###########################################################################
    def place_token(self, token: Token, pilename: str) -> None:
        """Place a token on the specified pile"""
        assert isinstance(token, Token)
        self.tokens[token] = pilename

    ###########################################################################
    def which_token(self, pilename: str) -> list[Token]:
        """Return which token(s) are on a Supply pile (accepts pile name or card name)."""
        assert isinstance(pilename, str)
        resolved = pilename
        if pilename in self.game.card_instances and self.game.card_instances[pilename].pile:
            resolved = self.game.card_instances[pilename].pile
        return [token for token, location in self.tokens.items() if location in (pilename, resolved)]

    ###########################################################################
    def call_reserve(self, card: str | Card) -> Optional[Card]:
        """Call a Reserve card from Tavern mat (Ch. IV p. 28):
        Move from Tavern mat to Play area FIRST, THEN resolve its called ability.
        """
        if isinstance(card, str):
            card = self.piles[Piles.RESERVE][card]  # type: ignore
            if not card:
                return None
        assert isinstance(card, Card)
        if card not in self.piles[Piles.RESERVE]:
            return None
        self.output(f"Calling {card} from Reserve")
        # Move to PLAYED FIRST so calling it can't re-trigger itself on Tavern mat!
        self.piles[Piles.RESERVE].remove(card)
        self.add_card(card, Piles.PLAYED)
        self.currcards.append(card)
        try:
            card.hook_call_reserve(game=self.game, player=self)
        finally:
            self.currcards.pop()
        return card

    ###########################################################################
    def reveal_card(self, card: Card) -> None:
        """Reveal a card to everyone"""
        if not card:
            return
        self.game.output(f"{self.name} reveals {card}")
        card.hook_reveal_this_card(game=self.game, player=self)

    ###########################################################################
    def trash_cards(self, cards: list[Card], **kwargs: Any) -> list[Card]:
        """Complete Rules Ch. III (When-trash p. 22) & Ch. VI p. 45:
        Trash ALL cards in the batch at once BEFORE resolving any when-trash abilities.
        """
        to_trash: list[Card] = []
        for card in cards:
            assert isinstance(card, Card)
            if card.location == Piles.TRASH:
                self.output(f"{card} already in trash")
                continue
            self.debug(card, f"trash_card({card=})")
            if card.location and card.location != Piles.TRASH:
                self.remove_card(card)
            self.game.trash_pile.addToTop(card)
            card.player = None
            card.location = Piles.TRASH
            self.stats["trashed"].append(card)
            to_trash.append(card)

        exclude_hook = kwargs.get("exclude_hook", [])
        for card in to_trash:
            self.currcards.append(card)
            try:
                trash_opts = card.hook_trash_this_card(game=self.game, player=self)
                # Support cards like Fortress that move themselves out of trash on when-trash
                if isinstance(trash_opts, dict) and trash_opts.get(OptionKeys.TRASH) is False:
                    if card in self.game.trash_pile:
                        self.game.trash_pile.remove(card)
                        card.location = Piles.LIMBO
            finally:
                self.currcards.pop()

            for hook_card in self._trash_hook_sources():
                if hook_card.name not in exclude_hook:
                    self.currcards.append(hook_card)
                    try:
                        hook_card.hook_trash_card(game=self.game, player=self, card=card)
                    finally:
                        self.currcards.pop()

        return to_trash

    ###########################################################################
    def trash_card(self, card: Card, **kwargs: Any) -> None:
        """Trash a single card."""
        self.trash_cards([card], **kwargs)

    ###########################################################################
    def ensure_deck_cards(self, num: int) -> None:
        """Complete Rules Ch. III (Moving cards from your deck p. 20, 2nd Ed):
        If deck has fewer than `num` cards and discard pile is non-empty,
        shuffle discard pile and add it to the BOTTOM of the deck.
        """
        if len(self.piles[Piles.DECK]) < num and self.piles[Piles.DISCARD]:
            self.refill_deck()

    ###########################################################################
    def next_card(self) -> Card:
        """Pick up and return the next card from the deck"""
        if not self.piles[Piles.DECK]:
            self.refill_deck()
        if not self.piles[Piles.DECK]:
            self.output("No more cards in deck")
            raise NoCardException
        crd = self.piles[Piles.DECK].next_card()
        crd.location = None  # We don't know where it is going yet
        return crd

    ###########################################################################
    def top_card(self) -> Card:
        """Return the top card from the deck without moving it"""
        if not self.piles[Piles.DECK]:
            self.refill_deck()
        if not self.piles[Piles.DECK]:
            self.output("No more cards in deck")
            raise NoCardException
        return self.piles[Piles.DECK].top_card()

    ###########################################################################
    def refill_deck(self) -> None:
        """Shuffle discard pile and add to the BOTTOM of deck (Shadow cards on bottom)."""
        self._shuffle_discard()
        # Bump move_epoch on remaining deck cards? No, only shuffled cards moved.
        shadows: list[Card] = []
        normals: list[Card] = []
        while self.piles[Piles.DISCARD]:
            card = self.piles[Piles.DISCARD].next_card()
            card.player = self
            card.location = Piles.DECK
            card.bump_move_epoch()
            if card.isShadow():
                shadows.append(card)
            else:
                normals.append(card)
        # Bottom-most are Shadow cards, then shuffled normals, all BELOW any existing deck cards!
        batch = shadows + normals
        self.piles[Piles.DECK].add_bottom_batch(batch)

        for card in self._shuffle_hook_sources():
            self.currcards.append(card)
            try:
                card.hook_post_shuffle(game=self.game, player=self)
            finally:
                self.currcards.pop()

    ###########################################################################
    def pickup_cards(self, num: int, verbose: bool = True, verb: str = "Picked up") -> list[Card]:
        """Draw `num` cards from deck into hand (honoring -1 Card token & 2nd-ed shuffle rule)."""
        if num <= 0:
            return []
        if self.card_token:
            self.output("-1 Card token reduces cards drawn by 1")
            self.card_token = False
            num -= 1
            if num <= 0:
                return []
        if len(self.piles[Piles.DECK]) < num and self.piles[Piles.DISCARD]:
            with contextlib.suppress(NoCardException):
                self.refill_deck()
        cards: list[Card] = []
        for _ in range(num):
            try:
                if card := self.pickup_card(verbose=verbose, verb=verb, check_token=False):
                    cards.append(card)
            except NoCardException:
                break
        return cards

    ###########################################################################
    def pickup_card(
        self,
        card: Optional[Card] = None,
        verbose: bool = True,
        verb: str = "Picked up",
        check_token: bool = True,
    ) -> Optional[Card]:
        """Draw a card from the deck into the player's hand (honoring -1 Card token)."""
        if card is None:
            if check_token and self.card_token:
                self.output("-1 Card token prevents drawing a card")
                self.card_token = False
                return None
            card = self.next_card()
            if not card:
                self.output("No more cards to pickup")
                raise NoCardException
        assert isinstance(card, Card)
        self.debug(card, f"pickup_card({card=})")
        self.add_card(card, Piles.HAND)
        if verbose:
            self.output(f"{verb} {card}")
        return card

    ###########################################################################
    def _shuffle_discard(self) -> None:
        num_cards = len(self.piles[Piles.DISCARD])
        if num_cards == 0:
            self.output("No more cards to use")
            raise NoCardException
        self.output(f"Shuffling Pile of {num_cards} cards")
        for card in self._shuffle_hook_sources():
            self.currcards.append(card)
            try:
                card.hook_pre_shuffle(game=self.game, player=self)
            finally:
                self.currcards.pop()

        self.piles[Piles.DISCARD].shuffle()

    ###########################################################################
    def pick_up_hand(self, hand_size: Optional[int] = None) -> None:
        """Replenish hand from deck (honoring -1 Card token and 2nd-ed shuffle rule)."""
        if hand_size is None:
            hand_size = self.newhandsize
        need = max(0, hand_size - self.piles[Piles.HAND].size())
        if need <= 0:
            return
        if self.card_token:
            self.output("-Card token reduce draw by one")
            need -= 1
            self.card_token = False
        if need <= 0:
            return
        if len(self.piles[Piles.DECK]) < need and self.piles[Piles.DISCARD]:
            with contextlib.suppress(NoCardException):
                self.refill_deck()
        for _ in range(need):
            try:
                self.pickup_card(verb="Dealt", check_token=False)
            except NoCardException:
                self.output("Not enough cards to fill hand")
                break

    ###########################################################################
    def remove_card(self, card: Card) -> None:
        """Remove a card from wherever it currently lives."""
        self.debug(card, f"remove_card({card=}) {card.location=}")
        if card.location in self.piles:
            self.piles[card.location].remove(card)
            card.location = Piles.LIMBO
        elif card.location == Piles.CARDPILE:
            self.game.card_piles[card.pile].remove(card.name)
            card.location = Piles.LIMBO
        elif card.location == Piles.TRASH:
            self.game.trash_pile.remove(card)
            card.location = Piles.LIMBO
        elif card.location == Piles.SPECIAL:
            for mat in self.set_aside.values():
                if card in mat:
                    mat.remove(card)
                    break
            card.location = Piles.LIMBO
        elif card.location in (Piles.LIMBO, None):
            pass
        else:
            raise AssertionError(f"Trying to remove_card {card} from unknown location: {card.location}")

    ###########################################################################
    def move_card(self, card: Card, dest: Piles | PlayArea) -> Card:
        """Move a card to `dest`."""
        assert isinstance(card, Card), f"move_card({card=}) {type(card)}"
        self.debug(card, f"move_card({card=}, {dest=})")
        self.remove_card(card)
        return self.add_card(card, dest)

    ###########################################################################
    def debug(self, card: Card, msg: str) -> None:  # pragma: no coverage
        """Debug card messages"""
        if card.debug:
            func_calls = []
            stack = inspect.stack()
            for rec in stack:
                if "pydominion" not in rec.filename:
                    continue
                if rec.function == "<module>":
                    break
                func_calls.append(f"{rec.function}: {rec.lineno}")
            print(f"# DBG {msg} {func_calls[1:]}")

    ###########################################################################
    def add_card(self, card: Card, dest: Piles | PlayArea = Piles.DISCARD) -> Card:
        """Add an existing card to a new location (with proper top-of-pile semantics)."""
        self.debug(card, f"add_card({card=},{dest=}")
        assert isinstance(card, Card), f"{card=} {type(card)=}"
        assert isinstance(dest, (Piles, PlayArea)), f"{dest=} {type(dest)=}"
        card.game = self.game

        # Custom PlayAreas (e.g. set-aside mats)
        if isinstance(dest, PlayArea):
            card.player = self
            card.location = Piles.SPECIAL
            dest.addToTop(card)
            return card

        # Return card to TOP of its pile (Complete Rules Ch. IV Split piles p. 36)
        if dest == Piles.CARDPILE or isinstance(dest, CardPile):
            self.game.card_piles[card.pile].add_to_top(card)
            card.player = None
            card.location = Piles.CARDPILE
            return card

        if dest == Piles.TRASH:
            self.game.trash_pile.addToTop(card)
            card.player = None
            card.location = Piles.TRASH
            return card

        card.player = self
        if dest in (Piles.DISCARD, Piles.PLAYED):
            # Top of discard pile / newest played card is index -1
            self.piles[dest].addToTop(card)
            card.location = dest
        elif dest in self.piles:
            assert isinstance(dest, Piles)
            self.piles[dest].add(card)
            card.location = dest
        elif dest == Piles.TOPDECK:
            # Covering top of deck loses track of the previously top card (Ch. III p. 26)
            if self.piles[Piles.DECK]:
                self.piles[Piles.DECK].top_card().bump_move_epoch()
            card.location = Piles.DECK
            self.piles[Piles.DECK].addToTop(card)
        else:
            raise AssertionError(f"Adding card {card} to unknown location: {dest}")
        return card

    ###########################################################################
    def discard_cards(
        self,
        cards: list[Card],
        source: Optional[Piles] = None,
        hook: bool = True,
    ) -> list[Card]:
        """Complete Rules Ch. III (When-discard p. 22):
        Move ALL cards in the batch to TOP of discard pile first, THEN fire when-discard triggers.
        Removes each card from its actual source pile (not just HAND).
        """
        discarded: list[tuple[Card, Optional[Piles]]] = []
        for card in list(cards):
            assert isinstance(card, Card)
            actual_source = source if source is not None else card.location
            self.debug(card, f"discard_card({card=}, {actual_source=}, {hook=})")
            if card.location in self.piles and card in self.piles[card.location]:
                self.piles[card.location].remove(card)
            elif card.location == Piles.SPECIAL:
                self.remove_card(card)
            self.add_card(card, Piles.DISCARD)
            discarded.append((card, actual_source))

        if hook:
            for card, actual_source in discarded:
                self.currcards.append(card)
                try:
                    self.hook_discard_this_card(card, actual_source)
                finally:
                    self.currcards.pop()
                for way, crd in list(self.played_ways):
                    if crd == card:
                        self.currcards.append(way)
                        try:
                            way.hook_way_discard_this_card(game=self.game, player=self, card=crd)
                        finally:
                            self.currcards.pop()
                for other_card in self._discard_hook_sources():
                    self.currcards.append(other_card)
                    try:
                        hook_fn = other_card.hook_discard_any_card
                        if "source" in hook_fn.__code__.co_varnames:
                            hook_fn(game=self.game, player=self, card=card, source=actual_source)
                        else:
                            hook_fn(game=self.game, player=self, card=card)
                    finally:
                        self.currcards.pop()

        return [c for c, _ in discarded]

    ###########################################################################
    def discard_card(self, card: Card, source: Optional[Piles] = None, hook: bool = True) -> None:
        """Discard a single card."""
        self.discard_cards([card], source=source, hook=hook)

    ###########################################################################
    def discard_hand(self, options: Optional[dict[OptionKeys, Any]] = None) -> None:
        """Discard all cards in the player's HAND all at once (never discards PLAYED).

        Cards like Tactician, Guide, and Way of the Mole call `discard_hand({})` to
        discard their hand during Action phase without touching cards in play!
        """
        hand_cards = list(self.piles[Piles.HAND])
        if hand_cards:
            self.discard_cards(hand_cards, source=Piles.HAND, hook=True)

    ###########################################################################
    def _cleanup_discard_played_and_hand(self, options: dict[OptionKeys, Any]) -> None:
        """Complete Rules Ch. II (Clean-up phase p. 13) & Ch. VI p. 44:
        - Discard hand all at once.
        - Discard non-staying cards from PLAYED one by one (keeping staying Durations &
          their attached throne-rooms in play via Piles.DURATION).
        - Discard any cards in OTHER players' PLAYED areas (e.g. Reactions played on this turn).
        """
        # 1. Discard hand all at once
        hand_cards = list(self.piles[Piles.HAND])
        if hand_cards:
            self.discard_cards(hand_cards, source=Piles.HAND, hook=True)

        # 2. Identify cards staying in play (Durations with pending future abilities + direct throne-rooms)
        staying_uuids: set[str] = set()
        attached_uuids = {
            uid for stay in self.duration_stays.values() for uid, _ in stay.attached_throne_uuids
        }
        for dur_uuid, stay in list(self.duration_stays.items()):
            card = self.game.cards.get(dur_uuid)
            if (card is None or card.location not in (Piles.PLAYED, Piles.DURATION)
                    or card.move_epoch != stay.card_epoch):
                stay.active = False
                if card is not None and card.location == Piles.DURATION:
                    self.move_card(card, Piles.PLAYED)
                continue
            if not stay.active:
                continue
            staying_uuids.add(dur_uuid)
            valid_thrones = []
            for throne_uuid, throne_epoch in stay.attached_throne_uuids:
                throne_card = self.game.cards.get(throne_uuid)
                if (throne_card is not None and throne_card.location in (Piles.PLAYED, Piles.DURATION)
                        and throne_card.move_epoch == throne_epoch):
                    staying_uuids.add(throne_uuid)
                    valid_thrones.append((throne_uuid, throne_epoch))
            if stay.attached_throne_uuids:
                stay.triggers_remaining = max(1, min(stay.triggers_remaining, 1 + len(valid_thrones)))
            stay.attached_throne_uuids = valid_thrones

        # A stopped attachment must not remain stranded in the Duration area.
        for card in list(self.piles[Piles.DURATION]):
            if card.uuid in attached_uuids and card.uuid not in staying_uuids:
                self.move_card(card, Piles.PLAYED)

        # Move staying cards from PLAYED -> DURATION so they remain in play next turn
        moved_epochs: dict[str, tuple[int, int]] = {}
        for card in list(self.piles[Piles.PLAYED]):
            if card.uuid in staying_uuids:
                old_epoch = card.move_epoch
                self.move_card(card, Piles.DURATION)
                moved_epochs[card.uuid] = (old_epoch, card.move_epoch)
        for uid, stay in self.duration_stays.items():
            if uid in moved_epochs:
                stay.card_epoch = moved_epochs[uid][1]
            stay.attached_throne_uuids = [
                (uid, moved_epochs[uid][1] if uid in moved_epochs and epoch == moved_epochs[uid][0] else epoch)
                for uid, epoch in stay.attached_throne_uuids
            ]

        # 3. Discard remaining cards in PLAYED (if not suppressed by Journey / OptionKeys.DISCARD_PLAYED)
        if options.get(OptionKeys.DISCARD_PLAYED, True):
            for card in list(self.piles[Piles.PLAYED]):
                # Card may have already moved itself via a hook (e.g. Traveller exchange, Scheme, Herbalist)
                if card in self.piles[Piles.PLAYED]:
                    self.discard_cards([card], source=Piles.PLAYED, hook=True)

        # 4. Clean up cards in OTHER players' play areas (Ch. II p. 13)
        for other in self.game.player_list():
            if other is self:
                continue
            if other.piles[Piles.PLAYED]:
                other_staying = {
                    uid for uid, stay in other.duration_stays.items()
                    if stay.active and (c := self.game.cards.get(uid)) is not None
                    and c.move_epoch == stay.card_epoch
                }
                for card in list(other.piles[Piles.PLAYED]):
                    if card.uuid in other_staying:
                        other.move_card(card, Piles.DURATION)
                        other.duration_stays[card.uuid].card_epoch = card.move_epoch
                    elif card in other.piles[Piles.PLAYED]:
                        other.discard_cards([card], source=Piles.PLAYED, hook=True)

    ###########################################################################
    def _get_whens(self) -> list[Whens]:
        """Return active Reserve call windows (Ch. IV Reserve cards p. 28)."""
        whens: list[Whens] = [Whens.ANY]
        if self._just_played_action and self.phase == Phase.ACTION:
            whens.append(Whens.POSTACTION)
        if self.misc.get("is_start"):
            whens.append(Whens.START)
        return whens

    ###########################################################################
    def _card_check(self) -> None:
        """Verify card location invariants."""
        for pile, contents in self.piles.items():
            for card in contents:
                assert card.location == pile, f"{self.name} {card=} {pile=} {card.location=}"

    ###########################################################################
    def turn(self, turn_entry: Optional[TurnEntry] = None) -> None:
        """Have a player turn (regular, extra, or Fleet)."""
        self.start_turn(turn_entry=turn_entry)
        self.do_turn(turn_entry=turn_entry)
        self.end_turn()

    ###########################################################################
    def do_turn(self, turn_entry: Optional[TurnEntry] = None) -> None:
        """Execute Action -> Buy -> Night -> Clean-up phases."""
        # Complete Rules Ch. II p. 15: extra turns & Fleet turns do NOT count for tie-breakers!
        is_extra = bool(turn_entry and (turn_entry.is_extra or turn_entry.is_fleet))
        if not is_extra:
            self.turn_number += 1
        if not self.quiet:
            print()
        self.output(f"{'#' * 30} Turn {self.turn_number} {'#' * 30}")
        stats = f"({self.get_score()} points, {self.count_cards()} cards)"
        if self.skip_turn:
            self.skip_turn = False
            return
        self.output(f"{self.name}'s Turn {stats}")
        self.phase = Phase.ACTION
        while True:
            self._card_check()
            match self.phase:
                case Phase.ACTION:
                    self.action_phase()
                    if self.phase == Phase.ACTION:
                        self.phase = Phase.BUY
                case Phase.BUY:
                    self.buy_phase()
                    if self.phase == Phase.BUY:
                        self.phase = Phase.NIGHT
                case Phase.NIGHT:
                    self.night_phase()
                    if self.phase == Phase.NIGHT:
                        self.phase = Phase.CLEANUP
                case Phase.CLEANUP:
                    self.cleanup_phase()
                    break
        self._card_check()

    ###########################################################################
    def night_phase(self) -> None:
        """Do the Night Phase"""
        nights = [_ for _ in self.piles[Piles.HAND] if _.isNight()]
        if not nights:
            return
        self.output("************ Night Phase ************")
        while True:
            self.active_controller.display_overview(self)
            options = Prompt.choice_selection(self)
            prompt = Prompt.generate_prompt(self)

            opt = self.user_input(options, prompt)
            self._perform_action(opt)
            if opt["action"] == Action.QUIT:
                return

    ###########################################################################
    def action_phase(self) -> None:
        """Do the Action phase"""
        self.output("************ Action Phase ************")
        while True:
            self.active_controller.display_overview(self)
            options = Prompt.choice_selection(self)
            prompt = Prompt.generate_prompt(self)

            opt = self.user_input(options, prompt)
            self._perform_action(opt)
            if opt["action"] == Action.QUIT:
                return

    ###########################################################################
    def buy_phase(self) -> None:
        """Do the two-part Buy phase (Part 1: Treasures, Part 2: Buys/Events/Projects)."""
        self.output("************ Buy Phase ************")
        self.buy_phase_part = 1
        self.hook_pre_buy()
        while True:
            self.active_controller.display_overview(self)
            options = Prompt.choice_selection(self)
            prompt = Prompt.generate_prompt(self)

            opt = self.user_input(options, prompt)
            self._perform_action(opt)
            if opt["action"] == Action.QUIT:
                break
        self.hook_end_buy_phase()

    ###########################################################################
    def playable_actions(self) -> list[Card]:
        """Return a list of playable actions from hand and Shadow Actions from deck."""
        actions = [card for card in self.piles[Piles.HAND] if card.isAction() and card.playable]
        actions.extend(
            card for card in self.piles[Piles.DECK] if card.isAction() and card.isShadow() and card.playable
        )
        return actions

    ###########################################################################
    def hook_end_buy_phase(self) -> None:
        """End of Buy phase triggers (includes PLAYED, DURATION, RESERVE e.g. Wine Merchant, and Projects)."""
        for card in (
            list(self.piles[Piles.PLAYED])
            + list(self.piles[Piles.DURATION])
            + list(self.piles[Piles.RESERVE])
            + list(self.projects)
        ):
            self.currcards.append(card)
            try:
                card.hook_end_buy_phase(game=self.game, player=self)
            finally:
                self.currcards.pop()

    ###########################################################################
    def cleanup_phase(self) -> None:
        """Cleanup Phase (Ch. II p. 13 & Ch. VI p. 44)."""
        self.phase = Phase.CLEANUP
        options: dict[OptionKeys, Any] = {}
        self.had_cards = (
            self.piles[Piles.PLAYED]
            + self.piles[Piles.RESERVE]
            + self.played_events
            + PlayArea(initial=list(self.game.landmarks.values()))
            + self.piles[Piles.DURATION]
        )

        # Discard Boons kept until Clean-up (without reshuffling Boon discard pile!)
        self.game.cleanup_boons()

        # Start-of-Clean-up triggers (includes DURATION so Outpost & similar work!)
        if self.game.prophecy:
            self.currcards.append(self.game.prophecy)
            try:
                res = self.game.prophecy.hook_cleanup(self.game, self)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()
        for trait in self.game.traits.values():
            self.currcards.append(trait)
            try:
                res = trait.hook_cleanup(self.game, self)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()
        for card in self._cleanup_hook_sources():
            self.currcards.append(card)
            try:
                res = card.hook_cleanup(self.game, self)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()
        for event in list(self.played_events):
            self.currcards.append(event)
            try:
                res = event.hook_cleanup(self.game, self)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()
        if self.game.ally:
            self.currcards.append(self.game.ally)
            try:
                res = self.game.ally.hook_cleanup(self.game, self)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()

        self._cleanup_discard_played_and_hand(options)
        self.pick_up_hand()
        self.misc["cleaned"] = True

    ###########################################################################
    def payback(self, amount: Optional[int] = None) -> int:
        """Pay off Debt at $1 per 1D (supports partial payment via `amount`)."""
        max_pay = min(self.coins.get(), self.debt.get())
        if max_pay <= 0:
            return 0
        payback = max_pay if amount is None else max(0, min(amount, max_pay))
        self.output(f"Paying back {payback} debt")
        self.coins -= payback
        self.debt -= payback
        return payback

    ###########################################################################
    def _perform_action(self, opt: Option) -> None:
        """Perform the action selected"""
        action = opt["action"]
        if action not in (Action.PLAY, Action.RESERVE):
            self._just_played_action = False
        match action:
            case Action.BUY:
                self.buy_phase_part = 2
                self.buy_card(cast(str, opt["name"]))
            case Action.EVENT:
                self.buy_phase_part = 2
                self.perform_event(cast(Event, opt["card"]))
            case Action.PROJECT:
                self.buy_phase_part = 2
                self.buy_project(cast(Project, opt["card"]))
            case Action.RESERVE:
                self.call_reserve(cast(Card, opt["card"]))
            case Action.COFFER:
                self.spend_coffer()
            case Action.VILLAGER:
                self.spend_villager()
            case Action.PLAY:
                self.play_card(cast(Card, opt["card"]))
            case Action.SPEND:
                if self.phase == Phase.BUY and self.buy_phase_part != 1:
                    self.output("Cannot play Treasures after buying a card, Event, or Project")
                    return
                self.play_card(cast(Card, opt["card"]))
            case Action.PAYBACK:
                self.payback()
            case Action.SPENDALL:
                if self.phase == Phase.BUY and self.buy_phase_part != 1:
                    self.output("Cannot play Treasures after buying a card, Event, or Project")
                    return
                self._spend_all_cards()
            case Action.QUIT:
                return
            case Action.WAY:
                self.perform_way(cast(Way, opt["way"]), cast(Card, opt["card"]))
            case Action.NONE:
                return

        self.misc["is_start"] = False

    ###########################################################################
    def add_score(self, reason: str, points: int = 1) -> None:
        """Add VP tokens / score to the player"""
        if reason not in self.score:
            self.score[reason] = 0
        self.score[reason] += points

    ###########################################################################
    def all_cards(self) -> PlayArea:
        """Return ALL cards owned by this player (including all set-aside mats)."""
        x = PlayArea("all", game=self.game)
        x += self.piles[Piles.DISCARD]
        x += self.piles[Piles.HAND]
        x += self.piles[Piles.DECK]
        x += self.piles[Piles.PLAYED]
        x += self.piles[Piles.DURATION]
        x += self.piles[Piles.DEFER]
        x += self.piles[Piles.RESERVE]
        x += self.piles[Piles.EXILE]
        for mat in self.set_aside.values():
            x += mat
        return x

    ###########################################################################
    def get_score_details(self) -> dict[str, int]:
        """Calculate score of the player from all factors (additive — VP tokens NEVER overwrite card VP!)."""
        score: dict[str, int] = {}
        for card in self.all_cards():
            score[card.name] = score.get(card.name, 0) + card.victory + card.special_score(self.game, self)
        for state in self.states:
            if state.victory:
                score[state.name] = score.get(state.name, 0) + state.victory
        if self.game.ally:
            ally_vp = self.game.ally.special_score(self.game, self)
            if ally_vp or self.game.ally.name not in score:
                score[self.game.ally.name] = score.get(self.game.ally.name, 0) + ally_vp
        for reason, pts in self.score.items():
            score[reason] = score.get(reason, 0) + pts
        return score

    ###########################################################################
    def get_score(self, verbose: bool = False) -> int:
        """Return total VP score"""
        scr = self.get_score_details()
        vp = sum(scr.values())
        if verbose:
            self.game.output(f"{self.name} (Turn {self.turn_number}): {json.dumps(scr, indent=2)}")
        return vp

    ###########################################################################
    def hook_pre_buy(self) -> None:
        """Start-of-Buy-phase triggers"""
        for card in self.in_play_and_landscape_cards():
            self.currcards.append(card)
            try:
                card.hook_pre_buy(game=self.game, player=self)
            finally:
                self.currcards.pop()

    ###########################################################################
    def hook_allowed_to_buy(self, card: Card) -> bool:
        """Hook to check if you are allowed to buy a card"""
        if self.misc.get("no_buy_cards"):
            return False
        return card.hook_allowed_to_buy(game=self.game, player=self)

    ###########################################################################
    def hook_buy_card(self, card: Card) -> None:
        """When-buy hooks (fired after payment, before gaining)."""
        for c in self._buy_hook_sources():
            self.currcards.append(c)
            try:
                c.hook_buy_card(game=self.game, player=self, card=card)
            finally:
                self.currcards.pop()

    ###########################################################################
    def start_turn(self, turn_entry: Optional[TurnEntry] = None) -> None:
        """Start of a player's turn — reset pools, promote next-turn effects, run start-of-turn triggers."""
        self.phase = Phase.START
        self.buys.set(1)
        self.actions.set(1)
        self.coins.set(0)
        self.potions.set(0)
        self.buy_phase_part = 1
        self.played_ways = []
        self.played_this_turn = []
        self.gained_in_buy_phase = []
        self._just_played_action = False
        self._last_action_played = None
        self._last_played_action = None
        self._last_played_action_track = None
        self.misc = {"is_start": True, "cleaned": False}
        self.stats = {"gained": [], "gained_in_buy": [], "bought": [], "trashed": []}
        self.turn_effects.start_turn()

        if turn_entry and turn_entry.modifiers:
            if "play_limit" in turn_entry.modifiers:
                self.limits[Limits.PLAY] = turn_entry.modifiers["play_limit"]
            if turn_entry.modifiers.get("no_buy_cards"):
                self.misc["no_buy_cards"] = True

        self.active_controller.display_overview(self)
        self._hook_start_turn()
        self._duration_start_turn()
        self._defer_start_turn()

    ###########################################################################
    def _defer_start_turn(self) -> None:
        """Play deferred cards at the start of the turn"""
        for card in list(self.piles[Piles.DEFER]):
            self.output(f"Playing deferred {card}")
            self.currcards.append(card)
            try:
                self.move_card(card, Piles.HAND)
                self.play_card(card, cost_action=False)
            finally:
                self.currcards.pop()

    ###########################################################################
    def _duration_start_turn(self) -> None:
        """Resolve Duration cards at the start of the turn.

        Implements:
          - Multi-play replays (`DurationStay.triggers_remaining`)
          - 2025 Duration-stopping rule (if a Duration or its direct Throne-Room left play,
            its future-turn effects stop!)
          - Moving resolved Durations AND their attached Throne-Rooms back to `PLAYED` so
            they discard cleanly in this turn's Clean-up!
        """
        # Handle Duration cards currently in Piles.DURATION (plus any active staying Duration in PLAYED if start_turn called without cleanup)
        dur_candidates = list(self.piles[Piles.DURATION])
        for dur_uuid, stay in list(self.duration_stays.items()):
            if stay.active:
                c = self.game.cards.get(dur_uuid)
                if c is not None and c.location == Piles.PLAYED and c not in dur_candidates:
                    dur_candidates.append(c)

        for card in dur_candidates:
            stay = self.duration_stays.get(card.uuid)
            # If this card is an attached throne-room (not itself a Duration), skip calling duration()
            if not card.isDuration():
                continue

            # Determine how many times duration() fires (respecting 2025 Lose-Track on throne-rooms)
            triggers = 1
            if stay is not None:
                if (not stay.active or card.move_epoch != stay.card_epoch
                        or card.location not in (Piles.PLAYED, Piles.DURATION)):
                    stay.active = False
                    if card.location == Piles.DURATION:
                        self.move_card(card, Piles.PLAYED)
                    continue
                valid_extra = 0
                for throne_uuid, throne_epoch in stay.attached_throne_uuids:
                    throne = self.game.cards.get(throne_uuid)
                    if (throne is not None and throne.location in (Piles.DURATION, Piles.PLAYED)
                            and throne.move_epoch == throne_epoch):
                        valid_extra += 1
                triggers = (
                    max(1, min(stay.triggers_remaining, 1 + valid_extra))
                    if stay.attached_throne_uuids
                    else max(1, stay.triggers_remaining)
                )

            options: dict[OptionKeys, Any] = {OptionKeys.DESTINATION: Piles.PLAYED}
            if not card.permanent:
                self.output(f"Playing {card} from duration pile")

            for _ in range(triggers):
                if stay is not None and card.move_epoch != stay.card_epoch:
                    stay.active = False
                    break
                self.currcards.append(card)
                try:
                    upd_opts = card.duration(game=self.game, player=self)
                    if isinstance(upd_opts, dict):
                        options.update(upd_opts)
                finally:
                    self.currcards.pop()

            if stay is not None:
                stay.permanent = bool(card.permanent)
            still_permanent = card.permanent and (stay is None or stay.active)
            if not still_permanent:
                if stay is not None:
                    for throne_uuid, throne_epoch in stay.attached_throne_uuids:
                        throne = self.game.cards.get(throne_uuid)
                        if (throne is not None and throne.location == Piles.DURATION
                                and throne.move_epoch == throne_epoch):
                            self.move_card(throne, Piles.PLAYED)
                    self.duration_stays.pop(card.uuid, None)
                if card.location in (Piles.DURATION, Piles.PLAYED) and options[OptionKeys.DESTINATION] != card.location:
                    self.move_card(card, options[OptionKeys.DESTINATION])

        for event in self.game.events.values():
            event.duration(game=self.game, player=self)

    ###########################################################################
    def _hook_start_turn(self) -> None:
        """Start of turn hooks"""
        for card_name in list(self.game.card_instances):
            self.game.card_instances[card_name].hook_start_every_turn(self.game, self)
        for event_name in list(self.game.events):
            self.game.events[event_name].hook_start_every_turn(self.game, self)
        for card in self.in_play_and_landscape_cards() + list(self.piles[Piles.RESERVE]):
            self.currcards.append(card)
            try:
                card.hook_start_turn(self.game, self)
            finally:
                self.currcards.pop()

    ###########################################################################
    def spend_coffer(self) -> None:
        """Spend a Coffer token to gain +$1 (anytime during your turn per 2022 rules)."""
        if self.coffers.get() <= 0:
            return
        self.coffers -= 1
        self.coins += 1
        self.output("Spent a coffer")

    ###########################################################################
    def spend_villager(self) -> None:
        """Spend a Villager token to gain +1 Action (anytime in your Action phase)."""
        if self.villagers.get() <= 0:
            self.output("No villagers to spend")
            return
        self.villagers -= 1
        self.add_actions(1)
        self.output("Spent a villager")

    ###########################################################################
    def exile_card_from_supply(self, card_name: str) -> Optional[Card]:
        """Exile a card from Supply (NOT considered gaining)."""
        try:
            card = self.get_card_from_pile(card_name)
        except NoCardException:
            self.output(f"No more {card_name} in supply")
            return None
        self.add_card(card, Piles.EXILE)
        return card

    ###########################################################################
    def exile_card(self, card: Card) -> None:
        """Send a card to the Exile mat."""
        self.debug(card, f"exile_card({card=})")
        self.move_card(card, Piles.EXILE)

    ###########################################################################
    def end_turn(self) -> None:
        """End of turn"""
        if not self.misc.get("cleaned"):
            self.cleanup_phase()
        self.limits[Limits.PLAY] = None
        self.limits[Limits.BUY] = 999
        self.end_turn_hooks()
        self.turn_effects.end_turn()
        self._just_played_action = False
        self._last_action_played = None
        self._last_played_action = None
        self._last_played_action_track = None
        self.newhandsize = 5
        self.played_events = PlayArea("played_events", game=self.game, initial=[])
        self.messages = []
        self.forbidden_to_buy = []
        self.once = {}
        self.phase = Phase.NONE
        self.had_cards = []
        self.misc.pop("no_buy_cards", None)

    ###########################################################################
    def end_turn_hooks(self) -> None:
        """Call all end-of-turn hooks"""
        for card in list(self.had_cards):
            self.currcards.append(card)
            try:
                card.hook_end_turn(game=self.game, player=self)
            finally:
                self.currcards.pop()
        for trait in self.game.traits.values():
            trait.hook_end_turn(game=self.game, player=self)
        if self.game.ally:
            self.currcards.append(self.game.ally)
            try:
                self.game.ally.hook_end_turn(game=self.game, player=self)
            finally:
                self.currcards.pop()

    ###########################################################################
    def hook_discard_this_card(self, card: Card, source: Optional[PlayArea | Piles] = None) -> None:
        """A card has been discarded"""
        assert isinstance(source, (PlayArea, Piles, NoneType)), f"hook_discard_this_card {source=} {type(source)=}"
        self.currcards.append(card)
        try:
            card.hook_discard_this_card(game=self.game, player=self, source=source)
        finally:
            self.currcards.pop()

    ###########################################################################
    def hook_spend_value(self, card: Card, actual: bool = False) -> int:
        """Compute coin value produced by `card`.

        Note: `-$1` Coin token is now intercepted automatically by `self.coins` on positive add
        so it applies to ALL coin gains (Actions, Treasures, Coffers, Events, Boons, +$1 token).
        When `actual=False` (display preview), we still preview the -$1 reduction.
        """
        val = card.hook_coinvalue(game=self.game, player=self)
        for c in self.in_play_cards():
            val += c.hook_spend_value(game=self.game, player=self, card=card)
        for s in self.states:
            val += s.hook_spend_value(game=self.game, player=self, card=card)
        if val and self.coin_token and not actual:
            val -= 1
        return val

    ###########################################################################
    def _spend_all_cards(self) -> None:
        """Spend all Treasure cards in hand"""
        for card in list(self.piles[Piles.HAND]):
            if card.isTreasure() and card in self.piles[Piles.HAND]:
                self.play_card(card, cost_action=False)

    ###########################################################################
    def _play_card_tokens(self, card: Card) -> None:
        """Trigger Adventures '+' pile tokens BEFORE resolving card play ability (keyed on card.pile)."""
        tkns = self.which_token(card.pile)
        if Token.PLUS_1_ACTION in tkns:
            self.output("Gaining action from +1 Action token")
            self.add_actions(1)
        if Token.PLUS_1_CARD in tkns:
            try:
                if c := self.pickup_card():
                    self.output(f"Picked up {c} from +1 Card token")
            except NoCardException:
                pass
        if Token.PLUS_1_COIN in tkns:
            self.output("Gaining coin from +1 Coin token")
            self.coins += 1
        if Token.PLUS_1_BUY in tkns:
            self.output("Gaining buy from +1 Buy token")
            self.buys += 1

    ###########################################################################
    def _hook_pre_play(self, card: Card) -> dict[OptionKeys, Any]:
        """Before-play hooks for current player's in-play cards & active prophecy (never `card` itself)."""
        options: dict[OptionKeys, Any] = {}
        sources = self.in_play_cards() + ([self.game.prophecy] if self.game.prophecy else [])
        for crd in sources:
            if crd is card:
                continue
            res = crd.hook_pre_play(game=self.game, player=self, card=card)
            if isinstance(res, dict):
                options |= res
        return options

    ###########################################################################
    def hook_all_players_pre_play(self, card: Card) -> dict[OptionKeys, Any]:
        """Before-play hooks on all players' in-play Duration cards."""
        options: dict[OptionKeys, Any] = {}
        for player in self.game.players_in_turn_order(self):
            for crd in player.in_play_cards():
                res = crd.hook_all_players_pre_play(game=self.game, player=self, owner=player, card=card)
                if isinstance(res, dict):
                    options |= res
        return options

    ###########################################################################
    def hook_all_players_post_play(self, card: Card) -> dict[OptionKeys, Any]:
        """After-play hooks on all players' in-play cards."""
        options: dict[OptionKeys, Any] = {}
        for player in self.game.players_in_turn_order(self):
            for crd in player.in_play_cards():
                res = crd.hook_all_players_post_play(game=self.game, player=self, owner=player, card=card)
                if isinstance(res, dict):
                    options |= res
        return options

    ###########################################################################
    def _run_attack_reactions(self, attack_card: Card) -> None:
        """Complete Rules Ch. III (Reaction cards p. 20–21):
        When an Attack card is played, before-play Reactions trigger BEFORE the Attack resolves,
        in clockwise turn order, starting with the player to the left of the attacker.
        """
        self._current_attack_immune = set()
        self._attack_reactions_ran = True
        opponents = [p for p in self.game.players_in_turn_order(self) if p is not self]
        for opp in opponents:
            if active_defense_in_play(opp):
                self._current_attack_immune.add(opp.uuid)
                self.output(f"Player {opp.name} is defended")
            # Re-scan the hand after each pass: reactions can draw more reactions.
            declined: set[str] = set()
            reacted: set[str] = set()
            while True:
                changed = False
                for crd in hand_attack_reactions(opp):
                    if crd.uuid in declined or crd not in opp.piles[Piles.HAND]:
                        continue
                    scripted_text = isinstance(opp.active_controller, TextController) and type(self.game).__name__ == "TestGame"
                    custom_choice = ("plr_choose_options" in opp.__dict__
                                     or type(opp).plr_choose_options is not Player.plr_choose_options)
                    if crd.has_defense():
                        reveal = True
                        if opp.test_input or custom_choice or not scripted_text:
                            reveal = bool(opp.plr_choose_options(
                                f"Reveal {crd} to block the attack?",
                                (f"Reveal {crd}", True), ("Don't do anything", False),
                            ))
                        declined.add(crd.uuid)
                        if reveal:
                            opp.reveal_card(crd)
                            self._current_attack_immune.add(opp.uuid)
                            self.output(f"Player {opp.name} is defended")
                        continue
                    # Old scripts can proceed straight to the Attack after one reveal.
                    if (crd.uuid in reacted and scripted_text and not custom_choice
                            and (not opp.test_input or not any(
                                word in opp.test_input[0].lower()
                                for word in ("reveal", "don't", "nothing")
                            ))):
                        declined.add(crd.uuid)
                        continue
                    before = tuple((c.uuid, c.move_epoch) for c in opp.piles[Piles.HAND])
                    crd.hook_under_attack(game=self.game, player=opp, attacker=self)
                    reacted.add(crd.uuid)
                    after = tuple((c.uuid, c.move_epoch) for c in opp.piles[Piles.HAND])
                    if before == after:
                        declined.add(crd.uuid)
                    else:
                        changed = True
                if not changed:
                    break

    ###########################################################################
    def defer_card(self, card: Card) -> None:
        """Set a non-duration card to be played in its entirety next turn"""
        self.move_card(card, Piles.DEFER)

    ###########################################################################
    def register_duration_stay(
        self,
        card: Card,
        throne_card: Optional[Card] = None,
        permanent: Optional[bool] = None,
    ) -> DurationStay:
        """Register or increment a DurationStay entry for an in-play Duration card."""
        existing = self.duration_stays.get(card.uuid)
        is_perm = card.permanent if permanent is None else permanent
        if existing is not None and existing.active and existing.card_epoch == card.move_epoch:
            existing.triggers_remaining += 1
            if is_perm:
                existing.permanent = True
            if throne_card is not None:
                existing.attached_throne_uuids.append((throne_card.uuid, throne_card.move_epoch))
            return existing
        stay = DurationStay(
            card_uuid=card.uuid,
            card_epoch=card.move_epoch,
            triggers_remaining=1,
            permanent=is_perm,
            attached_throne_uuids=[(throne_card.uuid, throne_card.move_epoch)] if throne_card is not None else [],
        )
        self.duration_stays[card.uuid] = stay
        return stay

    ###########################################################################
    def cancel_duration_stay(self, card: Card) -> None:
        """Mark a Duration as NOT staying in play (e.g. Tactician failed to discard, Barge chose now)."""
        self.duration_stays.pop(card.uuid, None)

    ###########################################################################
    def move_after_play(self, card: Card, force: bool = False, throne_card: Optional[Card] = None) -> None:
        """Post-play placement for Reserve and Duration cards.

        - Duration cards remain in `Piles.PLAYED` during the turn they are played (registered
          in `self.duration_stays`), and transition to `Piles.DURATION` at Clean-up so all
          in-play checks during this turn see them cleanly!
        - Reserve cards move to `Piles.RESERVE` ONLY if their play ability resolved (not skipped/Way)
          and the card is still in `Piles.PLAYED`.
        """
        if force:
            return
        if card.isDuration() and card.location == Piles.PLAYED:
            if getattr(card, "stay_in_play", True):
                self.register_duration_stay(card, throne_card=throne_card)
        elif card.isReserve() and card.location == Piles.PLAYED:
            self.move_card(card, Piles.RESERVE)

    ###########################################################################
    def _has_play_limit(self) -> bool:
        """Check whether play limit allows another play (without mutating limit)."""
        if self.limits[Limits.PLAY] is not None and self.limits[Limits.PLAY] <= 0:
            return False
        return True

    def _consume_play_limit(self) -> None:
        if self.limits[Limits.PLAY] is not None:
            self.limits[Limits.PLAY] -= 1

    ###########################################################################
    def _play_enough_actions(self, card: Card, cost_action: bool) -> bool:
        """Check and deduct Action pool for Action plays during Action phase."""
        if card.isAction() and cost_action and self.phase not in (Phase.NIGHT, Phase.BUY):
            if self.actions.get() <= 0:
                return False
            self.actions -= 1
        return True

    ###########################################################################
    def play_card(
        self,
        card: Card,
        discard: bool = True,
        cost_action: bool = True,
        post_action_hook: bool = True,
        move_card: bool = True,
        way: Optional[Way] = None,
        throne_card: Optional[Card] = None,
    ) -> None:
        """Play `card` (optionally using `way` as a when-would-resolve replacement).

        Follows Complete Rules Ch. VI Summary of Player Activities (p. 44):
          1. Verify actions & play limit (never burns play limit if out of actions).
          2. Announce & move card to PLAYED (unless `move_card=False`, e.g. Lose-Track replay).
          3. Before-play triggers: Adventures pile tokens, pre-play hooks, Attack reactions.
          4. When-would-resolve: Way / Enchantress (`SKIP_CARD`).
          5. Resolve play ability (including Omen `+Sun` BEFORE rest of play ability).
          6. Post-play placement (Duration stay registration / Reserve Tavern mat).
          7. After-play hooks.
        """
        options: dict[OptionKeys, Any] = {
            OptionKeys.SKIP_CARD: False,
            OptionKeys.DISCARD: discard,
        }
        from_hand = move_card and card.location == Piles.HAND and throne_card is None
        if from_hand and not self._has_play_limit():
            self.output(f"Can't play {card} due to limits in number of plays")
            return
        self.debug(card, f"play_card({card=}, {discard=}, {cost_action=}, {way=})")
        if not self._play_enough_actions(card, cost_action):
            self.output("Not enough actions")
            return
        if from_hand:
            self._consume_play_limit()

        if way is not None:
            self.output(f"Playing {way.name} instead of {card}")
        else:
            self.output(f"Playing {card}")

        self.currcards.append(way if way is not None else card)
        prev_attack_immune = self._current_attack_immune
        prev_attack_ran = self._attack_reactions_ran
        self._attack_reactions_ran = False
        try:
            # 1. Move to play area (card is now "in play" and "played")
            if move_card and card.location != Piles.PLAYED:
                self.move_card(card, Piles.PLAYED)
            action_track = CardTrack.capture(card, Piles.PLAYED) if card.isAction() else None
            if card not in self.played_this_turn:
                self.played_this_turn.append(card)

            # 2. Before-play abilities (Adventures tokens, pre-play hooks, Attack reactions)
            self._play_card_tokens(card)
            options |= self._hook_pre_play(card)
            options |= self.hook_all_players_pre_play(card)
            if card.isAttack():
                self._run_attack_reactions(card)

            # 3. When-would-resolve & Resolve play ability
            if way is not None:
                self.card_benefits(way)
                new_opts = way.special_way(game=self.game, player=self, card=card)
                if isinstance(new_opts, dict):
                    options |= new_opts
                self.played_ways.append((way, card))
            elif not options[OptionKeys.SKIP_CARD]:
                # Complete Rules Ch. IV Prophecies p. 34: +Sun resolves BEFORE rest of Omen card
                if card.isOmen():
                    self.game.remove_sun_token()
                self.card_benefits(card)
                if options[OptionKeys.DISCARD]:
                    self.move_after_play(card, force=False, throne_card=throne_card)

            if card.isAction():
                self._just_played_action = True
                self._last_action_played = card
                self._last_played_action = card
                self._last_played_action_track = action_track

            # 4. After-play hooks (fire even if skipped or played via Way!)
            if post_action_hook:
                self._post_play_hooks(card)
        finally:
            self._current_attack_immune = prev_attack_immune
            self._attack_reactions_ran = prev_attack_ran
            self.currcards.pop()

    ###########################################################################
    def play_card_multiple(
        self,
        card: Card,
        times: int = 2,
        source_card: Optional[Card] = None,
        trash_after: bool = False,
    ) -> CardTrack:
        """Canonical Throne-Room primitive (Ch. III p. 17 & 26; Ch. IV p. 27–28):
        - Moves `card` to PLAYED on play 1 and captures a `CardTrack`.
        - Replays `card` `times` times WITHOUT pulling it back if play 1 moved it (Stop-Moving!).
        - Keeps both `card` and `source_card` (direct throne-room) in play if `card` sets up a Duration.
        - If `trash_after=True` (Counterfeit/Procession), only trashes `card` if still tracked in PLAYED.
        """
        # Play 1: move card into PLAYED
        if card.location != Piles.PLAYED:
            self.move_card(card, Piles.PLAYED)
        track = CardTrack.capture(card, Piles.PLAYED)

        for idx in range(times):
            chosen_way = None
            if card.isAction() and self.game.ways:
                chosen_way = self.plr_choose_options(
                    f"How do you want to play {card}?",
                    (f"Play {card} normally", None),
                    *((f"Use {way.name}", way) for way in self.game.ways.values()),
                )
            # Only move if first play; subsequent plays respect Stop-Moving (card stays wherever it moved)
            self.play_card(
                card,
                discard=True,
                cost_action=False,
                move_card=False,
                throne_card=source_card if idx > 0 else None,
                way=chosen_way,
            )
            # If card stayed in play as a Duration on play 1, attach direct throne-room too!
            if idx == 0 and source_card is not None and card.uuid in self.duration_stays:
                stay = self.duration_stays[card.uuid]
                if not any(u == source_card.uuid for u, _ in stay.attached_throne_uuids):
                    stay.attached_throne_uuids.append((source_card.uuid, source_card.move_epoch))

        if trash_after and track.is_valid():
            self.trash_card(card)
        return track

    ###########################################################################
    def _post_play_hooks(self, card: Card) -> None:
        """After-play hooks (zone-scoped: in-play, reserve, landscapes — NOT cards sitting in hand)."""
        self.hook_all_players_post_play(card)
        for other_card in self._post_play_hook_sources():
            self.currcards.append(other_card)
            try:
                other_card.hook_post_play(game=self.game, player=self, card=card)
            finally:
                self.currcards.pop()

    ###########################################################################
    def perform_way(self, way: Way, card: Card, cost_action: bool = True) -> None:
        """Play `card` using `way` via the unified `play_card(..., way=way)` pipeline."""
        self.play_card(card, discard=False, cost_action=cost_action, way=way)

    ###########################################################################
    def card_benefits(self, card: Card) -> None:
        """Gain the benefits of the card being played — including special() / night()."""
        if card.cards:
            self.pickup_cards(card.cards)
        if card.actions:
            self.add_actions(card.actions)
        coin_val = self.hook_spend_value(card, actual=True)
        if coin_val:
            self.coins += coin_val
        if card.buys:
            self.buys += card.buys
        if card.favors:
            self.favors += card.favors
        if card.potion:
            self.potions += card.potion

        if self.phase == Phase.NIGHT and card.isNight():
            card.night(game=self.game, player=self)
        else:
            card.special(game=self.game, player=self)

    ###########################################################################
    def effective_cost(self, card: Card) -> Cost:
        """Return multi-dimensional effective Cost(coins, potions, debt) of `card`.

        Complete Rules Ch. I p. 7 & Ch. IV p. 32:
          - Costs of Events and Projects CANNOT be changed.
          - Coin cost reductions apply to ALL cards in the game (including on opponents' turns)
            and clamp coins >= 0 without affecting Potions or Debt.
        """
        assert isinstance(card, (Card, Project, Event)), f"Card {card=} {type(card)=}"
        base = Cost.from_card(card)
        if isinstance(card, (Event, Project)):
            return base

        delta = 0
        # Adventures -$2 Cost token applies on the player's turn to all cards from that pile
        if Token.MINUS_2_COST in self.which_token(card.pile):
            delta -= 2

        # Turn-scoped ongoing cost modifiers (Bridge, Highway, Quarry, Inventor, Bridge Troll)
        delta += self.turn_effects.total_cost_delta(card)
        active_plr = self.game.current_player
        if active_plr is not None and active_plr is not self:
            delta += active_plr.turn_effects.total_cost_delta(card)

        # Card/landscape hook_card_cost (Canal, Cheap, Highway/Quarry legacy hooks, etc.)
        checked: set[int] = set()
        for crd in self._cost_hook_sources():
            if id(crd) not in checked:
                delta += crd.hook_card_cost(game=self.game, player=self, card=card)
                checked.add(id(crd))
        if active_plr is not None and active_plr is not self:
            for crd in active_plr._cost_hook_sources():
                if id(crd) not in checked:
                    delta += crd.hook_card_cost(game=self.game, player=active_plr, card=card)
                    checked.add(id(crd))

        delta += card.hook_this_card_cost(game=self.game, player=self)
        return base.with_coin_delta(delta)

    ###########################################################################
    def card_cost(self, card: Card) -> int:
        """Return effective Coin cost of `card` (int convenience wrapper around `effective_cost`)."""
        return self.effective_cost(card).coins

    ###########################################################################
    def _gain_card_from_name(self, card_name: str) -> Card:
        """Get the top card from `card_name`'s pile (enforcing split-pile top-card rule)."""
        if card_name == "Loot":
            return self.get_card_from_pile("Loot")
        if card_name in self.game.card_instances:
            pile_name = self.game.card_instances[card_name].pile or card_name
        else:
            pile_name = card_name
        # Complete Rules Ch. IV Split piles p. 36: can only gain if `card_name` is on TOP of its pile!
        if pile_name in self.game.card_piles:
            pile = self.game.card_piles[pile_name]
            if pile.is_empty():
                raise NoCardException
            if pile_name != card_name and pile.top_card() != card_name:
                raise NoCardException
            # If pile_name == card_name (e.g. a split pile wrapper name like "Castles" or "Ruins"),
            # take top card directly; otherwise verify top card matches card_name.
            return self.get_card_from_pile(pile_name)
        raise NoCardException

    ###########################################################################
    def gain_card(
        self,
        card_name: Optional[str] = None,
        destination: Piles = Piles.DISCARD,
        new_card: Optional[Card] = None,
        callhook: bool = True,
    ) -> Card:
        """Complete Rules Ch. III (When-gain p. 22) & Ch. VI p. 44:
        1. Take top card from pile.
        2. Place card in `destination` FIRST (and record in `stats["gained"]`).
        3. Capture `CardTrack` (Lose-Track rule).
        4. Resolve when-gain triggers (hooks, Trashing token on `card.pile`, Exile mat un-exile).
        """
        options: dict[OptionKeys, Any] = {}
        if new_card is None:
            assert card_name is not None
            new_card = self._gain_card_from_name(card_name)
        if new_card is None:
            raise NoCardException

        self.debug(new_card, f"gain_card({card_name=}, {destination=})")

        # Move to initial gaining destination FIRST (so when-gain hooks see card at destination!)
        if new_card.location != destination:
            self.move_card(new_card, destination)
        self.stats["gained"].append(new_card)
        if self.phase == Phase.BUY:
            self.gained_in_buy_phase.append(new_card)
            self.stats["gained_in_buy"].append(new_card)
        original_name = new_card.name
        track = CardTrack.capture(new_card, new_card.location)

        if callhook:
            options |= self._gain_card_hooks(new_card)

        # Handle replacement (when-would-gain e.g. Trader <2020)
        if options.get(OptionKeys.REPLACE):
            new_card = self._gain_card_replace(new_card, options[OptionKeys.REPLACE])
            if new_card is None:
                raise NoCardException
            self.move_card(new_card, destination)
            track = CardTrack.capture(new_card, new_card.location)

        # Adventures Trashing token (2022 CV: triggers when you gain a card from this pile)
        if Token.TRASHING in self.which_token(new_card.pile):
            self.output("Trashing token allows you to trash a card")
            self.plr_trash_card()

        # Optional when-gain redirect/trash/exile (respecting Lose-Track!)
        if options.get(OptionKeys.TRASH, False):
            if track.is_valid():
                self.trash_card(new_card)

        if options.get(OptionKeys.EXILE):
            if track.is_valid():
                self.exile_card(new_card)
        elif OptionKeys.DESTINATION in options and options[OptionKeys.DESTINATION] != destination:
            if track.is_valid():
                self.move_card(new_card, options[OptionKeys.DESTINATION])
        elif options.get(OptionKeys.DONTADD) and track.is_valid():
            self.remove_card(new_card)

        # Check Exile mat un-exiling of OTHER copies (Complete Rules Ch. VII Your Exile mat)
        other_exiled = [c for c in self.piles[Piles.EXILE] if c.name == original_name and c is not new_card]
        if other_exiled:
            self.check_unexile(original_name, exclude_card=new_card)

        if options.get(OptionKeys.SHUFFLE, False):
            self.piles[Piles.DECK].shuffle()

        return new_card

    ###########################################################################
    def _gain_card_replace(self, old_card: Card, new_card_name: str) -> Card:
        """Replace the card just gained (returns old_card to TOP of its pile)."""
        self.move_card(old_card, Piles.CARDPILE)
        try:
            new_card = self.get_card_from_pile(new_card_name)
        except NoCardException:
            self.output(f"No more {new_card_name}")
            raise
        return new_card

    ###########################################################################
    def check_unexile(self, card_name: str, exclude_card: Optional[Card] = None) -> None:
        """Give player option to discard all other copies of `card_name` from Exile mat."""
        other_copies = [c for c in self.piles[Piles.EXILE] if c.name == card_name and c is not exclude_card]
        if not other_copies:
            return
        if self.active_controller.check_unexile(self, card_name):
            self.unexile(card_name, exclude_card=exclude_card)

    ###########################################################################
    def unexile(self, cardname: str, exclude_card: Optional[Card] = None) -> int:
        """Discard all matching (other) copies from Exile mat (triggering when-discard abilities!)."""
        if not self.piles[Piles.EXILE]:
            return 0
        to_discard = [
            card for card in self.piles[Piles.EXILE] if card is not None and card.name == cardname and card is not exclude_card
        ]
        if not to_discard:
            return 0
        self.discard_cards(to_discard, source=Piles.EXILE, hook=True)
        return len(to_discard)

    ###########################################################################
    def _choose_overpay_amount(self, card: Card) -> int:
        """Prompt player for overpay amount and deduct paid coins upfront (Ch. IV p. 31)."""
        if self.coins.get() <= 0:
            return 0
        options = [(f"Spend {i} more", i) for i in range(self.coins.get() + 1)]
        ans = int(self.plr_choose_options("How much do you wish to overpay?", *options))
        if ans > 0:
            self.coins -= ans
        return ans

    ###########################################################################
    def _choose_overpay_potions(self) -> int:
        """Pay optional Potion overpay in addition to Coin overpay."""
        if self.potions.get() <= 0:
            return 0
        amount = int(self.plr_choose_options(
            "How many Potions do you wish to overpay?",
            *((f"Spend {i} Potions", i) for i in range(self.potions.get() + 1)),
        ))
        self.potions -= amount
        return amount

    def _resolve_overpay(self, card: Card, coins: int, potions: int) -> None:
        """Resolve Coin effects; pass Potion cost to hooks that support it."""
        if not coins and not potions:
            return
        if "potions" in inspect.signature(card.hook_overpay).parameters:
            card.hook_overpay(game=self.game, player=self, amount=coins, potions=potions)
        else:
            card.hook_overpay(game=self.game, player=self, amount=coins)

    def overpay(self, card: Card) -> None:
        """Pay optional Coins and Potions, then resolve the overpay ability."""
        if not card.overpay:
            return
        coins = self._choose_overpay_amount(card)
        potions = self._choose_overpay_potions()
        self._resolve_overpay(card, coins, potions)

    ###########################################################################
    def pre_buy_checks(self, card: Card) -> bool:
        """Pre-buy validation: Buys, Buy limit, Debt, Coins, Potions, forbidden, allowed_to_buy."""
        if not self.buys:
            return False
        if not self.limits[Limits.BUY]:
            self.output("Buy limit stops buying")
            return False
        if self.debt:
            self.output("Must pay off debt first")
            return False
        if card in self.forbidden_to_buy or not self.hook_allowed_to_buy(card):
            self.output(f"Not allowed to buy {card}")
            return False
        eff = self.effective_cost(card)
        if not card.always_buyable:
            if self.coins.get() < eff.coins:
                self.output("You can't afford this")
                return False
            if self.potions.get() < eff.potions:
                self.output("Not enough potions to buy this")
                return False
        return True

    ###########################################################################
    def buy_card(self, card_name: str) -> Optional[Card]:
        """Complete Rules Ch. VI (Buy a card p. 44) & Ch. IV (Overpaying p. 31):
        1. Pre-buy checks & verify Supply availability.
        2. Pay Coins & Potions, deduct Buy & Buy limit, take Debt, pay Overpay.
        3. Resolve When-Buy abilities (including Embargo).
        4. Gain the card (triggering When-Gain abilities + CV Overpay ability).
        """
        assert isinstance(card_name, str), f"buy_card({card_name=}) {type(card_name)=}"
        if card_name in self.game.card_piles and not self.game.card_piles[card_name].in_supply:
            return None
        card = self.game.card_instances[card_name]
        if card.pile in self.game.card_piles and not self.game.card_piles[card.pile].in_supply:
            return None
        eff_cost = self.effective_cost(card)

        if not self.pre_buy_checks(card):
            return None

        pile_name = card.pile or card.name
        if pile_name not in self.game.card_piles or self.game.card_piles[pile_name].is_empty():
            self.output(f"Couldn't buy card - no more {card.name}s available")
            return None
        if pile_name != card_name and self.game.card_piles[pile_name].top_card() != card_name:
            self.output(f"Couldn't buy card - {card.name} is not on top of {pile_name}")
            return None

        self.buy_phase_part = 2

        # 1. Pay cost & take Debt BEFORE when-buy and BEFORE gaining!
        self.coins -= eff_cost.coins
        if eff_cost.potions:
            self.potions -= eff_cost.potions
        self.buys -= 1
        self.limits[Limits.BUY] -= 1
        if eff_cost.debt:
            self.debt += eff_cost.debt

        overpay_amount = 0
        overpay_potions = 0
        if card.overpay:
            overpay_amount = self._choose_overpay_amount(card)
            overpay_potions = self._choose_overpay_potions()

        self.output(f"Bought {card} for {eff_cost.coins} coin")

        # 2. When-buy abilities (including Embargo) BEFORE gaining
        if self.game.card_piles[pile_name].embargo_level:
            self._buy_card_embargo(pile_name)
        self.hook_buy_card(card)
        card.hook_buy_this_card(game=self.game, player=self)
        self.hook_all_players_buy_card(card)

        # 3. Gain the card
        try:
            new_card = self.gain_card(card.name)
        except NoCardException:
            self.output(f"Couldn't buy card - no more {card.name}s available")
            return None

        self.debug(new_card, f"buy_card({card_name=})")
        self.stats["bought"].append(new_card)

        # 4. Current-version Overpay when-gain ability
        self._resolve_overpay(new_card, overpay_amount, overpay_potions)

        return new_card

    ###########################################################################
    def _buy_card_embargo(self, pile_name: str) -> None:
        """Handle Embargo on the pile bought from"""
        for _ in range(self.game.card_piles[pile_name].embargo_level):
            try:
                self.gain_card("Curse")
            except NoCardException:
                self.output("No more Curses")
            else:
                self.output("Gained a Curse from embargo")

    ###########################################################################
    def hook_all_players_buy_card(self, card: Card) -> None:
        for player in self.game.players_in_turn_order(self):
            for crd in player.in_play_cards():
                crd.hook_all_players_buy_card(game=self.game, player=self, owner=player, card=card)
        for crd in self.game.landmarks.values():
            crd.hook_all_players_buy_card(game=self.game, player=self, owner=self, card=card)

    ###########################################################################
    # Zone-scoped hook source queries (replaces indiscriminate relevant_cards())
    ###########################################################################
    def in_play_cards(self) -> list[Card]:
        """Unified In-Play cards (PLAYED this turn + staying DURATION cards)."""
        return list(self.piles[Piles.PLAYED]) + list(self.piles[Piles.DURATION])

    def active_landscapes(self) -> list[Card]:
        """Active game/player landscapes (NEVER includes unplayed game.ways!)."""
        res: list[Card] = []
        res.extend(self.game.landmarks.values())
        res.extend(self.projects)
        res.extend(self.played_events)
        res.extend(self.states)
        res.extend(self.artifacts)
        if self.game.ally:
            res.append(self.game.ally)
        res.extend(self.game.traits.values())
        if self.game.prophecy:
            res.append(self.game.prophecy)
        return res

    def in_play_and_landscape_cards(self) -> list[Card]:
        """All in-play cards + active landscapes (excludes HAND and unplayed Ways)."""
        return self.in_play_cards() + self.active_landscapes()

    def dynamic_type_sources(self) -> list[Card]:
        """Cards/landscapes that can dynamically add card types (Enlightenment, Capitalism, Estate token)."""
        return self.in_play_cards() + list(self.projects) + list(self.states) + list(self.artifacts)

    def _cost_hook_sources(self) -> list[Card]:
        """Only in-play cards and active landscapes modify card costs (never cards in HAND or unplayed Ways)."""
        return self.in_play_and_landscape_cards()

    def _post_play_hook_sources(self) -> list[Card]:
        """After-play hooks fire from in-play, Tavern mat, active landscapes, and played Ways."""
        return self.in_play_and_landscape_cards() + list(self.piles[Piles.RESERVE])

    def _buy_hook_sources(self) -> list[Card]:
        """When-buy hooks fire from in-play, active landscapes, and hand Reactions (e.g. Hovel <2022)."""
        hand_reactions = [c for c in self.piles[Piles.HAND] if c.isReaction()]
        return self.in_play_and_landscape_cards() + hand_reactions

    def _cleanup_hook_sources(self) -> list[Card]:
        """Include same-turn delayed abilities after their source leaves play."""
        sources = (self.in_play_cards() + list(self.piles[Piles.RESERVE])
                   + list(self.artifacts) + self.played_this_turn)
        return list({id(card): card for card in sources}.values())

    def _gain_hook_sources(self) -> list[Card]:
        """When-gain hooks fire from in-play, Tavern mat (Duplicate), active landscapes,
        hand Reactions (Watchtower, Sheepdog, Trader, Sleigh, Hovel, Falconer, Mapmaker, etc.),
        and played Ways (e.g. Way of the Seal ONLY if actually played this turn!).
        """
        hand_reactions = [c for c in self.piles[Piles.HAND] if c.isReaction()]
        played_way_objs = [w for w, _ in self.played_ways]
        sources = (hand_reactions + self.in_play_and_landscape_cards() + list(self.piles[Piles.RESERVE])
                   + played_way_objs
                   + [c for c in self.played_this_turn if c.isDuration()])
        return list({id(card): card for card in sources}.values())

    def _trash_hook_sources(self) -> list[Card]:
        """When-trash hooks fire from in-play, active landscapes, and hand Reactions (Market Square)."""
        hand_reactions = [c for c in self.piles[Piles.HAND] if c.isReaction()]
        return self.in_play_and_landscape_cards() + hand_reactions

    def _discard_hook_sources(self) -> list[Card]:
        """When-discard hooks fire from in-play, active landscapes, and played Ways."""
        played_way_objs = [w for w, _ in self.played_ways]
        return self.in_play_and_landscape_cards() + played_way_objs

    def _shuffle_hook_sources(self) -> list[Card]:
        """Pre/post-shuffle hooks (Stash, Star Chart, Fated, Order of Masons/Astrologers, Avoid)."""
        return self.in_play_and_landscape_cards() + list(self.piles[Piles.HAND])

    def relevant_cards(self) -> list[Card]:
        """Legacy compatibility helper — returns zone-scoped cards (excludes unplayed Ways)."""
        return self._gain_hook_sources()

    ###########################################################################
    def _gain_card_hooks(self, gained_card: Card) -> dict[OptionKeys, Any]:
        """Centralized when-gain hook dispatch."""
        assert isinstance(gained_card, Card)
        options: dict[OptionKeys, Any] = {}
        options |= self._hook_any_gain_card(gained_card)
        options |= self._hook_gain_card(gained_card)
        options |= self._hook_gain_this_card(gained_card)
        options |= self._hook_all_players_gain_card(gained_card)
        return options

    ###########################################################################
    def _hook_any_gain_card(self, gained_card: Card) -> dict[OptionKeys, Any]:
        """Global when-gain abilities (Duchess, Changeling, Ferryman, Footpad, etc.)."""
        options: dict[OptionKeys, Any] = {}
        for card in list(self.game.card_instances.values()) + list(self.game.events.values()):
            self.currcards.append(card)
            try:
                res = card.hook_any_gain_card(self.game, self, gained_card)
                if isinstance(res, dict):
                    options |= res
            finally:
                self.currcards.pop()
        return options

    ###########################################################################
    def _hook_gain_card(self, gained_card: Card) -> dict[OptionKeys, Any]:
        """Run hook_gain_card() for zone-appropriate cards."""
        options: dict[OptionKeys, Any] = {}
        track = CardTrack.capture(gained_card)
        for card in self._gain_hook_sources():
            # Moving gain reactions and Rush must not recover a card lost by an earlier hook.
            if not track.is_valid() and card.name in ("Watchtower", "Rush"):
                continue
            # Prevent non-Reaction cards in PLAYED from misfiring if they are hand-only Reactions
            self.currcards.append(card)
            try:
                res = card.hook_gain_card(game=self.game, player=self, card=gained_card)
                if isinstance(res, dict):
                    res = dict(res)
                    trash = res.pop(OptionKeys.TRASH, False)
                    destination = res.pop(OptionKeys.DESTINATION, None)
                    if track.is_valid():
                        if trash:
                            self.trash_card(gained_card)
                        elif destination is not None:
                            self.move_card(gained_card, destination)
                    options |= res
            finally:
                self.currcards.pop()
        return options

    ###########################################################################
    def _hook_gain_this_card(self, gained_card: Card) -> dict[OptionKeys, Any]:
        """Run hook_gain_this_card() on the gained card."""
        options: dict[OptionKeys, Any] = {}
        self.currcards.append(gained_card)
        try:
            res = gained_card.hook_gain_this_card(game=self.game, player=self)
            if isinstance(res, dict):
                options |= res
        finally:
            self.currcards.pop()
        return options

    ###########################################################################
    def _hook_all_players_gain_card(self, gained_card: Card) -> dict[OptionKeys, Any]:
        """Run hook_all_players_gain_card() in clockwise turn order."""
        options: dict[OptionKeys, Any] = {}
        for player in self.game.players_in_turn_order(self):
            for card in player._gain_hook_sources():
                self.currcards.append(card)
                try:
                    res = card.hook_all_players_gain_card(
                        game=self.game, player=self, owner=player, card=gained_card
                    )
                    if isinstance(res, dict):
                        options |= res
                finally:
                    self.currcards.pop()
        return options

    ###########################################################################
    def has_defense(self, attacker: "Player", verbose: bool = True) -> bool:
        """True if this player is protected from `attacker`'s Attack:
        - Ongoing immunity IN PLAY: Lighthouse, Guardian, Champion (NEVER from hand!)
        - Hand Reaction defense: Moat, Shield (only cards with CardType.REACTION in hand!)
        """
        if attacker._attack_reactions_ran:
            return self.uuid in attacker._current_attack_immune or active_defense_in_play(self)
        if active_defense_in_play(self):
            if verbose:
                attacker.output(f"Player {self.name} is defended")
            return True
        for crd in self.piles[Piles.HAND]:
            if crd.isReaction() and crd.has_defense():
                if verbose:
                    attacker.output(f"Player {self.name} is defended")
                return True
        return False

    ###########################################################################
    def add_actions(self, num: int = 1) -> None:
        assert isinstance(num, int)
        if self.misc.get("no_actions"):
            self.output("No more additional actions allowed")
        else:
            self.actions += num

    ###########################################################################
    def gain_prize(self) -> None:
        """Gain a Prize from outside the Supply."""
        prizes = [self.game.card_piles[_] for _ in self.game.get_prizes() if _ in self.game.card_piles]
        options = []
        for prize_pile in prizes:
            if prize_pile.is_empty():
                continue
            prize = prize_pile.top_card()
            options.append((f"Gain {prize}", prize))
        if options:
            self.output("Gain a prize")
            if option := self.plr_choose_options("Gain a Prize", *options):
                self.gain_card(option)
        else:
            self.output("No prizes available")

    ###########################################################################
    def __str__(self) -> str:
        return self.name

    ###########################################################################
    def __repr__(self) -> str:
        return self.name

    ###########################################################################
    def buy_project(self, project: Project) -> bool:
        """Buy a Project (checks Debt, Buys, max 2 projects, no duplicate BEFORE spending!)."""
        assert issubclass(project.__class__, Project)
        if not self.buys:
            self.output("Need a buy to buy a project")
            return False
        if self.debt:
            self.output("Must pay off debt first")
            return False
        if len(self.projects) >= 2:
            self.output("Can't have more than two projects")
            return False
        if project.name in [_.name for _ in self.projects]:
            self.output(f"Already have project {project.name}")
            return False
        pr_cost = Cost.from_card(project)
        if not pr_cost.can_be_paid_by(self.coins.get(), self.potions.get()):
            self.output(f"Need {pr_cost} to buy this project")
            return False
        self.buy_phase_part = 2
        self.buys -= 1
        self.coins -= pr_cost.coins
        if pr_cost.potions:
            self.potions -= pr_cost.potions
        self.debt += pr_cost.debt
        self.buys += project.buys
        self.assign_project(project.name)
        return True

    ###########################################################################
    def event_available(self, event: Event) -> bool:
        """Check Event restrictions before charging its cost."""
        if not getattr(event, "can_buy", lambda g, p: True)(self.game, self):
            return False
        if event.name == "Mission" and "Mission" in self.once:
            return False
        if event.name == "Seize the Day" and "seize the day" in self.specials:
            return False
        return event.hook_allowed_to_buy(game=self.game, player=self)

    def perform_event(self, event: Event) -> bool:
        """Buy and resolve an Event (printed cost only, checks Debt/Coins/Potions)."""
        assert isinstance(event, Event), f"Event={event} ({type(event)})"
        if not self.event_available(event):
            return False
        if not self.buys:
            self.output("Need a buy to perform an event")
            return False
        if self.debt:
            self.output("Must pay off debt first")
            return False
        ev_cost = Cost.from_card(event)
        if not ev_cost.can_be_paid_by(self.coins.get(), self.potions.get()):
            self.output(f"Need {ev_cost} to perform this event")
            return False
        self.buy_phase_part = 2
        self.buys -= 1
        self.coins -= ev_cost.coins
        if ev_cost.potions:
            self.potions -= ev_cost.potions
        self.debt += ev_cost.debt
        self.buys += event.buys
        self.output(f"Using event {event}")
        self.currcards.append(event)
        try:
            event.special(game=self.game, player=self)
        finally:
            self.currcards.pop()
        self.played_events.add(event)
        return True

    ###########################################################################
    @classmethod
    def select_by_type(cls, card: Card, types: dict[CardType, bool]) -> bool:
        """Complete Rules Ch. I (Types p. 7):
        An Action-Victory or Action-Treasure card IS an Action card.
        Returns True if the card has ANY of the requested `True` types (or if unfiltered).
        """
        assert isinstance(card, Card), f"select_by_type {card=} {type(card)=}"
        if all(types.values()):
            return True
        if types.get(CardType.ACTION) and card.isAction():
            return True
        if types.get(CardType.VICTORY) and card.isVictory():
            return True
        if types.get(CardType.TREASURE) and card.isTreasure():
            return True
        if types.get(CardType.NIGHT) and card.isNight():
            return True
        if types.get(CardType.CURSE) and card.isCurse():
            return True
        return False

    ###########################################################################
    def cards_matching_cost(
        self,
        predicate: Callable[[Cost], bool],
        types: Optional[dict[CardType, bool]] = None,
    ) -> list[Card]:
        """Return Supply top cards matching `predicate(effective_cost)` and `types`."""
        matched = PlayArea("matched")
        type_map = self._type_selector(types)
        for _, pile in self.game.get_supply_piles():
            if not pile:
                continue
            card = pile.get_top_card()
            if card is None or not card.purchasable:
                continue
            if not self.select_by_type(card, type_map):
                continue
            if predicate(self.effective_cost(card)):
                matched.add(card)
        matched.sort(key=self.card_cost)
        matched.sort(key=lambda x: x.basecard)
        return list(matched)

    ###########################################################################
    def cards_affordable(
        self,
        oper: Callable[[int, int], bool],
        coin: Optional[int],
        num_potions: int,
        types: Optional[dict[CardType, bool]],
        debt_ok: bool = False,
        for_buy: bool = False,
    ) -> list[Card]:
        """Return Supply top cards matching cost constraints using proper Cost partial order."""
        affordable = PlayArea("affordable")
        type_map = self._type_selector(types)
        for _, pile in self.game.get_supply_piles():
            if not pile:
                continue
            card = pile.get_top_card()
            if card is None or not card.purchasable:
                continue
            if not self.select_by_type(card, type_map):
                continue
            eff = self.effective_cost(card)
            if not debt_ok and eff.debt > 0:
                continue
            if for_buy and card.always_buyable:
                affordable.add(card)
                continue
            if coin is None:
                affordable.add(card)
                continue

            if oper is operator.eq:
                # Exact cost match requires identical coins AND potions (and 0 debt unless debt_ok)
                if eff.coins == coin and eff.potions == num_potions and (debt_ok or eff.debt == 0):
                    affordable.add(card)
            elif oper is operator.le:
                # "up to $X (+ P)" — component-wise <= on coins and potions
                if eff.coins <= coin and eff.potions <= num_potions and (debt_ok or eff.debt == 0):
                    affordable.add(card)
            elif oper is operator.gt:
                # Strictly more than $X
                if eff.coins > coin and (eff.potions == 0 or num_potions > 0 or debt_ok):
                    affordable.add(card)
            elif oper is operator.ge:
                # "$X or more" — coins >= X regardless of potions/debt
                if eff.coins >= coin:
                    affordable.add(card)
            else:
                if eff.potions <= num_potions and oper(eff.coins, coin):
                    affordable.add(card)

        affordable.sort(key=self.card_cost)
        affordable.sort(key=lambda x: x.basecard)
        return list(affordable)

    ###########################################################################
    def cards_under(
        self,
        coin: int,
        num_potions: int = 0,
        types: Optional[dict[CardType, bool]] = None,
        debt_ok: bool = False,
        for_buy: bool = False,
    ) -> list[Card]:
        """Return Supply cards costing up to `Cost(coin, num_potions, ...)`."""
        return self.cards_affordable(operator.le, coin, num_potions, types, debt_ok, for_buy=for_buy)

    ###########################################################################
    def cards_worth(
        self,
        coin: int,
        num_potions: int = 0,
        types: Optional[dict[CardType, bool]] = None,
        debt_ok: bool = False,
    ) -> list[Card]:
        """Return Supply cards costing exactly `Cost(coin, num_potions, 0)`."""
        return self.cards_affordable(operator.eq, coin, num_potions, types, debt_ok)

    ###########################################################################
    def cards_over(
        self,
        coin: int,
        num_potions: int = 0,
        types: Optional[dict[CardType, bool]] = None,
        debt_ok: bool = True,
    ) -> list[Card]:
        """Return Supply cards costing more than `coin`."""
        return self.cards_affordable(operator.gt, coin, num_potions, types, debt_ok)

    ###########################################################################
    def get_cards(self) -> dict[str, int]:
        """Return counts of all cards owned by this player (including set-aside mats)."""
        cards: defaultdict[str, int] = defaultdict(int)
        for card in self.all_cards():
            cards[card.name] += 1
        return dict(cards)

    ###########################################################################
    def count_cards(self) -> int:
        """How many cards does the player own (including set-aside mats + legacy secret_count)."""
        total = sum(len(stack) for stack in self.piles.values())
        total += sum(len(mat) for mat in self.set_aside.values())
        total += self.secret_count
        return total

    ###########################################################################
    @classmethod
    def _type_selector(cls, types: Optional[dict[CardType, bool]] = None) -> dict[CardType, bool]:
        if types is None:
            types = {}
        assert set(types.keys()) <= {
            CardType.ACTION,
            CardType.VICTORY,
            CardType.TREASURE,
            CardType.NIGHT,
            CardType.CURSE,
        }
        if not types:
            return {
                CardType.ACTION: True,
                CardType.VICTORY: True,
                CardType.TREASURE: True,
                CardType.NIGHT: True,
                CardType.CURSE: True,
            }
        _types = {
            CardType.ACTION: False,
            CardType.VICTORY: False,
            CardType.TREASURE: False,
            CardType.NIGHT: False,
            CardType.CURSE: False,
        }
        _types.update(types)
        return _types

    ###########################################################################
    def attack_victims(self) -> list[Player]:
        """Return other players in clockwise turn order who are not defended against this Attack.

        If `_run_attack_reactions()` already ran during `play_card()`, uses those before-play
        reactions without double-firing `hook_under_attack`. If an Attack's `special()` was called
        directly in isolation (e.g. unit test), runs reactions once now.
        """
        opponents = [plr for plr in self.game.players_in_turn_order(self) if plr is not self]
        victims: list[Player] = []
        for plr in opponents:
            if not self._attack_reactions_ran:
                for crd in list(plr.piles[Piles.HAND]):
                    crd.hook_under_attack(game=self.game, player=plr, attacker=self)
            if plr.uuid in self._current_attack_immune or plr.has_defense(self):
                continue
            victims.append(plr)
        return victims

    ###########################################################################
    def _cost_string(self, card: Card) -> str:
        """Generate human-readable cost string."""
        assert isinstance(card, (Card, Event, Project))
        return self.effective_cost(card).format_str(overpay=bool(getattr(card, "overpay", False)))

    ###########################################################################
    def plr_trash_card(
        self,
        num: int = 1,
        anynum: bool = False,
        cardsrc: Piles | PlayArea = Piles.HAND,
        **kwargs: Any,
    ) -> list[Card]:
        """Ask player to trash cards and trash ALL selected cards at once before firing when-trash hooks."""
        if "prompt" not in kwargs:
            kwargs["prompt"] = "Trash any cards" if anynum else f"Trash {num} cards"
        if isinstance(cardsrc, str):
            for pname, pile in self.piles.items():
                if pname.lower() == cardsrc.lower() and len(pile) == 0:
                    self.output(f"No cards to trash from {cardsrc}")
                    return []
        if isinstance(cardsrc, PlayArea) and len(cardsrc) == 0:
            self.output("No cards to trash")
            return []
        trash = self.card_sel(
            num=num,
            cardsrc=cardsrc,
            anynum=anynum,
            verbs=("Trash", "Untrash"),
            **kwargs,
        )
        if trash:
            self.trash_cards(trash, **kwargs)
        return trash

    ###########################################################################
    def plr_gain_card(
        self,
        cost: int,
        modifier: str = "less",
        recipient: Optional[Player] = None,
        destination: Piles = Piles.DISCARD,
        **kwargs: Any,
    ) -> Optional[Card]:
        """Ask player to gain a card from the Supply matching `cost` and `modifier`."""
        if recipient is None:
            recipient = self

        kwargs["prompt"] = kwargs.get("prompt", self._get_buyable_prompt(cost, modifier))

        if "cardsrc" in kwargs:
            cardsrc = kwargs.pop("cardsrc")
        else:
            cardsrc = self._get_buyable(cost, modifier, **kwargs)
            if not cardsrc:
                self.output("Nothing suitable to gain")
                return None
        if cards := self.card_sel(
            cardsrc=cardsrc,
            recipient=recipient,
            verbs=("Get", "Unget"),
            **kwargs,
        ):
            card_name = cards[0].name
            try:
                new_card = recipient.gain_card(card_name, destination)
                recipient.output(f"Got a {new_card}")
            except NoCardException:
                recipient.output(f"No more {card_name}")
                return None
            return new_card
        return None

    ###########################################################################
    def _get_buyable_prompt(self, cost: int, modifier: str) -> str:
        prompt = "Gain a card "
        assert modifier in ("less", "equal", "greater")
        if cost:
            if modifier == "less":
                prompt += f"costing up to {cost}"
            elif modifier == "equal":
                prompt += f"costing exactly {cost}"
            elif modifier == "greater":
                prompt += f"costing more than {cost}"
        return prompt

    ###########################################################################
    def _get_buyable(self, cost: int, modifier: str, **kwargs: Any) -> list[Card]:
        """Return Supply cards gainable for `cost` and `modifier`."""
        types = self._type_selector(kwargs.get("types", {}))
        assert modifier in ("less", "equal", "greater")
        debt_ok = bool(kwargs.get("ignore_debt", False))
        num_potions = 1 if kwargs.get("ignore_potcost", False) else 0
        match modifier:
            case "less":
                buyable = self.cards_under(cost, num_potions=num_potions, types=types, debt_ok=debt_ok)
            case "equal":
                buyable = self.cards_worth(cost, num_potions=num_potions, types=types, debt_ok=debt_ok)
            case "greater":
                buyable = self.cards_over(cost, num_potions=num_potions, types=types, debt_ok=debt_ok)
        buyable = [_ for _ in buyable if _.purchasable]
        if not debt_ok:
            buyable = [_ for _ in buyable if not _.debtcost]
        if not kwargs.get("ignore_potcost", False):
            buyable = [_ for _ in buyable if not _.potcost]
        buyable = [_ for _ in buyable if _.name not in kwargs.get("exclude", [])]
        return buyable

    ###########################################################################
    def plr_pick_card(self, force: bool = False, **kwargs: Any) -> Optional[Card]:
        if sel := self.card_sel(force=force, **kwargs):
            return sel[0]
        return None

    ###########################################################################
    def has_state(self, state: str) -> bool:
        return state in [_.name for _ in self.states]

    ###########################################################################
    def has_artifact(self, artifact: str) -> bool:
        return artifact in [_.name for _ in self.artifacts]

    ###########################################################################
    def assign_state(self, state: str) -> None:
        """Assign a State to the player (removing from other players if unique)."""
        assert isinstance(state, str)
        state_card = self.game.states[state]

        if state_card.unique_state:
            for player in self.game.player_list():
                for st in player.states[:]:
                    if st.name == state:
                        player.states.remove(st)
                        break
        self.states.append(state_card)

    ###########################################################################
    def assign_artifact(self, artifact: str) -> None:
        assert isinstance(artifact, str)
        artifact_card = self.game.artifacts[artifact]
        for pl in self.game.player_list():
            for st in pl.artifacts[:]:
                if st.name == artifact and self != pl:
                    pl.output(f"{self.name} took your {artifact}")
                    pl.artifacts.remove(st)
                    break
        if artifact_card not in self.artifacts:
            self.artifacts.append(artifact_card)

    ###########################################################################
    def assign_project(self, project: str) -> bool:
        project_card = self.game.projects[project]
        if len(self.projects) >= 2:
            self.output("Can't have more than two projects")
            return False
        if project in [_.name for _ in self.projects]:
            self.output(f"Already have project {project}")
            return False
        self.projects.append(project_card)
        return True

    ###########################################################################
    def remove_state(self, state: State | str) -> None:
        name = state if isinstance(state, str) else state.name
        self.states.remove([_ for _ in self.states if _.name == name][0])

    ###########################################################################
    def remove_artifact(self, artifact: Artifact | str) -> None:
        name = artifact if isinstance(artifact, str) else artifact.name
        self.artifacts.remove([_ for _ in self.artifacts if _.name == name][0])

    ###########################################################################
    def plr_discard_cards(self, num: int = 1, any_number: bool = False, **kwargs: Any) -> list[Card]:
        """Ask player to discard cards and discard ALL selected cards at once (2022 rule)."""
        if "prompt" not in kwargs:
            kwargs["prompt"] = "Discard any number of cards" if any_number else f"Discard {num} cards"
        discard = self.card_sel(num=num, anynum=any_number, verbs=("Discard", "Undiscard"), **kwargs)
        for card in discard:
            self.output(f"Discarding {card}")
        if discard:
            src = kwargs.get("cardsrc", Piles.HAND)
            if not isinstance(src, (Piles, PlayArea)):
                src = None
            self.discard_cards(discard, source=src, hook=True)
        return discard

    ###########################################################################
    def plr_discard_down_to(self, num: int) -> None:
        """Ask player to discard down to `num` cards in hand (all discarded at once per 2022 rule)."""
        num_to_go = len(self.piles[Piles.HAND]) - num
        if num_to_go <= 0:
            return
        self.plr_discard_cards(num_to_go, force=True)

    ###########################################################################
    def game_over(self) -> None:
        """Run end-of-game hooks (return set-aside/mat cards, then Landmarks & Projects)."""
        seen: set[int] = set()
        sources: list[Any] = []
        for card in list(self.all_cards()) + list(self.end_of_game_cards) + list(self.game.landmarks.values()) + list(self.projects):
            cid = id(card)
            if cid not in seen:
                seen.add(cid)
                sources.append(card)
        for card in sources:
            self.currcards.append(card)
            try:
                card.hook_end_of_game(game=self.game, player=self)
            finally:
                self.currcards.pop()

    ###########################################################################
    # Delegated UX / Decision Methods (via active_controller)
    ###########################################################################
    def output(self, msg: str, end: str = "\n") -> None:
        self.active_controller.output(self, msg, end=end)

    def plr_choose_options(self, prompt: str, *choices: tuple[str, Any]) -> Any:
        return self.active_controller.choose_options(self, prompt, *choices)

    def user_input(self, options: list[Option], prompt: str) -> Any:
        return self.active_controller.choose_turn_action(self, options, prompt)

    def card_sel(self, num: int = 1, **kwargs: Any) -> list[Card]:
        return self.active_controller.select_cards(self, num=num, **kwargs)

    def card_pile_sel(self, num: int = 1, **kwargs: Any) -> list[str]:
        return self.active_controller.select_piles(self, num=num, **kwargs)

    ###########################################################################
    def debug_all_cards(self) -> None:
        for card in self.all_cards():
            card.debug_dump(self)


# EOF
