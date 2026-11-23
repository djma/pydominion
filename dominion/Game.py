#!/usr/bin/env python3
"""Dominion Game Engine Core."""

from __future__ import annotations

import json
import random
import sys
from typing import Any, Optional

from dominion import NoCardException, Piles
from dominion import game_setup
from dominion.Artifact import Artifact
from dominion.Boon import Boon
from dominion.Card import Card, CardType
from dominion.CardPile import CardPile
from dominion.Event import Event
from dominion.GameConfig import GameConfig
from dominion.Hex import Hex
from dominion.Landmark import Landmark
from dominion.PlayArea import PlayArea
from dominion.Player import Player
from dominion.Project import Project
from dominion.Prophecy import Prophecy
from dominion.State import State
from dominion.Trait import Trait
from dominion.TurnQueue import TurnEntry, TurnQueue
from dominion.Way import Way


###############################################################################
class Game:
    """Game class — owns global rules state, piles, turn scheduler, and game end."""

    def __init__(self, **kwargs: Any) -> None:
        self.rng = random.Random(random.getrandbits(64))
        self.players: dict[str, Player] = {}
        self.cards: dict[str, Card] = {}
        self.card_piles: dict[str, CardPile] = {}
        self.kingdom_piles: list[str] = []
        self.states: dict[str, State] = {}
        self.artifacts: dict[str, Artifact] = {}
        self.projects: dict[str, Project] = {}
        self.events: dict[str, Event] = {}
        self.ways: dict[str, Way] = {}
        self.landmarks: dict[str, Landmark] = {}
        self.boons: list[Boon] = []
        self.traits: dict[str, Trait] = {}
        self.discarded_boons: list[Boon] = []
        self.retained_boons: list[Boon] = []
        self.hexes: list[Hex] = []
        self._turns: list[str] = []
        self.turn_queue: TurnQueue = TurnQueue()
        self.ally: Optional[Card] = None
        self.discarded_hexes: list[Hex] = []
        self.trash_pile = PlayArea("trash", game=self)
        self.inactive_prophecy: Optional[Prophecy] = None
        self.prophecy: Optional[Prophecy] = None
        self.game_over = False
        self.quiet = False
        self.use_shelters = False
        self.heirlooms: list[str] = []
        self.current_player: Optional[Player] = None
        self.sun_tokens: int = 0
        self.numplayers = 0
        self.specials: dict[str, Any] = {}  # Special areas for specific card related stuff

        self.config: GameConfig = GameConfig.from_kwargs(**kwargs)
        self.quiet = bool(kwargs.get("quiet", False))
        self.numplayers = int(kwargs.get("numplayers", 2))
        game_setup.parse_args(self, **kwargs)

        self.card_mapping = game_setup.get_available_card_classes(self.config)
        self._original: dict[str, Any] = {}
        self.card_instances: dict[str, Card] = {}

    ###########################################################################
    def start_game(self, player_names: Optional[list[str]] = None) -> None:
        """Start the game - pass everything through to game_setup"""
        game_setup.start_game(self, player_names)
        self._save_original()

    ###########################################################################
    def _save_original(self) -> None:
        """Save original card state for debugging purposes"""
        self._original["count"] = self._count_all_cards()
        self._original["total_cards"] = self.count_cards()

    ###########################################################################
    def remove_sun_token(self) -> int:
        """Remove a Sun token"""
        self.sun_tokens -= 1
        if self.sun_tokens <= 0:
            self.reveal_prophecy()
            self.sun_tokens = 0
        return self.sun_tokens

    ###########################################################################
    def reveal_prophecy(self) -> None:
        """Prophecy has been enacted - make it active"""
        if self.prophecy is not None:
            return
        if self.inactive_prophecy is None:
            return
        self.output(f"Prophecy {self.inactive_prophecy.name} is now active")
        self.prophecy = self.inactive_prophecy
        assert self.prophecy is not None
        self.prophecy.hook_reveal_prophecy(self)

    ###########################################################################
    def player_list(self) -> list[Player]:
        """List of players in seating order"""
        return list(self.players.values())

    ###########################################################################
    def players_in_turn_order(self, start_player: Optional[Player] = None) -> list[Player]:
        """Return players in clockwise turn order starting with `start_player` (or current_player)."""
        players = self.player_list()
        if not players:
            return []
        anchor = start_player or self.current_player or players[0]
        if self.turn_queue.player_order and anchor.uuid in self.turn_queue.player_order:
            uuids = self.turn_queue.turn_order_from(anchor.uuid)
            return [self.players[u] for u in uuids if u in self.players]
        idx = players.index(anchor)
        return players[idx:] + players[:idx]

    ###########################################################################
    def count_cards(self) -> int:
        """Count the number of cards in the game"""
        count = {"trash": self.trash_pile.size()}
        for name, pile in list(self.card_piles.items()):
            count[f"pile_{name}"] = len(pile)
        for plr in self.player_list():
            count[f"player_{plr.name}"] = plr.count_cards()
        return sum(count.values())

    ###########################################################################
    def output(self, msg: str) -> None:
        """Send output to all players"""
        if not self.quiet:
            sys.stdout.write(f"ALL: {msg}\n")

    ###########################################################################
    def assign_trait(self, trait: str, card_pile: str) -> None:
        """Assign the trait to the card pile"""
        self.card_piles[card_pile].trait = trait
        self.traits[trait].card_pile = card_pile

    ###########################################################################
    def get_prizes(self) -> list[str]:
        """Return a list of prizes"""
        return list(self.card_mapping["PrizeCard"].keys())

    ###########################################################################
    def get_card_from_pile(self, pile: str, name: str = "", top_only: bool = False) -> Card:
        """Get and return a card from pile (with name if specified)."""
        assert isinstance(pile, str), f"{pile=} {type(pile)=}"
        assert pile in self.card_piles, f"{pile=} not in {self.card_piles=}"
        card_pile = self.card_piles[pile]
        num_cards = len(card_pile)
        card = card_pile.remove(name, top_only=top_only)
        if card is None:
            raise NoCardException
        card.game = self
        new_num_cards = len(card_pile)
        if num_cards > 0 and new_num_cards == 0 and card_pile.in_supply:
            self.emptied_pile(card)
        card.location = Piles.LIMBO
        return card

    ###########################################################################
    def emptied_pile(self, card: Card) -> None:
        """A Supply card pile has been emptied (triggers Search, etc.)."""
        for player in self.players_in_turn_order():
            for hook_card in player.in_play_and_landscape_cards():
                hook_card.hook_emptied_pile(self, player, card)

    ###########################################################################
    def get_supply_piles(self) -> list[tuple[str, CardPile]]:
        """Return ONLY Supply piles (excludes Loot, Shelters, Travellers, Heirlooms, Spoils, Prizes, etc.)."""
        return [(key, pile) for key, pile in self.card_piles.items() if pile.in_supply]

    ###########################################################################
    def get_card_piles(self) -> list[tuple[str, CardPile]]:
        """Return the Supply card piles in this game."""
        return self.get_supply_piles()

    ###########################################################################
    def get_all_piles(self) -> list[tuple[str, CardPile]]:
        """Return all card piles in this game (both Supply and non-Supply)."""
        return list(self.card_piles.items())

    ###########################################################################
    def get_available_cards(self, prefix: str = "Card") -> list[str]:
        """Return a list of all available cards"""
        return list(self.card_mapping[prefix].keys())

    ###########################################################################
    def _pile_types(self, name: str, pile: CardPile) -> list[CardType]:
        """Return the pile's types per Split Pile Randomizer rules (Ch. IV p. 36)."""
        if pile.randomizer_types:
            return list(pile.randomizer_types)
        if name in self.card_instances:
            return self.card_instances[name].get_raw_card_type()
        if not pile.is_empty():
            return pile.get_top_card().get_raw_card_type()
        return []

    ###########################################################################
    def _pile_cost(self, name: str, pile: CardPile) -> int:
        """Return the pile's printed coin cost per Split Pile Randomizer rules."""
        if pile.randomizer_cost is not None:
            return pile.randomizer_cost.coins
        if name in self.card_instances and self.card_instances[name].cost >= 0:
            return self.card_instances[name].cost
        if not pile.is_empty():
            return pile.get_top_card().cost
        return 0

    ###########################################################################
    def get_action_piles(self, cost: int = 999) -> list[str]:
        """Return all Supply card stacks that are Action piles costing <= `cost` (per Randomizer rules)."""
        action_piles = []
        for name, pile in self.get_supply_piles():
            types = self._pile_types(name, pile)
            if CardType.ACTION not in types:
                continue
            pile_cost = self._pile_cost(name, pile)
            if pile_cost > cost:
                continue
            action_piles.append(name)
        return action_piles

    ###########################################################################
    def get_treasure_piles(self) -> list[str]:
        """Return all Supply card stacks that are Treasure piles."""
        treasure_piles = []
        for name, pile in self.get_supply_piles():
            top_card = pile.get_top_card() if not pile.is_empty() else None
            instance = self.card_instances.get(name)
            if (
                CardType.TREASURE in self._pile_types(name, pile)
                or (top_card is not None and top_card.isTreasure())
                or (instance is not None and instance.isTreasure())
            ):
                treasure_piles.append(name)
        return treasure_piles

    ###########################################################################
    def get_victory_piles(self) -> list[str]:
        """Return all Supply card stacks that are Victory piles."""
        victory_piles = []
        for name, pile in self.get_supply_piles():
            if CardType.VICTORY in self._pile_types(name, pile):
                victory_piles.append(name)
        return victory_piles

    ###########################################################################
    def empty_supply_piles(self) -> list[str]:
        """Return names of all currently empty Supply piles."""
        return [name for name, pile in self.get_supply_piles() if pile.is_empty()]

    ###########################################################################
    def isGameOver(self) -> bool:
        """Check Dominion game-end conditions (Complete Rules Ch. II p. 15):
        - Province pile is empty
        - OR Colony pile is in the game and empty
        - OR 3 Supply piles are empty (4 Supply piles for 5–6 players)
        Non-Supply piles NEVER count toward empty piles.
        """
        empties = self.empty_supply_piles()
        threshold = 4 if self.numplayers >= 5 else 3
        if len(empties) >= threshold:
            self.output(f"Game Over: {', '.join(empties)} piles are empty")
            return True
        if "Colony" in self.card_piles and self.card_piles["Colony"].is_empty():
            self.output("Game Over: Colony pile is empty")
            return True
        if "Province" in self.card_piles and self.card_piles["Province"].is_empty():
            self.output("Game Over: Province pile is empty")
            return True
        return False

    ###########################################################################
    def receive_hex(self) -> Optional[Hex]:
        """Draw the top Hex card from the Hex deck (shuffling discarded hexes if empty)."""
        if not self.hexes:
            self.cleanup_hexes()
        if not self.hexes:
            return None
        return self.hexes.pop()

    ###########################################################################
    def cleanup_hexes(self) -> None:
        """Reshuffle discarded hexes into a new Hex deck when empty."""
        for hx in self.discarded_hexes[:]:
            self.hexes.append(hx)
        self.rng.shuffle(self.hexes)
        self.discarded_hexes = []

    ###########################################################################
    def discard_hex(self, hx: Hex) -> None:
        """Discard a Hex card to the Hex discard pile."""
        self.discarded_hexes.append(hx)

    ###########################################################################
    def distribute_hex(self, attacker: Player, victims: list[Player]) -> Optional[Hex]:
        """Complete Rules Ch. IV (Boons and Hexes p. 35):
        Take ONE top Hex card (even if all opponents blocked), each unblocked victim
        receives that same Hex in turn order, then discard the Hex once.
        """
        hx = self.receive_hex()
        if hx is None:
            return None
        ordered_victims = [p for p in self.players_in_turn_order(attacker) if p in victims]
        for victim in ordered_victims:
            victim.receive_hex(hx=hx, discard=False)
        self.discard_hex(hx)
        return hx

    ###########################################################################
    def receive_boon(self) -> Optional[Boon]:
        """Draw the top Boon card (shuffling discarded boons ONLY when Boon deck is empty)."""
        if not self.boons:
            self.boons = self.discarded_boons[:]
            self.discarded_boons = []
            self.rng.shuffle(self.boons)
        try:
            boon = self.boons.pop()
        except IndexError:
            boon = None
        return boon

    ###########################################################################
    def cleanup_boons(self) -> None:
        """Discard any Boons kept until Clean-up to the Boon discard pile.

        Per Complete Rules Ch. IV p. 35, the Boon discard pile is NOT reshuffled
        into the Boon deck until a Boon is needed and the deck is empty.
        """
        for boon in self.retained_boons[:]:
            self.discarded_boons.append(boon)
        self.retained_boons = []

    ###########################################################################
    def discard_boon(self, boon: Boon) -> None:
        """Discard a Boon (or retain it until Clean-up if `boon.retain_boon`)."""
        if boon.retain_boon:
            self.retained_boons.append(boon)
        else:
            self.discarded_boons.append(boon)

    ###########################################################################
    def print_state(self, card_dump: bool = False) -> None:  # pragma: no cover
        """This is used for debugging"""
        print("\n" + "#" * 80)
        print(f"Trash: {', '.join([_.name for _ in self.trash_pile])}")
        print(f"Boons: {', '.join([_.name for _ in self.boons])}")
        print(f"Hexes: {', '.join([_.name for _ in self.hexes])}")
        print(f"Prophecy: {self.inactive_prophecy} Tokens: {self.sun_tokens}")
        if self.ally:
            print(f"Ally: {self.ally.name}")
        print(f"Projects: {', '.join([_.name for _ in self.projects.values()])}")
        print(f"Traits: {self.traits}")
        for name, card_pile in self.card_piles.items():
            tokens = ""
            for plr in self.player_list():
                tkns = plr.which_token(name)
                if tkns:
                    tokens += f"{plr.name}[{','.join(tkns)}]"
            trait = f"Trait: {card_pile.trait}" if card_pile.trait else ""
            supply_tag = "" if card_pile.in_supply else " [non-supply]"
            print(f"CardPile {name}: {len(card_pile)} cards {tokens} {trait}{supply_tag}")
        print(f"Instances: {', '.join(self.card_instances)}")

        for plr in self.player_list():
            plr.print_state()
        print()
        if card_dump:
            for v in self.cards.values():
                print(f"    {v} ({v.uuid} {v._player}@{v._location})")

    ###########################################################################
    def player_to_left(self, plr: Player) -> Player:
        """Return the player to the 'left' of `plr` (next in clockwise turn order)."""
        if self.turn_queue.player_order and plr.uuid in self.turn_queue.player_order:
            return self.players[self.turn_queue.left_uuid(plr.uuid)]
        players = self.player_list()
        place = (players.index(plr) + 1) % len(players)
        return players[place]

    ###########################################################################
    def playerToRight(self, plr: Player) -> Player:
        """Return the player to the 'right' of `plr` (previous in clockwise turn order)."""
        if self.turn_queue.player_order and plr.uuid in self.turn_queue.player_order:
            return self.players[self.turn_queue.right_uuid(plr.uuid)]
        players = self.player_list()
        place = (players.index(plr) - 1) % len(players)
        return players[place]

    player_to_right = playerToRight

    ###########################################################################
    def determine_winners(self) -> list[Player]:
        """Return the winning player(s) per Complete Rules Ch. II (Game end p. 15):
        Highest VP wins; ties broken by fewest regular turns (excluding extra/Fleet turns);
        still tied players share the victory.
        """
        players = self.player_list()
        if not players:
            return []
        scored = [(plr.get_score(), -plr.turn_number, plr) for plr in players]
        best_key = max((vp, neg_turns) for vp, neg_turns, _ in scored)
        return [plr for vp, neg_turns, plr in scored if (vp, neg_turns) == best_key]

    ###########################################################################
    def whoWon(self) -> dict[str, int]:
        """Compute, display, and return final scores (and announce winner(s) with tie-breaker)."""
        scores: dict[str, int] = {}
        self.output("")
        self.output("Scores:")
        for plr in self.player_list():
            scores[plr.name] = plr.get_score(verbose=True)
        self.output(json.dumps(scores, indent=2))
        winners = self.determine_winners()
        if winners:
            winner_names = ", ".join(p.name for p in winners)
            self.output(f"Winner(s): {winner_names}")
        self.output("")
        for plr in self.player_list():
            self.output(f"Cards of {plr.name}:")
            for k, v in plr.get_cards().items():
                self.output(f"{plr.name}: {k}={v}")
        return scores

    ###########################################################################
    def last_turn(self, plr: Player) -> bool:
        """True if `plr` took the most recently completed turn."""
        if self.turn_queue.history:
            return self.turn_queue.history[-1].player_uuid == plr.uuid
        try:
            return self._turns[-1] == plr.uuid
        except IndexError:
            return False

    ###########################################################################
    def schedule_extra_turn(
        self,
        plr: Player,
        source: str = "",
        modifiers: Optional[dict[str, Any]] = None,
        enforce_consecutive_cap: bool = True,
    ) -> bool:
        """Schedule an extra turn for `plr` via TurnQueue (enforcing 2023 max-2-in-a-row rule)."""
        return self.turn_queue.schedule_extra_turn(
            plr.uuid,
            source=source,
            modifiers=modifiers,
            enforce_consecutive_cap=enforce_consecutive_cap,
        )

    ###########################################################################
    def _count_all_cards(self) -> dict[str, dict[str, int]]:  # pragma: no cover
        """Return where all the cards are"""
        tmp: dict[str, dict[str, int]] = {}
        for pile_name, pile in self.card_piles.items():
            tmp[pile_name] = {}
            total = len(pile)
            tmp[pile_name]["pile"] = total
            for plr in self.player_list():
                for stack_name, stack in plr.piles.items():
                    if count := sum(1 for card in stack if card.pile == pile_name):
                        tmp[pile_name][f"{plr.name}:{stack_name}"] = count
                        total += count
                for mat_name, mat in plr.set_aside.items():
                    if count := sum(1 for card in mat if card.pile == pile_name):
                        tmp[pile_name][f"{plr.name}:set_aside:{mat_name}"] = count
                        total += count
            if count := sum(1 for card in self.trash_pile if card.pile == pile_name or card.name == pile_name):
                tmp[pile_name]["trash"] = count
                total += count
            tmp[pile_name]["total"] = total
        return tmp

    ###########################################################################
    def _card_loc_debug(self) -> None:  # pragma: no coverage
        """Dump info to help debug card location errors"""
        now = self._count_all_cards()
        print(f"\n{'- -' * 10} Card Loc Debug {'- -' * 10}", file=sys.stderr)
        print(
            f"current={self.count_cards()} original={self._original['total_cards']}\n",
            file=sys.stderr,
        )
        for name in self.card_piles:
            if name in self._original["count"] and self._original["count"][name]["total"] == now[name]["total"]:
                continue
            print(f"{name} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", file=sys.stderr)
            print(f" {name} Original:")
            try:
                print(json.dumps(self._original["count"][name], indent=2), file=sys.stderr)
            except KeyError:
                print(f"Unhandled card {name}")
            print(f" {name} Now:")
            print(json.dumps(now[name], indent=2), file=sys.stderr)
        print(f"{'- -' * 20}", file=sys.stderr)
        for plr in self.players.values():
            plr.debug_all_cards()
            if self.inactive_prophecy:
                self.inactive_prophecy.debug_dump(plr)
        print(f"{'- -' * 20}", file=sys.stderr)

    ###########################################################################
    def _validate_cards(self) -> None:  # pragma: no coverage
        seen = set()
        areas = list(self.card_piles.values()) + [self.trash_pile]
        for player in self.players.values():
            areas.extend(player.piles.values())
            areas.extend(player.set_aside.values())
        for area in areas:
            for card in area:
                assert card.uuid not in seen, f"Duplicate card UUID: {card.uuid} ({card.name})"
                seen.add(card.uuid)
        if "total_cards" not in self._original:
            return
        try:
            assert self.count_cards() == self._original["total_cards"]
        except AssertionError:
            self._card_loc_debug()
            raise

    ###########################################################################
    def turn(self) -> None:
        """Execute one turn using TurnQueue (regular, extra, or Fleet)."""
        if self.game_over:
            return
        self._validate_cards()
        entry: TurnEntry = self.turn_queue.pop_next_turn()
        self.current_player = self.players[entry.player_uuid]
        assert self.current_player is not None
        self.current_player.turn(turn_entry=entry)
        self._validate_cards()
        self.turn_queue.finish_current_turn()
        self._turns.append(entry.player_uuid)

        if self.turn_queue.fleet_started or self.isGameOver():
            # Check for Fleet extra round (Ch. VI After turn p. 44)
            if not self.turn_queue.fleet_started:
                fleet_uuids = {
                    plr.uuid for plr in self.player_list() if any(p.name == "Fleet" for p in plr.projects)
                }
                if fleet_uuids:
                    self.turn_queue.start_fleet_round(entry.player_uuid, fleet_uuids)
            if (
                not self.turn_queue.fleet_started
                or not self.turn_queue.has_pending_post_game_turns()
            ):
                self.game_over = True
                for plr in self.players_in_turn_order():
                    plr.game_over()

    ###########################################################################
    def snapshot(self) -> Any:
        """Capture a complete, deterministic snapshot of engine state (for undo/rewind/search)."""
        from dominion.Snapshot import capture_snapshot

        return capture_snapshot(self)

    ###########################################################################
    def restore(self, snap: Any) -> None:
        """Rewind this Game instance in-place to a previously captured snapshot."""
        from dominion.Snapshot import restore_snapshot

        restore_snapshot(self, snap)

    ###########################################################################
    def clone(self) -> "Game":
        """Create an independent deep copy of this Game's engine state (with headless controllers)."""
        from dominion.Snapshot import clone_game

        return clone_game(self)


###############################################################################
class TestGame(Game):
    """Game for testing purposes (deterministic defaults unless overridden)."""

    __test__ = False

    def __init__(self, **kwargs: Any) -> None:
        if "allies" not in kwargs:
            kwargs["allies"] = []
            kwargs["ally_path"] = "tests/allies"
        if "shelters" not in kwargs:
            kwargs["shelters"] = False
        if "prosperity" not in kwargs:
            kwargs["prosperity"] = False
        if "random_start_player" not in kwargs:
            kwargs["random_start_player"] = False
        if "quiet" not in kwargs:
            kwargs["quiet"] = True
        # Shaman and Changeling trigger global prompts that interfere with unrelated unit tests
        badcards = list(kwargs.get("badcards", []))
        for noisy in ("Shaman", "Changeling"):
            if noisy not in kwargs.get("initcards", []) and noisy not in badcards:
                badcards.append(noisy)
        kwargs["badcards"] = badcards
        super().__init__(**kwargs)


# EOF
