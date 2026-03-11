"""Translated from the best-performing opus-4.6 HeuristicBot (86% win rate vs BigMoney).

For Chapel Village Smithy Market Militia Moat Festival Laboratory Workshop Witch

Strategy: Chapel-engine with Witch attack. Aggressively trash Coppers/Estates,
build Village+draw engine, use Witch to curse opponent, green late.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Optional

from rich.console import Console

from dominion import Action, Piles
from dominion.Card import Card
from dominion.Option import Option
from dominion.Player import Player


class OpusHeuristicBotV1(Player):
    """Chapel-engine bot with Witch attack, translated from opus-4.6 HeuristicBot."""

    def __init__(self, game: Any, name: str = "", quiet: bool = False, **kwargs: Any):
        self.colour = "bright_cyan on black"
        self.console = Console()
        Player.__init__(self, game, name, quiet, **kwargs)

    def output(self, msg: str, end: str = "\n") -> None:
        self.messages.append(msg)
        if self.quiet:
            return
        prompt = f"[{self.colour}]{self.name}[/]: "
        current_card_stack = ""
        try:
            for card in self.currcards:
                current_card_stack += f"{card.name}> "
        except IndexError:
            pass
        self.console.print(f"{prompt}{current_card_stack}{msg}", end=end)

    ###########################################################################
    # Helpers
    ###########################################################################

    def _deck_counts(self) -> dict[str, int]:
        """Card name -> count across entire deck (hand+deck+discard+played+etc)."""
        return self.get_cards()

    def _supply_remaining(self, card_name: str) -> int:
        pile = self.game.card_piles.get(card_name)
        if pile is None:
            return 0
        return len(pile)

    def _provinces_remaining(self) -> int:
        return self._supply_remaining("Province")

    ###########################################################################
    # Buy logic (called from user_input during BUY phase)
    ###########################################################################

    def _buy_priority(self) -> Optional[str]:
        """Return the name of the card to buy, or None to buy nothing."""
        deck = self._deck_counts()
        coins = self.coins.get()
        provinces_left = self._provinces_remaining()

        # Count key cards in entire deck
        chapels = deck.get("Chapel", 0)
        villages = deck.get("Village", 0)
        festivals = deck.get("Festival", 0)
        smithies = deck.get("Smithy", 0)
        labs = deck.get("Laboratory", 0)
        witches = deck.get("Witch", 0)
        markets = deck.get("Market", 0)
        silvers = deck.get("Silver", 0)
        golds = deck.get("Gold", 0)
        coppers = deck.get("Copper", 0)
        estates = deck.get("Estate", 0)
        curses = deck.get("Curse", 0)

        total_villages = villages + festivals
        total_terminals = smithies + witches + deck.get("Militia", 0)
        total_non_terminal_draw = labs + markets
        junk = coppers + estates + curses

        # ---- GREENING ----
        if coins >= 8 and self._can_buy("Province"):
            return "Province"
        if provinces_left <= 4 and coins >= 5 and self._can_buy("Duchy"):
            return "Duchy"
        if provinces_left <= 2 and coins >= 2 and self._can_buy("Estate"):
            return "Estate"

        # ---- Get Chapel ASAP ----
        if chapels == 0 and self._can_buy("Chapel"):
            return "Chapel"

        # ---- ENGINE BUILDING ----

        # First Witch
        if witches == 0 and coins >= 5 and self._can_buy("Witch"):
            return "Witch"

        # Gold — but prefer engine pieces in many cases
        if coins >= 6 and self._can_buy("Gold"):
            if total_villages >= 2 and total_terminals + total_non_terminal_draw >= 3:
                return "Gold"
            if junk > 4:
                return "Gold"

        if coins >= 5:
            if labs < 2 and self._can_buy("Laboratory"):
                if total_villages > 0 or labs == 0:
                    return "Laboratory"
            if festivals < 2 and self._can_buy("Festival"):
                if total_terminals > total_villages + total_non_terminal_draw:
                    return "Festival"
                if festivals == 0:
                    return "Festival"
            if witches < 2 and self._can_buy("Witch"):
                curse_remaining = self._supply_remaining("Curse")
                if curse_remaining > 2 and total_villages > total_terminals:
                    return "Witch"
            if markets < 2 and self._can_buy("Market"):
                return "Market"
            if labs < 3 and self._can_buy("Laboratory"):
                return "Laboratory"
            if total_terminals > total_villages + total_non_terminal_draw and self._can_buy("Festival"):
                return "Festival"
            if coins >= 6 and self._can_buy("Gold"):
                return "Gold"
            if self._can_buy("Market"):
                return "Market"
            if provinces_left <= 5 and self._can_buy("Duchy"):
                return "Duchy"

        if coins >= 4:
            if smithies == 0 and self._can_buy("Smithy"):
                if total_villages > total_terminals or total_terminals == 0:
                    return "Smithy"
            if smithies < 2 and total_villages > total_terminals + 1 and self._can_buy("Smithy"):
                return "Smithy"

        if coins >= 3:
            if self._can_buy("Village"):
                if total_villages <= total_terminals:
                    return "Village"
                if villages < 2 and total_villages < 2:
                    return "Village"
            if silvers < 2 and self._can_buy("Silver"):
                return "Silver"
            if total_villages < total_terminals and self._can_buy("Village"):
                return "Village"
            if self._can_buy("Silver"):
                if junk <= 3 and silvers >= 2 and golds >= 1:
                    return None
                return "Silver"

        if coins >= 2:
            if deck.get("Moat", 0) == 0 and witches == 0 and self._can_buy("Moat"):
                return "Moat"
            return None

        return None

    def _can_buy(self, card_name: str) -> bool:
        """Check if a card is available in supply and affordable."""
        pile = self.game.card_piles.get(card_name)
        if pile is None or len(pile) == 0:
            return False
        card = self.game.card_instances.get(card_name)
        if card is None:
            return False
        return self.coins.get() >= card.cost

    ###########################################################################
    # Action play logic (called from user_input during ACTION phase)
    ###########################################################################

    def _action_priority(self, playable: list[str]) -> Optional[str]:
        """Return the name of the action card to play, or None to stop."""
        if not playable:
            return None

        hand_names = [c.name for c in self.piles[Piles.HAND]]
        actions = self.actions.get()

        junk_in_hand = [c for c in hand_names if c in ("Copper", "Estate", "Curse")]

        terminals_in_hand = [c for c in playable if c in ("Witch", "Smithy", "Chapel", "Militia", "Moat")]
        non_terminals_in_hand = [c for c in playable if c in ("Village", "Festival", "Market", "Laboratory")]

        if actions == 1:
            # With 1 action, play a non-terminal first if we have terminals to unlock
            if non_terminals_in_hand and terminals_in_hand:
                if "Festival" in playable:
                    return "Festival"
                if "Village" in playable:
                    return "Village"
                if "Laboratory" in playable:
                    return "Laboratory"
                if "Market" in playable:
                    return "Market"

            # No non-terminals — pick the best terminal
            if not non_terminals_in_hand:
                if "Witch" in playable:
                    return "Witch"
                if "Smithy" in playable:
                    return "Smithy"
                if "Chapel" in playable and junk_in_hand:
                    return "Chapel"
                if "Moat" in playable:
                    return "Moat"
                if "Militia" in playable:
                    return "Militia"
                if "Chapel" in playable:
                    return None  # don't waste Chapel with no junk
        else:
            # Multiple actions: non-terminals first, then terminals
            if "Festival" in playable:
                return "Festival"
            if "Village" in playable:
                return "Village"
            if "Laboratory" in playable:
                return "Laboratory"
            if "Market" in playable:
                return "Market"
            if "Witch" in playable:
                return "Witch"
            if "Smithy" in playable:
                return "Smithy"
            if "Chapel" in playable and junk_in_hand:
                return "Chapel"
            if "Moat" in playable:
                return "Moat"
            if "Militia" in playable:
                return "Militia"

        return None

    ###########################################################################
    # user_input — main decision entry point
    ###########################################################################

    def user_input(self, options: list[Option], prompt: str) -> Option:
        del prompt
        legal = [opt for opt in options if self._option_is_legal(opt)]
        legal = [opt for opt in legal if opt["action"] != Action.SPENDALL]

        action_moves: list[Option] = []
        treasure_moves: list[Option] = []
        buy_moves: list[Option] = []
        quit_opt: Optional[Option] = None

        for move in legal:
            if move["action"] in (Action.PLAY, Action.WAY):
                card = move["card"]
                if move["action"] == Action.PLAY and isinstance(card, Card) and card.isTreasure():
                    treasure_moves.append(move)
                else:
                    action_moves.append(move)
            elif move["action"] == Action.SPEND:
                treasure_moves.append(move)
            elif move["action"] == Action.BUY:
                buy_moves.append(move)
            elif move["action"] == Action.QUIT:
                quit_opt = move

        # --- Action phase: use priority logic ---
        if action_moves:
            playable_names = []
            for m in action_moves:
                card = m["card"]
                if isinstance(card, Card):
                    playable_names.append(card.name)

            target = self._action_priority(playable_names)
            if target is not None:
                for m in action_moves:
                    card = m["card"]
                    if isinstance(card, Card) and card.name == target:
                        return m
            # No action worth playing — fall through to treasures or quit
            if treasure_moves:
                return self._play_best_treasure(treasure_moves)
            if quit_opt:
                return quit_opt

        # --- Treasure phase: play all treasures ---
        if treasure_moves:
            return self._play_best_treasure(treasure_moves)

        # --- Buy phase: use priority logic ---
        if buy_moves or quit_opt:
            if self.buys.get() == 0 and quit_opt:
                return quit_opt

            target = self._buy_priority()
            if target is not None:
                for m in buy_moves:
                    card = m["card"]
                    if isinstance(card, Card) and card.name == target:
                        return m
                    # Some buy options use "name" instead of card
                    if m.get("name") == target:
                        return m

            if quit_opt:
                return quit_opt

        if legal:
            return random.choice(legal)
        if not options:
            raise NotImplementedError("No options passed to user_input")
        return options[0]

    ###########################################################################
    # Treasure play helper
    ###########################################################################

    @staticmethod
    def _play_best_treasure(treasure_moves: list[Option]) -> Option:
        """Play the highest-value treasure first."""
        best = treasure_moves[0]
        best_cost = 0
        for m in treasure_moves:
            card = m["card"]
            cost = getattr(card, "cost", 0) if card else 0
            if cost > best_cost:
                best = m
                best_cost = cost
        return best

    ###########################################################################
    # card_sel — card selection for Chapel trash, Militia discard, etc.
    ###########################################################################

    def card_sel(self, num: int = 1, **kwargs: Any) -> list[Card]:
        cards = self._card_sel_source(**kwargs)
        if not cards:
            return []

        any_num = kwargs.get("anynum", False)
        force = kwargs.get("force", False)
        verbs = kwargs.get("verbs", ("Select", "Unselect"))
        primary_verb = str(verbs[0]).lower() if verbs else ""
        max_to_select = len(cards) if any_num else min(num, len(cards))
        if max_to_select <= 0:
            return []

        if primary_verb == "trash":
            return self._select_to_trash(cards, max_to_select, force)
        if primary_verb == "discard":
            return self._select_to_discard(cards, max_to_select, force)
        if primary_verb in ("get", "gain", "buy"):
            return self._select_to_gain(cards, max_to_select, force)

        # Default: random
        return self._default_card_selection(cards, max_to_select, force, any_num)

    def _card_sel_source(self, **kwargs: Any) -> list[Card]:
        if "cardsrc" in kwargs:
            if isinstance(kwargs["cardsrc"], Piles):
                return list(self.piles[kwargs["cardsrc"]])
            return list(kwargs["cardsrc"])
        return list(self.piles[Piles.HAND])

    def _default_card_selection(
        self, cards: list[Card], max_to_select: int, force: bool, any_num: bool
    ) -> list[Card]:
        if any_num and not force:
            return []
        if not force and max_to_select == 1:
            return [random.choice(cards)]
        available = cards[:]
        selected: list[Card] = []
        while available and len(selected) < max_to_select:
            card = random.choice(available)
            selected.append(card)
            available.remove(card)
        return selected

    ###########################################################################
    # Trash selection (Chapel)
    ###########################################################################

    def _select_to_trash(self, cards: list[Card], max_to_select: int, force: bool) -> list[Card]:
        deck = self._deck_counts()
        coppers = deck.get("Copper", 0)
        silvers = deck.get("Silver", 0)
        golds = deck.get("Gold", 0)
        festivals = deck.get("Festival", 0)
        markets = deck.get("Market", 0)

        non_copper_economy = silvers * 2 + golds * 3 + festivals * 2 + markets
        if non_copper_economy >= 5:
            min_coppers = 0
        elif non_copper_economy >= 3:
            min_coppers = 2
        else:
            min_coppers = 3

        # Priority: Curse > Estate > Copper (down to min)
        trash_order = ["Curse", "Estate"]

        selected: list[Card] = []
        available = cards[:]

        for target_name in trash_order:
            for card in available[:]:
                if card.name == target_name and len(selected) < max_to_select:
                    selected.append(card)
                    available.remove(card)

        # Copper trashing
        coppers_trashable = coppers - min_coppers
        for card in available[:]:
            if card.name == "Copper" and len(selected) < max_to_select and coppers_trashable > 0:
                selected.append(card)
                available.remove(card)
                coppers_trashable -= 1

        # If forced and still need more, trash cheapest remaining
        if force:
            remaining = sorted(available, key=lambda c: c.cost)
            while remaining and len(selected) < max_to_select:
                selected.append(remaining.pop(0))

        return selected

    ###########################################################################
    # Discard selection (Militia attack, etc.)
    ###########################################################################

    def _select_to_discard(self, cards: list[Card], max_to_select: int, force: bool) -> list[Card]:
        discard_order = [
            "Curse", "Estate", "Duchy", "Province", "Copper", "Chapel",
            "Moat", "Village", "Workshop", "Silver", "Smithy", "Militia",
            "Market", "Festival", "Laboratory", "Witch", "Gold",
        ]

        selected: list[Card] = []
        available = cards[:]

        for target_name in discard_order:
            for card in available[:]:
                if card.name == target_name and len(selected) < max_to_select:
                    selected.append(card)
                    available.remove(card)

        while available and len(selected) < max_to_select:
            selected.append(available.pop(0))

        if not force and not selected:
            return []
        return selected

    ###########################################################################
    # Gain selection (Workshop, etc.)
    ###########################################################################

    def _select_to_gain(self, cards: list[Card], max_to_select: int, force: bool) -> list[Card]:
        deck = self._deck_counts()
        villages = deck.get("Village", 0)
        festivals = deck.get("Festival", 0)
        total_villages = villages + festivals
        terminals = deck.get("Smithy", 0) + deck.get("Witch", 0) + deck.get("Militia", 0)

        # Build preference list
        preferences: list[str] = []
        if total_villages > terminals:
            preferences = ["Smithy", "Village", "Silver"]
        elif total_villages <= terminals:
            preferences = ["Village", "Smithy", "Silver"]
        else:
            preferences = ["Silver", "Village", "Smithy"]

        selected: list[Card] = []
        available = cards[:]

        for target_name in preferences:
            for card in available[:]:
                if card.name == target_name and len(selected) < max_to_select:
                    selected.append(card)
                    available.remove(card)
                    break  # one per preference

        if force and not selected and available:
            # Gain highest cost non-curse card
            non_curse = [c for c in available if c.name != "Curse"]
            if non_curse:
                non_curse.sort(key=lambda c: c.cost, reverse=True)
                selected.append(non_curse[0])
            elif available:
                selected.append(available[0])

        return selected

    ###########################################################################
    # card_pile_sel — choose a card pile (e.g. Workshop "gain a card costing up to $4")
    ###########################################################################

    def card_pile_sel(self, num: int = 1, **kwargs: Any) -> list[str] | None:
        del num
        if kwargs.get("cardsrc"):
            piles = [(key, value) for key, value in self.game.get_card_piles() if key in kwargs["cardsrc"]]
        else:
            piles = list(self.game.get_card_piles())

        if not piles:
            return None

        force = kwargs.get("force", False)
        deck = self._deck_counts()
        villages = deck.get("Village", 0) + deck.get("Festival", 0)
        terminals = deck.get("Smithy", 0) + deck.get("Witch", 0) + deck.get("Militia", 0)

        available_names = [name for name, _ in piles]

        # Preference order for gaining
        preferences: list[str]
        if villages > terminals:
            preferences = ["Smithy", "Village", "Silver", "Market", "Laboratory"]
        else:
            preferences = ["Village", "Smithy", "Silver", "Market", "Laboratory"]

        for pref in preferences:
            if pref in available_names:
                return [pref]

        if not force:
            return None
        # Gain the most expensive non-curse card
        best_name = None
        best_cost = -1
        for name, _ in piles:
            card = self.game.card_instances.get(name)
            if card and card.name != "Curse" and card.cost > best_cost:
                best_name = name
                best_cost = card.cost
        return [best_name] if best_name else None

    ###########################################################################
    # plr_choose_options — binary/multi-choice decisions from card effects
    ###########################################################################

    def plr_choose_options(self, prompt: str, *choices: tuple[str, Any]) -> Any:
        if not choices:
            return None

        lower_prompt = prompt.lower()

        # If asked about trashing, check if we have junk
        if "trash" in lower_prompt:
            has_junk = any(
                c.name in ("Curse", "Estate", "Copper") for c in self.piles[Piles.HAND]
            )
            for desc, value in choices:
                if isinstance(value, bool) and value == has_junk:
                    return value

        # If asked about discarding
        if "discard" in lower_prompt:
            has_discardable = any(
                c.name in ("Curse", "Estate", "Copper") for c in self.piles[Piles.HAND]
            )
            for desc, value in choices:
                if isinstance(value, bool) and value == has_discardable:
                    return value

        # Default: prefer positive/True choices, then highest numeric value
        best_choice = choices[0]
        best_score = self._score_choice(choices[0])
        for choice in choices[1:]:
            score = self._score_choice(choice)
            if score > best_score:
                best_choice = choice
                best_score = score

        return best_choice[1]

    @staticmethod
    def _score_choice(choice: tuple[str, Any]) -> float:
        desc, value = choice
        if value is False or value is None:
            return 0.0
        if isinstance(value, bool) and value:
            return 1.0
        if isinstance(value, (int, float)):
            return float(value)
        text = desc.lower()
        if text.startswith(("don't", "do not", "no ", "skip")):
            return 0.0
        return 0.5

    ###########################################################################
    # pick_to_discard — attacks that force discarding from hand
    ###########################################################################

    def pick_to_discard(self, num_to_discard: int, keepvic: bool = False) -> list[Card]:
        if num_to_discard <= 0:
            return []

        discard_order = [
            "Curse", "Estate", "Duchy", "Province", "Copper", "Chapel",
            "Moat", "Village", "Workshop", "Silver", "Smithy", "Militia",
            "Market", "Festival", "Laboratory", "Witch", "Gold",
        ]

        hand = list(self.piles[Piles.HAND])
        selected: list[Card] = []

        for target_name in discard_order:
            if target_name in ("Estate", "Duchy", "Province") and keepvic:
                continue
            for card in hand[:]:
                if card.name == target_name and len(selected) < num_to_discard:
                    selected.append(card)
                    hand.remove(card)

        while hand and len(selected) < num_to_discard:
            selected.append(hand.pop(0))

        if len(selected) < num_to_discard:
            hand_contents = ", ".join(c.name for c in self.piles[Piles.HAND])
            sys.stderr.write(f"Couldn't find cards to discard {num_to_discard} from {hand_contents}\n")
            return []

        return selected

    ###########################################################################
    @classmethod
    def _option_is_legal(cls, option: Option) -> bool:
        if option["selector"] == "-":
            return False
        if option["action"] in ("", None, Action.NONE):
            return False
        return True


# EOF
