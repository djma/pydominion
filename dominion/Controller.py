"""Player decision controller layer — decouples game engine from UX/IO."""

from __future__ import annotations

import inspect
import random
import shutil
import sys
import textwrap
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Optional, TYPE_CHECKING, cast

from rich.console import Console

from dominion import Piles, Prompt
from dominion.Card import Card, CardType
from dominion.CardPile import CardPile
from dominion.Option import Option
from dominion.PlayArea import PlayArea

if TYPE_CHECKING:
    from dominion.Player import Player

COLOURS = [
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
]


###############################################################################
class PlayerController(ABC):
    """Abstract controller that handles player IO and decision making.

    Engine state lives strictly on Game/Player; all terminal rendering, prompts,
    console handles, and AI/user decision policies live on a PlayerController.
    """

    @abstractmethod
    def output(self, player: "Player", msg: str, end: str = "\n") -> None:
        """Deliver a message to this player."""

    def display_overview(self, player: "Player") -> None:
        """Display the turn/phase overview to this player (no-op for headless/bots)."""

    @abstractmethod
    def choose_turn_action(self, player: "Player", options: list[Option], prompt: str) -> Option:
        """Choose one action option from the current phase prompt."""

    @abstractmethod
    def choose_options(self, player: "Player", prompt: str, *choices: tuple[str, Any]) -> Any:
        """Choose one of the given (label, answer) tuples."""

    @abstractmethod
    def select_cards(self, player: "Player", num: int = 1, **kwargs: Any) -> list[Card]:
        """Select card(s) from hand/discard/etc."""

    @abstractmethod
    def select_piles(self, player: "Player", num: int = 1, **kwargs: Any) -> list[str]:
        """Select Supply/card pile name(s)."""

    def check_unexile(self, player: "Player", card_name: str) -> bool:
        """Decide whether to discard all other copies of card_name from Exile mat."""
        num = sum(1 for _ in player.piles[Piles.EXILE] if _.name == card_name)
        if num == 0:
            return False
        choices = [
            (f"Un-exile {num} x {card_name}", True),
            ("Do nothing", False),
        ]
        return bool(self.choose_options(player, f"Un-exile {card_name}", *choices))

    @staticmethod
    def select_source(player: "Player", **kwargs: Any) -> PlayArea | list[Card]:
        """Resolve cardsrc argument into a sequence of cards."""
        if "cardsrc" in kwargs:
            if isinstance(kwargs["cardsrc"], Piles):
                return player.piles[kwargs["cardsrc"]]
            return kwargs["cardsrc"]
        return player.piles[Piles.HAND]

    @staticmethod
    def can_finish(any_num: bool, force: bool, num: int, num_selected: int) -> bool:
        """Return whether the player may stop selecting cards/piles."""
        if any_num:
            return True
        if force and num == num_selected:
            return True
        if not force and num >= num_selected:
            return True
        return False


###############################################################################
class TextController(PlayerController):
    """Interactive terminal / test_input controller using rich Console."""

    def __init__(self, player_number: int = 0) -> None:
        self.colour = COLOURS[player_number % len(COLOURS)]
        self.console = Console()

    def output(self, player: "Player", msg: str, end: str = "\n") -> None:
        player.messages.append(msg)
        if player.quiet:
            return
        prompt = f"[{self.colour}]{player.name}[/]: "
        current_card_stack = "".join(f"{card.name}> " for card in player.currcards)
        self.console.print(f"{prompt}{current_card_stack}{msg}", end=end)

    def display_overview(self, player: "Player") -> None:
        Prompt.display_overview(player)

    def selector_line(self, player: "Player", o: Option) -> str:
        """Format an Option as a display line."""
        output: list[str] = [f"{o['selector']})"]
        for key in ("print", "verb", "name"):
            if o[key]:
                output.append(str(o[key]))
        if o["details"]:
            output.append(f"({o['details']})")
        if o["name"] and not o["details"] and o["desc"]:
            output.append("-")
        if o["notes"]:
            output.append(str(o["notes"]))

        indent = len(player.name) + 5
        if player.currcards:
            indent += len(player.currcards[0].name) + 2
        desc = ""
        for line in str(o["desc"]).splitlines():
            desc += line.strip() + " "
        output.append(desc)
        text = " ".join(output)
        (cols, _) = shutil.get_terminal_size((80, 24))
        return textwrap.fill(text, subsequent_indent=" " * indent, width=max(20, cols - indent))

    def get_raw_input(self, player: "Player") -> str:
        """Get raw input string from test_input queue or stdin."""
        if player.test_input:
            inp = player.test_input.pop(0)
            self.output(player, f"Using '{inp}' test input")
            return inp
        try:  # pragma: no coverage
            return input()
        except (IOError, KeyboardInterrupt):  # pragma: no coverage
            player.game.print_state()
            raise

    def choose_turn_action(self, player: "Player", options: list[Option], prompt: str) -> Option:
        """Display options and pick one based on selector or unique substring match."""
        for opt in options:
            assert isinstance(opt, Option), f"user_input {opt=} {type(opt)=}"
        for o in options:
            line = self.selector_line(player, o)
            o["line"] = line
            self.output(player, line)
        self.output(player, prompt, end=" ")
        while True:
            if inp := self.get_raw_input(player):
                matching: list[Option] = []
                for o in options:
                    if o["selector"] == inp:
                        return o
                    if inp.lower() in str(o["line"]).lower() and o["selector"] != "-":
                        matching.append(o)
                if len(matching) == 1:
                    return matching[0]
            self.output(player, f"Invalid Option ({inp})")

    def choose_options(self, player: "Player", prompt: str, *choices: tuple[str, Any]) -> Any:
        options: list[Option] = []
        for index, (prnt, ans) in enumerate(choices):
            options.append(Option(selector=f"{index}", verb=prnt, answer=ans))
        o = self.choose_turn_action(player, options, prompt)
        return o["answer"]

    def card_pile_sel_options(
        self,
        player: "Player",
        selected: list[Any],
        piles: list[tuple[str, CardPile]],
        flags: dict[str, Any],
    ) -> list[Option]:
        """Build Option list for pile selection."""
        options: list[Option] = []
        if self.can_finish(flags["any_num"], flags["force"], flags["num"], len(selected)):
            options.append(Option(selector="0", verb="Finish Selecting", card=None))
        index = 1
        for name, _card_pile in piles:
            card = player.game.card_instances[name]
            if card.name in flags["exclude"]:
                continue
            verb = flags["verbs"][0] if name not in selected else flags["verbs"][1]
            o = Option(selector=f"{index}", verb=verb, card=name, name=name)
            index += 1
            if flags["showdesc"]:
                o["desc"] = card.description(player) if card else "Empty card pile"
            if flags["print_cost"]:
                o["details"] = str(player.card_cost(card))
            if flags["print_types"]:
                o["details"] = card.get_cardtype_repr()
            options.append(o)
        return options

    def select_piles(self, player: "Player", num: int = 1, **kwargs: Any) -> list[str]:
        flags: dict[str, Any] = {
            "force": kwargs.get("force", False),
            "showdesc": kwargs.get("showdesc", True),
            "verbs": kwargs.get("verbs", ("Select", "Unselect")),
            "any_num": False,
            "num": num,
            "exclude": kwargs.get("exclude", []),
            "print_cost": kwargs.get("printcost", False),
            "print_types": kwargs.get("printtypes", False),
        }
        if "prompt" in kwargs:
            self.output(player, kwargs["prompt"])
        if kwargs.get("anynum", False):
            flags["any_num"] = True
            flags["num"] = 0

        if kwargs.get("cardsrc"):
            piles = [(key, value) for key, value in player.game.get_card_piles() if key in kwargs["cardsrc"]]
        else:
            piles = player.game.get_card_piles()

        selected: list[str] = []
        while True:
            options = self.card_pile_sel_options(player, selected, piles, flags)
            ui = self.choose_turn_action(player, options, "Select which card pile?")
            if not ui["card"]:
                break
            if ui["card"] in selected:
                selected.remove(ui["card"])
            else:
                selected.append(ui["card"])
            if num == 1 and len(selected) == 1:
                break
        return selected

    def _card_sel_options(
        self,
        player: "Player",
        index: int,
        show_desc: bool,
        selected: list[Card],
        types: dict[CardType, bool],
        select_from: PlayArea | list[Card],
        kwargs: dict[str, Any],
        verbs: tuple[str, str],
    ) -> list[Option]:
        options: list[Option] = []
        for card in sorted(select_from):
            if "exclude" in kwargs and card.name in kwargs["exclude"]:
                continue
            if not player.select_by_type(card, types):
                continue
            verb = verbs[0] if card not in selected else verbs[1]
            o = Option(selector=f"{index}", verb=verb, card=card, name=card.name)
            index += 1
            if show_desc:
                o["desc"] = card.description(player)
            if kwargs.get("printcost"):
                o["details"] = str(player.card_cost(card))
            if kwargs.get("printtypes"):
                o["details"] = card.get_cardtype_repr()
            options.append(o)
        return options

    def select_cards(self, player: "Player", num: int = 1, **kwargs: Any) -> list[Card]:
        select_from = self.select_source(player, **kwargs)
        types = player._type_selector(kwargs.get("types", {}))
        if not any(player.select_by_type(card, types)
                   and card.name not in kwargs.get("exclude", []) for card in select_from):
            return []
        force = kwargs.get("force", False)
        show_desc = kwargs.get("showdesc", True)
        verbs = kwargs.get("verbs", ("Select", "Unselect"))

        if "prompt" in kwargs:
            self.output(player, kwargs["prompt"])

        if kwargs.get("anynum"):
            any_num = True
            num = 0
        else:
            any_num = False

        selected: list[Card] = []
        types = player._type_selector(kwargs.get("types", {}))
        while True:
            options: list[Option] = []
            if self.can_finish(any_num, force, num, num_selected=len(selected)):
                options.append(Option(selector="0", verb="Finish Selecting", card=None))
            options.extend(self._card_sel_options(player, 1, show_desc, selected, types, select_from, kwargs, verbs))
            ui = self.choose_turn_action(player, options, "Select which card?")
            if not ui["card"]:
                break
            if ui["card"] in selected:
                selected.remove(ui["card"])
            else:
                selected.append(ui["card"])
            if num == 1 and len(selected) == 1:
                break
        return selected


###############################################################################
class BigMoneyBotController(PlayerController):
    """Big-money bot controller decoupled from Player state."""

    def __init__(self) -> None:
        self.colour = "red on black"
        self.console = Console()

    def output(self, player: "Player", msg: str, end: str = "\n") -> None:
        player.messages.append(msg)
        if player.quiet:
            return
        prompt = f"[{self.colour}]{player.name}[/]: "
        current_card_stack = "".join(f"{card.name}> " for card in player.currcards)
        self.console.print(f"{prompt}{current_card_stack}{msg}", end=end)

    @classmethod
    def get_options(cls, options: list[Any]) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        for opt in options:
            if opt["action"] == "buy":
                name = opt["card"].name
                if name in ("Colony", "Province", "Platinum", "Gold", "Duchy", "Silver"):
                    opts[name.lower()] = opt
            if opt["action"] == "quit":
                opts["quit"] = opt
            if opt["action"] == "spendall":
                opts["spendall"] = opt
            if opt["action"] == "payback":
                opts["payback"] = opt
        return opts

    def choose_turn_action(self, player: "Player", options: list[Option], prompt: str) -> Option:
        opts = self.get_options(options)
        if "spendall" in opts:
            return opts["spendall"]
        if "payback" in opts:
            return opts["payback"]
        if player.buys.get() == 0:
            return opts["quit"]
        coin = player.coins.get()
        self.output(player, f"Have {coin} coins")
        if coin >= 11 and "colony" in opts:
            return opts["colony"]
        if coin >= 9 and "platinum" in opts:
            return opts["platinum"]
        if coin >= 8 and "province" in opts:
            return opts["province"]
        if coin >= 6 and "gold" in opts:
            return opts["gold"]
        if coin >= 5 and "duchy" in opts:
            return opts["duchy"]
        if coin >= 3 and "silver" in opts:
            return opts["silver"]
        return opts["quit"]

    @classmethod
    def get_calling_card(cls, player: Optional["Player"] = None) -> Optional[ModuleType]:
        """Resolve the card module requesting a bot decision.

        Prefers player.currcards[-1]'s module over stack inspection.
        """
        if player is not None and player.currcards:
            mod = inspect.getmodule(player.currcards[-1].__class__)
            if mod is not None and hasattr(mod, "botresponse"):
                return mod
        stack = inspect.stack()
        for rec in stack:
            mod = inspect.getmodule(rec[0])
            if mod is None:
                continue
            mod_name = mod.__name__.replace("dominion.", "")
            if mod_name not in ("BotPlayer", "Controller", "Player", "__main__"):
                return mod
        return None

    def check_unexile(self, player: "Player", card_name: str) -> bool:
        card = player.game.card_instances[card_name]
        return card.isTreasure()

    def select_cards(self, player: "Player", num: int = 1, **kwargs: Any) -> list[Card]:
        mod = self.get_calling_card(player)
        if mod is not None and hasattr(mod, "botresponse"):
            return cast(list[Card], mod.botresponse(player, "cards", kwargs=kwargs))
        assert False, f"BigMoneyBot can't select cards from {mod.__name__ if mod else None} {kwargs=}"

    def select_piles(self, player: "Player", num: int = 1, **kwargs: Any) -> list[str]:
        piles = [key for key, _ in player.game.get_card_piles()]
        if kwargs.get("cardsrc"):
            piles = [key for key in piles if key in kwargs["cardsrc"]]
        return piles[:num]

    def choose_options(self, player: "Player", prompt: str, *choices: tuple[str, Any]) -> Any:
        mod = self.get_calling_card(player)
        if mod is not None and hasattr(mod, "botresponse"):
            return mod.botresponse(player, "choices", args=choices)
        assert False, f"BigMoneyBot can't choose options from {mod.__name__ if mod else None} {choices=}"


###############################################################################
class RandoBotController(PlayerController):
    """Random-choice bot controller decoupled from Player state."""

    def output(self, player: "Player", msg: str, end: str = "\n") -> None:
        if not player.quiet:
            sys.stdout.write(f"{player.name}: ")
            sys.stdout.write(f"{msg}{end}")
        player.messages.append(msg)

    @staticmethod
    def selector_line(o: Any) -> str:
        output: list[str] = []
        if isinstance(o, dict):
            o_copy = dict(o)
            verb = o_copy.pop("print")
            o = Option(verb=verb, **o_copy)
        elif not isinstance(o, Option):
            raise TypeError(f"Unsupported option type {type(o)}")
        output.append(str(o["selector"]))
        if o["verb"]:
            output.append(str(o["verb"]))
        if o["name"]:
            output.append(str(o["name"]))
        if o["details"]:
            output.append(f"({o['details']})")
        if o["name"] and not o["details"] and o["desc"]:
            output.append("-")
        if o["notes"]:
            output.append(str(o["notes"]))
        output.append(str(o["desc"]))
        return " ".join(output)

    def choose_turn_action(self, player: "Player", options: list[Option], prompt: str) -> Option:
        for opt in options:
            if "action" not in opt:
                break
            if opt["action"] == "spendall" and random.randint(1, 10) < 7:
                return opt
            if opt["action"] == "payback":
                return opt

        avail = [_ for _ in options if _["selector"] != "-" and _.get("action") != "quit"]
        if avail:
            return random.choice(avail)

        for opt in options:
            if opt["action"] == "quit":
                return opt

        raise NotImplementedError(f"user_input - fail. {options=}")

    def select_cards(self, player: "Player", num: int = 1, **kwargs: Any) -> list[Card]:
        cards = list(self.select_source(player, **kwargs))
        if not cards:
            return []
        types = player._type_selector(kwargs.get("types", {}))
        filtered = [
            c
            for c in cards
            if c.name not in kwargs.get("exclude", []) and player.select_by_type(c, types)
        ]
        if not filtered:
            return []
        count = min(num, len(filtered)) if not kwargs.get("anynum") else random.randint(0, len(filtered))
        return random.sample(filtered, count) if count else []

    def select_piles(self, player: "Player", num: int = 1, **kwargs: Any) -> list[str]:
        if kwargs.get("cardsrc"):
            piles = [(key, value) for key, value in player.game.get_card_piles() if key in kwargs["cardsrc"]]
        else:
            piles = player.game.get_card_piles()
        if not piles:
            return []
        pile = random.choice(piles)
        return [pile[0]]

    def choose_options(self, player: "Player", prompt: str, *choices: tuple[str, Any]) -> Any:
        choice = random.choice(choices)
        return choice[1]


# EOF
