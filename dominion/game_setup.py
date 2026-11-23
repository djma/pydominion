"""Functions used to set up the game (per-instance via GameConfig, no module globals)."""

from __future__ import annotations

import glob
import importlib
import os
import random
import sys
import uuid
from enum import StrEnum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

from dominion import Keys, Piles
from dominion.Artifact import Artifact
from dominion.Boon import Boon, BoonPile
from dominion.BotPlayer import BotPlayer
from dominion.Card import Card, CardExpansion, CardType
from dominion.CardPile import CardPile
from dominion.Cost import Cost
from dominion.Event import Event
from dominion.GameConfig import GameConfig
from dominion.Hex import Hex, HexPile
from dominion.Landmark import Landmark
from dominion.Loot import LootPile
from dominion.Names import playerNames
from dominion.Player import Player
from dominion.Project import Project
from dominion.Prophecy import Prophecy
from dominion.RandobotPlayer import RandobotPlayer
from dominion.TextPlayer import TextPlayer
from dominion.Trait import Trait
from dominion.Way import Way

if TYPE_CHECKING:  # pragma: no coverage
    from dominion.Game import Game


###########################################################################
class Flag(StrEnum):
    """Flags to control game setup behaviour (kept for compatibility)."""

    ALLOW_POTIONS = auto()
    ALLOW_SHELTERS = auto()
    BOT = auto()
    LOADED_PRIZES = auto()
    LOADED_TRAVELLERS = auto()
    NUM_STACKS = auto()
    RANDOBOT = auto()
    USE_OLD_CARDS = auto()
    USE_PROSPERITY = auto()


BASE_CARDS = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]

PAGE_TRAVELLERS = ("Treasure Hunter", "Warrior", "Hero", "Champion")
PEASANT_TRAVELLERS = ("Soldier", "Fugitive", "Disciple", "Teacher")


###########################################################################
def _get_config(game: "Game") -> GameConfig:
    if not hasattr(game, "config") or game.config is None:
        game.config = GameConfig.from_kwargs()
    return game.config


###########################################################################
def _proportional_card(game: "Game", used: Optional[list[str]] = None) -> Optional[Card]:
    """Choose a Kingdom randomizer not used by the other proportional check."""
    names = [
        name for name in (game.kingdom_piles or list(game.card_piles))
        if name in game.card_instances and not game.card_instances[name].basecard
    ]
    if not names:
        return None
    candidates = [name for name in names if used is None or name not in used]
    name = random.choice(candidates or names)
    if used is not None:
        used.append(name)
    return game.card_instances[name]


def use_shelters_in_game(
    game: "Game", allow_shelters: bool, specified: list[str], used: Optional[list[str]] = None
) -> bool:
    """Determine whether Shelters are used based ONLY on the Kingdom card piles (Ch. I p. 9)."""
    if "shelters" in [_.lower() for _ in specified]:
        return True
    if not allow_shelters:
        return False

    card = _proportional_card(game, used)
    return card is not None and card.base == CardExpansion.DARKAGES


###########################################################################
def use_prosperity_in_game(
    game: "Game", specified_flag: Optional[bool], used: Optional[list[str]] = None
) -> bool:
    """Determine whether Colony/Platinum are used (Ch. I p. 8 proportional check)."""
    if specified_flag is not None:
        return bool(specified_flag)
    card = _proportional_card(game, used)
    return card is not None and card.base == CardExpansion.PROSPERITY


###########################################################################
def setup_shelters(game: "Game") -> None:
    """Setup Shelters (non-Supply staging pile; Shelters do not belong to any Supply pile)."""
    game.card_piles["Shelters"] = CardPile(game, in_supply=False)
    for _ in range(game.numplayers):
        for shelter in ("Overgrown Estate", "Hovel", "Necropolis"):
            inst = game.card_mapping["Shelter"][shelter]()
            inst.game = game
            game.card_instances[shelter] = inst
            shelter_card = game.card_mapping["Shelter"][shelter]()
            shelter_card.game = game
            shelter_card.pile = "Shelters"
            shelter_card.location = Piles.CARDPILE
            game.cards[shelter_card.uuid] = shelter_card
            game.card_piles["Shelters"].add(shelter_card)


###########################################################################
def load_travellers(game: "Game", line: Optional[str] = None) -> None:
    """Load Traveller upgrade piles outside the Supply (Ch. I p. 9: Page vs Peasant lines)."""
    cfg = _get_config(game)
    if line == "Page":
        names = [n for n in PAGE_TRAVELLERS if n in game.card_mapping["Traveller"]]
    elif line == "Peasant":
        names = [n for n in PEASANT_TRAVELLERS if n in game.card_mapping["Traveller"]]
    else:
        names = game.get_available_cards("Traveller")
    for trav in names:
        if trav in cfg.loaded_travellers or trav in game.card_piles:
            continue
        use_card_pile(game, None, trav, force=True, card_type="Traveller", in_supply=False)
        cfg.loaded_travellers.add(trav)


############################################################################
def load_loot(game: "Game") -> None:
    """Load Loot cards into game (non-Supply face-down pile)."""
    if "Loot" in game.card_piles:
        return
    cfg = _get_config(game)
    game.output("Using Loot cards")
    pile = LootPile(game, get_card_classes("Loot", cfg.paths[Keys.LOOT], "Loot_"))
    pile.init_cards()
    game.card_piles["Loot"] = pile
    for card in pile:
        game.cards[card.uuid] = card
    for card in pile:
        game.cards[card.uuid] = card


###########################################################################
def load_ways(game: "Game", specified: list[str], num_required: int) -> dict[str, Way]:
    """Load required Ways"""
    way_cards = []
    for way_name in specified:
        if not way_name.lower().startswith("way of the "):
            way_cards.append(f"Way of the {way_name}")
        else:
            way_cards.append(way_name)
    ways = load_non_kingdom_cards(
        game,
        cardtype="Way",
        specified=way_cards,
        num_required=num_required,
    )
    for way in ways:
        game.output(f"Playing with {way}")
    return cast(dict[str, Way], ways)


###########################################################################
def load_events(game: "Game", specified: list[str], num_required: int) -> dict[str, Event]:
    """Load required events"""
    events = load_non_kingdom_cards(
        game,
        cardtype="Event",
        specified=specified,
        num_required=num_required,
    )
    for event in events:
        game.output(f"Playing with Events: {event}")
    return cast(dict[str, Event], events)


###########################################################################
def load_hexes(game: "Game") -> list[Hex]:
    """Load Hexes into shuffled deck"""
    game.output("Using hexes")
    d = load_non_kingdom_pile(game, "Hex", HexPile)
    hexes = cast(list[Hex], list(d.values()))
    random.shuffle(hexes)
    return hexes


###########################################################################
def load_projects(game: "Game", specified: list[str], num_required: int) -> dict[str, Project]:
    """Load and return Projects"""
    projects = load_non_kingdom_cards(
        game,
        "Project",
        specified,
        num_required,
    )
    for project in projects:
        game.output(f"Playing with Project: {project}")
    return cast(dict[str, Project], projects)


###########################################################################
def load_prophecies(game: "Game", specified: list[str]) -> Prophecy:
    """Load all the prophecies and return one"""
    prophecies = load_non_kingdom_cards(
        game,
        "Prophecies",
        specified,
        1,
    )
    prophecy = prophecies[list(prophecies.keys())[0]]
    return cast(Prophecy, prophecy)


###########################################################################
def load_artifacts(game: "Game") -> dict[str, Artifact]:
    """Load and return artifacts"""
    artifacts = load_non_kingdom_cards(game, "Artifact", [], None)
    return cast(dict[str, Artifact], artifacts)


###########################################################################
def load_landmarks(game: "Game", specified: list[str], num_required: int) -> dict[str, Landmark]:
    """Load required landmarks"""
    if landmarks := load_non_kingdom_cards(
        game,
        "Landmark",
        specified,
        num_required,
    ):
        for landmark in landmarks:
            game.output(f"Playing with Landmark: {landmark}")
    return cast(dict[str, Landmark], landmarks)


###########################################################################
def card_piles_for_trait(game: "Game") -> list[str]:
    """Return Supply Kingdom Action/Treasure piles suitable for Traits (Ch. I p. 10–11).

    Excludes Base cards, Ruins, non-Supply piles (Ferryman/Loot/Travellers/etc.),
    and piles already carrying a Trait. Includes Young Witch / Approaching Army piles.
    """
    card_piles: list[str] = []
    for pile_name, pile in game.get_supply_piles():
        if pile.trait:
            continue
        if pile_name in BASE_CARDS or pile_name in ("Platinum", "Colony", "Potion", "Ruins"):
            continue
        if not pile.is_kingdom:
            continue
        types = game._pile_types(pile_name, pile)
        if CardType.ACTION not in types and CardType.TREASURE not in types:
            continue
        card_piles.append(pile_name)
    return card_piles


###########################################################################
def card_pile_for_trait(game: "Game", pile_selector: Callable[["Game"], list[str]] = card_piles_for_trait) -> str:
    """Return a card pile that is suitable for traits"""
    card_piles = pile_selector(game)
    return random.choice(card_piles)


###########################################################################
def load_traits(
    game: "Game",
    specified: list[str],
    num_required: int,
    pile_picker: Callable[["Game"], str] = card_pile_for_trait,
) -> None:
    """Load required Traits"""
    game.traits = cast(
        dict[str, Trait],
        load_non_kingdom_cards(
            game,
            cardtype="Trait",
            specified=specified,
            num_required=num_required,
        ),
    )
    for trait in game.traits:
        card_pile = pile_picker(game)
        game.assign_trait(trait, card_pile)
        game.output(f"Playing with Trait: {trait}")


###########################################################################
def load_states(game: "Game") -> None:
    """Load States"""
    if game.states:
        return
    game.output("Using states")
    game.states = load_non_kingdom_cards(game, "State", [], None)


###########################################################################
def load_boons(game: "Game") -> None:
    """Load boons into shuffled deck"""
    if game.boons:
        return
    game.output("Using boons")
    d = cast(dict[str, Boon], load_non_kingdom_pile(game, "Boon", BoonPile))
    game.boons = list(d.values())
    random.shuffle(game.boons)


###########################################################################
def load_ally(game: "Game", specified: str | list[str]) -> Card:
    """Load the ally"""
    if isinstance(specified, str):
        specified = [specified]
    allies = load_non_kingdom_cards(game, "Ally", specified, 1)
    ally = list(allies.values())[0]
    game.output(f"Playing with Ally {ally}")
    return ally


###########################################################################
def get_card_classes(prefix: str, path: str, class_prefix: str = "Card_") -> dict[str, type[Card]]:
    """Import all modules matching `{path}/{prefix}_*.py` and map card name -> class."""
    mapping: dict[str, type[Card]] = {}
    files = glob.glob(f"{path}/{prefix}_*.py")
    for file_name in [os.path.basename(_) for _ in files]:
        file_name = file_name.replace(".py", "")
        mod = importlib.import_module(f"{path.replace('/', '.')}.{file_name}")

        classes = dir(mod)
        for kls in classes:
            if kls.startswith(class_prefix):
                klass = getattr(mod, kls)
                break
        else:  # pragma: no cover
            raise ImportError(f"Couldn't find {prefix} class in {path}\n")
        inst = klass()
        mapping[inst.name] = klass
        inst.check()
    return mapping


###########################################################################
def handle_specified_card_requirements(game: "Game", card: Card) -> None:
    """Look at the 'required_cards' value"""
    for x in card.required_cards:
        if x == "Loot":
            load_loot(game)
            continue
        if isinstance(x, tuple):
            card_type, card_name = x
        else:
            card_type, card_name = "BaseCard", x
        if card_name not in game.card_piles and card_name not in game.card_instances:
            in_supply = card_name in ("Potion", "Curse", "Ruins", "Platinum", "Colony")
            if card_type == "Card" and card_name in game.card_mapping["Card"]:
                probe = game.card_mapping["Card"][card_name]()
                in_supply = probe.insupply
            use_card_pile(game, None, card_name, force=True, card_type=card_type, in_supply=in_supply)
            game.output(f"Using {card_name} as required by {card.name}")


###########################################################################
def check_heirloom_requirements(game: "Game", card: Card) -> None:
    """Check if Heirlooms are required (1 per player, NOT a Supply pile; Ch. I p. 10)."""
    if card.heirloom is not None and card.heirloom not in game.heirlooms:
        heirloom_name = card.heirloom
        klass = game.card_mapping["Heirloom"][heirloom_name]
        inst = klass()
        inst.game = game
        inst.insupply = False
        game.card_instances[heirloom_name] = inst
        # Stage exactly `game.numplayers` copies in a non-Supply pile until dealt to players
        pile = CardPile(game, in_supply=False)
        pile.init_cards(game.numplayers, klass)
        instantiate_card_pile(game, heirloom_name, pile, "Heirloom", in_supply=False)
        game.heirlooms.append(heirloom_name)
        game.output(f"Using {heirloom_name} as required by {card.name}")


###########################################################################
def check_ruins_requirements(game: "Game", card: Card) -> None:
    """Include Ruins Supply pile if any Looter is in the game (Ch. I p. 9)."""
    if card.isLooter() and "Ruins" not in game.card_piles:
        use_card_pile(game, None, "Ruins", force=True, in_supply=True, is_kingdom=False)
        game.output(f"Using Ruins as required by {card.name}")


###########################################################################
def check_prize_requirements(game: "Game", card: Card) -> None:
    """Include Prizes outside the Supply if required (Ch. I p. 9)."""
    cfg = _get_config(game)
    if card.needs_prizes and not cfg.loaded_prizes:
        for prize in game.get_available_cards("PrizeCard"):
            use_card_pile(game, None, prize, force=True, card_type="PrizeCard", in_supply=False)
        cfg.loaded_prizes = True
        game.output(f"Using Prizes as required by {card.name}")


###########################################################################
def check_artifact_requirements(game: "Game", card: Card) -> None:
    """Check to see if artifacts are required"""
    if card.needsartifacts and not game.artifacts:
        game.artifacts = load_artifacts(game)
        game.output(f"Using artifacts as required by {card.name}")


###########################################################################
def check_prophecy_requirements(game: "Game", card: Card) -> None:
    """Check to see if prophecies are required"""
    cfg = _get_config(game)
    if card.isOmen() and not game.inactive_prophecy:
        game.inactive_prophecy = load_prophecies(game, cfg.init_cards[Keys.PROPHECIES])
        game.output(f"Playing with Prophecy {game.inactive_prophecy.name}")
        game.sun_tokens = get_num_sun_tokens(game)


###########################################################################
def check_project_requirements(game: "Game", card: Card) -> None:
    """Check to see if projects are required"""
    cfg = _get_config(game)
    if card.needsprojects and not game.projects:
        game.projects = load_projects(game, cfg.init_cards[Keys.PROJECTS], cfg.init_numbers[Keys.PROJECTS])
        game.output(f"Using projects as required by {card.name}")


###########################################################################
def check_allies_requirements(game: "Game", card: Card) -> None:
    """Check to see if allies are required"""
    cfg = _get_config(game)
    if card.isLiaison() and not game.ally:
        game.ally = load_ally(game, cfg.init_cards[Keys.ALLIES])
        game.output(f"Using Allies as required by {card.name}")


###########################################################################
def check_traveller_requirements(game: "Game", card: Card) -> None:
    """Include ONLY the Traveller upgrade line needed for Page or Peasant (Ch. I p. 9)."""
    if card.traveller:
        if card.name == "Page":
            load_travellers(game, line="Page")
        elif card.name == "Peasant":
            load_travellers(game, line="Peasant")
        else:
            load_travellers(game)


###########################################################################
def check_boon_requirements(game: "Game", card: Card) -> None:
    """Check to see if boons are required"""
    if card.isFate() and not game.boons:
        load_boons(game)


###########################################################################
def check_hexes_requirements(game: "Game", card: Card) -> None:
    """Check to see if hexes are required"""
    if card.isDoom() and not game.hexes:
        game.hexes = load_hexes(game)
        game.output(f"Using hexes as required by {card.name}")


###########################################################################
def check_potion_requirements(game: "Game", card: Card) -> None:
    """Include 16 Potions in Supply if any Kingdom card has Potion in its cost (Ch. I p. 8)."""
    if card.potcost and "Potion" not in game.card_piles and "Potion" in game.card_mapping["BaseCard"]:
        use_card_pile(game, None, "Potion", force=True, card_type="BaseCard", in_supply=True)
        game.output(f"Using Potion as required by {card.name}")


###########################################################################
def check_card_requirement(game: "Game", card: Card) -> None:
    """Ensure all the requirements (e.g. Potion, Ruins, Heirlooms) for a card are loaded"""
    handle_specified_card_requirements(game, card)
    check_potion_requirements(game, card)
    check_heirloom_requirements(game, card)
    check_ruins_requirements(game, card)
    check_prize_requirements(game, card)
    check_artifact_requirements(game, card)
    check_prophecy_requirements(game, card)
    check_project_requirements(game, card)
    check_traveller_requirements(game, card)
    check_allies_requirements(game, card)
    check_boon_requirements(game, card)
    check_hexes_requirements(game, card)


###########################################################################
def get_num_sun_tokens(game: "Game") -> int:
    """Return the number of sun tokens based on the number of players"""
    return {1: 3, 2: 5, 3: 8, 4: 10, 5: 12, 6: 13}[game.numplayers]


###########################################################################
def get_available_card_classes(config: Optional[GameConfig] = None) -> dict[str, dict[str, type[Card]]]:
    """Create a mapping between the card name and the class of that card"""
    cfg = config or GameConfig.from_kwargs()
    mapping: dict[str, dict[str, type[Card]]] = {}
    for prefix in (
        "Card",
        "BaseCard",
        "Traveller",
        "PrizeCard",
        "Castle",
        "Heirloom",
        "Shelter",
        "Split",
    ):
        mapping[prefix] = get_card_classes(prefix, cfg.paths[Keys.CARDS], "Card_")
        if cfg.use_old_cards:
            old_path = os.path.join(cfg.paths[Keys.CARDS], "old")
            mapping[prefix].update(get_card_classes(prefix, old_path, "Card_"))
    mapping["Event"] = get_card_classes("Event", cfg.paths[Keys.EVENT], "Event_")
    mapping["Way"] = get_card_classes("Way", cfg.paths[Keys.WAY], "Way_")
    mapping["Landmark"] = get_card_classes("Landmark", cfg.paths[Keys.LANDMARK], "Landmark_")
    mapping["Boon"] = get_card_classes("Boon", cfg.paths[Keys.BOONS], "Boon_")
    mapping["Hex"] = get_card_classes("Hex", cfg.paths[Keys.HEXES], "Hex_")
    mapping["State"] = get_card_classes("State", cfg.paths[Keys.STATES], "State_")
    mapping["Artifact"] = get_card_classes("Artifact", cfg.paths[Keys.ARTIFACTS], "Artifact_")
    mapping["Project"] = get_card_classes("Project", cfg.paths[Keys.PROJECTS], "Project_")
    mapping["Prophecies"] = get_card_classes("Prophecy", cfg.paths[Keys.PROPHECIES], "Prophecy_")
    mapping["Ally"] = get_card_classes("Ally", cfg.paths[Keys.ALLIES], "Ally_")
    mapping["Trait"] = get_card_classes("Trait", cfg.paths[Keys.TRAITS], "Trait_")
    mapping["Loot"] = get_card_classes("Loot", cfg.paths[Keys.LOOT], "Loot_")
    return mapping


###########################################################################
def load_non_kingdom_pile(game: "Game", cardtype: str, pileClass: Any) -> dict[str, Any]:
    """Load non kingdom cards into a pile"""
    dest: dict[str, Any] = {}
    available = game.get_available_cards(cardtype)
    for nkc in available:
        klass = game.card_mapping[cardtype][nkc]
        dest[nkc] = pileClass(nkc, klass)
    return dest


###########################################################################
def instantiate_non_kingdom_card(game: "Game", cardtype: str, card_name: str) -> Card:
    """Create an instance of a non-kingdom card."""
    klass = game.card_mapping[cardtype][card_name]
    inst = klass()
    inst.game = game
    return inst


###########################################################################
def check_card_requirements(game: "Game") -> None:
    """Ensure all required companions for all cards, landscapes, boons, hexes, etc. are loaded."""
    cfg = _get_config(game)
    seen: set[int] = set()
    # Loop until fixed point (e.g., Fate -> Boons -> Swamp's Gift -> Will-o'-Wisp)
    while True:
        check_cards: list[Any] = (
            list(game.cards.values())
            + list(game.events.values())
            + list(game.traits.values())
            + list(game.projects.values())
            + list(game.ways.values())
            + list(game.hexes)
            + list(game.boons)
            + list(game.landmarks.values())
            + [game.inactive_prophecy]
        )
        if game.ally:
            check_cards.append(game.ally)

        new_cards = [c for c in check_cards if c is not None and id(c) not in seen]
        if not new_cards:
            break
        for card in new_cards:
            seen.add(id(card))
            check_card_requirement(game, card)

    if cfg.init_cards[Keys.ALLIES] and not game.ally:
        print(f"Need to specify a Liaison as well as an Ally {cfg.init_cards[Keys.ALLIES]}")
        sys.exit(1)


###########################################################################
def guess_card_name(game: "Game", name: str, prefix: str = "Card") -> Optional[str]:
    """Fuzzy-match a card name against available cards of `prefix`."""
    available = game.get_available_cards(prefix)
    if name in available:
        return name
    lower_name = name.lower()
    for card_name in available:
        if card_name.lower() == lower_name:
            return card_name
        newc = card_name.lower().replace("'", "")
        if newc == lower_name:
            return card_name
        if newc.replace(" ", "_") == lower_name:
            return card_name
        if newc.replace(" ", "-") == lower_name:
            return card_name
        if newc.replace(" ", "") == lower_name:
            return card_name
        if newc.replace(" ", "_").replace("_", "") == lower_name:
            return card_name
    return None


###########################################################################
def good_names(game: "Game", specified: list[str], cardtype: str) -> list[str]:
    """Resolve user-specified names to exact card names."""
    answer: list[str] = []
    for name in specified:
        if name in game.card_mapping[cardtype]:
            answer.append(name)
        else:
            good_name = guess_card_name(game, name, cardtype)
            if good_name is None:
                sys.stderr.write(f"Unknown {cardtype} '{name}'\n")
                sys.exit(1)
            answer.append(good_name)
    return answer


###########################################################################
def load_non_kingdom_cards(
    game: "Game", cardtype: str, specified: list[str], num_required: Optional[int] = None
) -> dict[str, Card]:
    """Load non-kingdom cards (Events, Projects, Ways, Landmarks, Traits, etc.)."""
    dest: dict[str, Card] = {}
    available = game.get_available_cards(cardtype)
    if specified is not None:
        names = good_names(game, specified, cardtype)
        for nkc in names:
            dest[nkc] = instantiate_non_kingdom_card(game, cardtype, nkc)
            if nkc in available:
                available.remove(nkc)

    if num_required is not None:
        while len(dest) < num_required and available:
            nkc = random.choice(available)
            dest[nkc] = instantiate_non_kingdom_card(game, cardtype, nkc)
            available.remove(nkc)
    else:
        for nkc in available:
            dest[nkc] = instantiate_non_kingdom_card(game, cardtype, nkc)

    return dest


###########################################################################
def use_card_pile(
    game: "Game",
    available: Optional[list[str]],
    card_name: str,
    force: bool = False,
    card_type: str = "Card",
    in_supply: Optional[bool] = None,
    is_kingdom: Optional[bool] = None,
) -> int:
    """Set up a card pile for use.
    Returns 1 if it counts against the number of Kingdom card piles in use.
    """
    cfg = _get_config(game)
    try:
        if available is not None:
            available.remove(card_name)
    except ValueError:  # pragma: no cover
        print(f"Unknown card '{card_name}'\n", file=sys.stderr)
        sys.exit(1)
    card = game.card_mapping[card_type][card_name]()
    card.game = game
    if not cfg.allow_potions and card.potcost:
        return 0
    if not force and not card.insupply:
        return 0
    if not force and not card.is_kingdom:
        return 0

    pile_in_supply = card.insupply if in_supply is None else in_supply
    if card_type in ("Traveller", "PrizeCard", "Heirloom", "Shelter"):
        pile_in_supply = False
    if card_name in ("Spoils", "Loot", "Shelters"):
        pile_in_supply = False

    if hasattr(card, "cardpile_setup"):
        card_pile = card.cardpile_setup(game)
        card_pile.game = game
        card_pile.in_supply = pile_in_supply
    else:
        card_pile = CardPile(game, in_supply=pile_in_supply)
    num_cards = num_cards_in_pile(game, card)
    card_pile.init_cards(num_cards, game.card_mapping[card_type][card_name])

    pile_is_kingdom = (
        is_kingdom if is_kingdom is not None else (card_type == "Card" and pile_in_supply and card.is_kingdom)
    )
    card_pile.is_kingdom = pile_is_kingdom

    # Record Randomizer metadata (Ch. IV Split Piles p. 36)
    if not card_pile.randomizer_types:
        raw_types = card.get_raw_card_type()
        if raw_types:
            card_pile.randomizer_types = list(raw_types)
        elif not card_pile.is_empty():
            card_pile.randomizer_types = list(card_pile.get_top_card().get_raw_card_type())
    if card_pile.randomizer_cost is None:
        if card.cost >= 0:
            card_pile.randomizer_cost = Cost.from_card(card)
        elif not card_pile.is_empty():
            card_pile.randomizer_cost = Cost.from_card(card_pile.get_top_card())

    instantiate_card_pile(game, card_name, card_pile, card_type, in_supply=pile_in_supply)

    if pile_is_kingdom and card_name not in game.kingdom_piles:
        game.kingdom_piles.append(card_name)

    game.output(f"Playing with {card_name}")
    return 1


###########################################################################
def instantiate_card_pile(
    game: "Game",
    card_name: str,
    card_pile: CardPile,
    card_type: str,
    in_supply: Optional[bool] = None,
) -> None:
    """Register cards in `card_pile` onto `game`."""
    if in_supply is not None:
        card_pile.in_supply = in_supply
    game.card_piles[card_name] = card_pile
    if card_name not in game.card_instances:
        inst = game.card_mapping[card_type][card_name]()
        inst.game = game
        game.card_instances[card_name] = inst
    for card in card_pile:
        card.game = game
        game.cards[card.uuid] = card
        if not card.pile:
            card.pile = card_name
        card.location = Piles.CARDPILE


###########################################################################
def num_cards_in_pile(game: "Game", card: Card) -> int:
    """Return the number of cards that should be in a card pile (Ch. I Basic Setup p. 8)."""
    if CardType.PRIZE in card.get_raw_card_type():
        return 1
    if hasattr(card, "calc_numcards"):
        return card.calc_numcards(game)
    # Victory Kingdom piles use 8 cards in 2p (or 1p) and 12 in 3+p
    if card.isVictory() and not card.basecard and getattr(card, "numcards", 10) == 10:
        return 8 if game.numplayers <= 2 else 12
    if hasattr(card, "numcards"):
        return card.numcards
    return 10


###########################################################################
def load_decks(game: "Game", initcards: list[str], numstacks: int) -> None:
    """Determine Kingdom cards, proportional Colonies/Shelters, and Base Supply piles."""
    cfg = _get_config(game)

    available = game.get_available_cards()
    unfilled = numstacks
    found_all = True
    specified_base_cards: list[str] = []

    for crd in initcards:
        if crd in ("Boons", "Hexes"):
            continue
        if base_name := guess_card_name(game, crd, prefix="BaseCard"):
            specified_base_cards.append(base_name)
            continue
        result = place_init_card(game, crd, available)
        if result is None:
            found_all = False
        else:
            unfilled -= result

    if not found_all:
        sys.exit(1)

    while unfilled > 0:
        if not available:
            break
        crd = random.choice(available)
        if crd in cfg.init_cards[Keys.BAD_CARDS]:
            available.remove(crd)
            continue
        unfilled -= use_card_pile(game, available, crd)

    # Determine Shelters and Colonies/Platinum based on the Kingdom piles BEFORE creating Estate pile
    used_randomizers: list[str] = []
    game.use_shelters = use_shelters_in_game(game, cfg.allow_shelters, cfg.init_cards[Keys.CARDS], used_randomizers)
    use_prosperity = use_prosperity_in_game(game, cfg.use_prosperity, used_randomizers)

    base_cards = BASE_CARDS[:]
    if use_prosperity:
        base_cards.extend(["Colony", "Platinum"])
    for extra_base in specified_base_cards:
        if extra_base not in base_cards:
            base_cards.append(extra_base)

    for card in base_cards:
        if card not in game.card_piles and card in game.card_mapping["BaseCard"]:
            use_card_pile(game, None, card, force=True, card_type="BaseCard", in_supply=(card != "Spoils"))

    check_card_requirements(game)


###########################################################################
def place_init_card(game: "Game", card: str, available: list[str]) -> Optional[int]:
    """For the specified card, load it into the correct deck or landscape list."""
    cfg = _get_config(game)
    if card_name := guess_card_name(game, card, prefix="BaseCard"):
        use_card_pile(game, None, card_name, force=True, card_type="BaseCard", in_supply=(card_name != "Spoils"))
        return 0
    if card_name := guess_card_name(game, card):
        return use_card_pile(game, available, card_name, force=True)
    mapping = {
        "Event": Keys.EVENT,
        "Way": Keys.WAY,
        "Landmark": Keys.LANDMARK,
        "Project": Keys.PROJECTS,
        "Ally": Keys.ALLIES,
        "Trait": Keys.TRAITS,
        "Prophecies": Keys.PROPHECIES,
    }
    for prefix, key in mapping.items():
        if name := guess_card_name(game, card, prefix):
            cfg.init_cards[key].append(name)
            return 0
    if guess_card_name(game, card, "Boon"):
        load_boons(game)
        return 0
    if guess_card_name(game, card, "Artifact"):
        return 0
    if card.lower() == "shelters":
        return 0
    for prefix in ("Traveller", "Castle", "Loot", "Heirloom", "State", "Hex", "PrizeCard"):
        if guess_card_name(game, card, prefix):
            return 0
    print(f"Can't guess what card '{card}' is")
    return None


###########################################################################
def instantiate_player_class(game: "Game", name: str, use_shelters: bool, player_num: int, the_uuid: str) -> Player:
    """Create the player instances"""
    cfg = _get_config(game)
    if cfg.bot:
        plr_class: type[Player] = BotPlayer
        name = f"{name}Bot"
        cfg.bot = False
    elif cfg.randobot:
        plr_class = RandobotPlayer
        name = f"{name}RandoBot"
        cfg.randobot -= 1
    else:
        plr_class = TextPlayer

    player = plr_class(
        game=game,
        quiet=game.quiet,
        name=name,
        heirlooms=game.heirlooms,
        use_shelters=use_shelters,
        number=player_num,
    )
    player.uuid = the_uuid
    return player


###########################################################################
def setup_players(
    game: "Game",
    use_shelters: bool,
    playernames: Optional[list[str]] = None,
) -> None:
    """Set up the players"""
    names = playerNames[:]
    if playernames is None:
        playernames = []
    else:
        playernames = list(playernames)

    for player_num in range(game.numplayers):
        try:
            name = playernames.pop()
        except IndexError:
            name = random.choice(names)
            names.remove(name)
        the_uuid = uuid.uuid4().hex
        game.players[the_uuid] = instantiate_player_class(game, name, use_shelters, player_num, the_uuid)

    # Heirlooms & Shelters do NOT belong to any Supply/non-Supply pile once dealt
    for heirloom_name in list(game.heirlooms):
        if heirloom_name in game.card_piles and game.card_piles[heirloom_name].is_empty():
            del game.card_piles[heirloom_name]
    if "Shelters" in game.card_piles and game.card_piles["Shelters"].is_empty():
        del game.card_piles["Shelters"]


###########################################################################
def parse_args(game: "Game", **args: Any) -> None:
    """Attach GameConfig from kwargs onto `game`."""
    game.config = GameConfig.from_kwargs(**args)
    game.quiet = bool(args.get("quiet", False))
    game.numplayers = int(args.get("numplayers", 2))


###########################################################################
def _supply_setup_cards(game: "Game") -> None:
    """Run pile-adding / Supply-modifying setups BEFORE Traits and setup_players.

    Handles Young Witch (Bane), Riverboat, Approaching Army, Necromancer/Zombies,
    and Druid so extra piles can receive Traits and their Heirlooms enter starting decks.
    """
    supply_phase_cards = {"Young Witch", "Riverboat", "Necromancer", "Druid"}
    done = game.specials.setdefault("_setup_done", set())
    while True:
        pending = [name for name in game.card_piles if name in supply_phase_cards and name not in done]
        if not pending:
            break
        for name in pending:
            setup_card_pile(game, name)
    if game.inactive_prophecy and game.inactive_prophecy.name == "Approaching Army":
        key = "Prophecy:Approaching Army"
        if key not in done:
            game.inactive_prophecy.setup(game=game)
            done.add(key)
            # The extra Attack can add another pile with an early setup rule.
            _supply_setup_cards(game)


def setup_card_pile(game: "Game", pile_name: str) -> None:
    """Run and record a pile's setup once."""
    done = game.specials.setdefault("_setup_done", set())
    if pile_name not in done:
        game.card_piles[pile_name].setup(game=game)
        done.add(pile_name)


###########################################################################
def card_setup(game: "Game") -> None:
    """Run the remaining setup() methods for all cards and landscapes (after players exist)."""
    done: set[str] = game.specials.setdefault("_setup_done", set())
    for pile_name, card_pile in list(game.card_piles.items()):
        if pile_name in done:
            continue
        setup_card_pile(game, pile_name)
    if set_aside := game.specials.get("riverboat"):
        key = f"Riverboat:{set_aside}"
        if key not in done:
            game.card_instances[set_aside].setup(game=game)
            done.add(key)
    for landmark in list(game.landmarks.values()):
        landmark.setup(game=game)
    for event in game.events.values():
        event.setup(game=game)
    if game.inactive_prophecy and f"Prophecy:{game.inactive_prophecy.name}" not in done:
        game.inactive_prophecy.setup(game=game)
        done.add(f"Prophecy:{game.inactive_prophecy.name}")


###########################################################################
def start_game(
    game: "Game",
    player_names: Optional[list[str]] = None,
) -> None:
    """Initialize all piles, landscapes, players, setups, and TurnQueue."""
    cfg = _get_config(game)
    cfg.loaded_travellers.clear()
    cfg.loaded_prizes = False

    load_decks(game, cfg.init_cards[Keys.CARDS], cfg.num_stacks)
    game.events = load_events(game, cfg.init_cards[Keys.EVENT], cfg.init_numbers[Keys.EVENT])
    game.ways = load_ways(game, cfg.init_cards[Keys.WAY], cfg.init_numbers[Keys.WAY])
    game.landmarks = load_landmarks(game, cfg.init_cards[Keys.LANDMARK], cfg.init_numbers[Keys.LANDMARK])
    game.artifacts = {}
    game.projects = load_projects(game, cfg.init_cards[Keys.PROJECTS], cfg.init_numbers[Keys.PROJECTS])

    if game.hexes or game.boons:
        load_states(game)
    check_card_requirements(game)

    # Run Supply-modifying setups (Young Witch, Riverboat, Approaching Army, Necromancer, Druid)
    _supply_setup_cards(game)
    check_card_requirements(game)
    if game.hexes or game.boons:
        load_states(game)

    # Traits can now be placed on Young Witch Bane & Approaching Army piles!
    load_traits(game, cfg.init_cards[Keys.TRAITS], cfg.init_numbers[Keys.TRAITS])

    if game.use_shelters:
        setup_shelters(game)
    setup_players(game, game.use_shelters, player_names)

    # Player-dependent & token-placement setups (Baker, Importer, Trade Route, Tax, Landmarks)
    card_setup(game)
    check_card_requirements(game)
    if game.hexes or game.boons:
        load_states(game)

    # Initialize TurnQueue (random starting player in real games; deterministic index 0 in TestGame)
    uuids = [plr.uuid for plr in game.player_list()]
    game.turn_queue.initialize(uuids, random_start=cfg.random_start_player)
    if uuids:
        start_uuid = game.turn_queue.player_order[game.turn_queue.next_regular_index]
        game.current_player = game.players[start_uuid]

    if game.ally:
        for plr in game.player_list():
            plr.favors.add(1)


# EOF
