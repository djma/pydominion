"""Per-game configuration dataclass (replaces module-level globals in game_setup)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from dominion import Keys


@dataclass
class GameConfig:
    """Configuration for a single Dominion game instance."""

    paths: dict[Keys, str] = field(default_factory=dict)
    init_numbers: dict[Keys, int] = field(default_factory=dict)
    init_cards: dict[Keys, list[Any]] = field(default_factory=dict)

    num_stacks: int = 10
    allow_potions: bool = True
    use_prosperity: Optional[bool] = None  # None = proportional random check, True/False = forced
    use_old_cards: bool = False
    allow_shelters: bool = True
    bot: bool = False
    randobot: int = 0
    random_start_player: bool = True

    loaded_travellers: set[str] = field(default_factory=set)
    loaded_prizes: bool = False

    @classmethod
    def _as_list(cls, val: Any) -> list[Any]:
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        return list(val)

    @classmethod
    def from_kwargs(cls, **args: Any) -> "GameConfig":
        """Build a GameConfig from Game(**kwargs) arguments."""
        paths = {
            Keys.ALLIES: args.get("ally_path", "dominion/allies"),
            Keys.ARTIFACTS: args.get("artifact_path", "dominion/artifacts"),
            Keys.BOONS: args.get("boon_path", "dominion/boons"),
            Keys.CARDS: args.get("card_path", "dominion/cards"),
            Keys.EVENT: args.get("event_path", "dominion/events"),
            Keys.HEXES: args.get("hex_path", "dominion/hexes"),
            Keys.LANDMARK: args.get("landmark_path", "dominion/landmarks"),
            Keys.LOOT: args.get("loot_path", "dominion/loot"),
            Keys.PROJECTS: args.get("project_path", "dominion/projects"),
            Keys.PROPHECIES: args.get("prophecies_path", "dominion/prophecies"),
            Keys.STATES: args.get("state_path", "dominion/states"),
            Keys.TRAITS: args.get("trait_path", "dominion/traits"),
            Keys.WAY: args.get("way_path", "dominion/ways"),
        }
        init_numbers = {
            Keys.EVENT: args.get("num_events", 0),
            Keys.LANDMARK: args.get("num_landmarks", 0),
            Keys.PROJECTS: args.get("num_projects", 0),
            Keys.TRAITS: args.get("num_traits", 0),
            Keys.WAY: args.get("num_ways", 0),
        }
        prophecies = cls._as_list(args.get("prophecies", []))
        if "prophecies" not in args and "prophecy" in args:
            prophecies = cls._as_list(args.get("prophecy", []))

        projects = cls._as_list(args.get("projects", []))
        if not projects and "init_projects" in args:
            projects = cls._as_list(args.get("init_projects", []))

        init_cards_arg = cls._as_list(args.get("initcards", []))
        if not init_cards_arg and "cards" in args:
            init_cards_arg = cls._as_list(args.get("cards", []))

        init_cards = {
            Keys.ALLIES: cls._as_list(args.get("allies", [])),
            Keys.BAD_CARDS: cls._as_list(args.get("badcards", [])),
            Keys.CARDS: init_cards_arg,
            Keys.EVENT: cls._as_list(args.get("events", [])),
            Keys.LANDMARK: cls._as_list(args.get("landmarks", [])),
            Keys.PROJECTS: projects,
            Keys.PROPHECIES: prophecies,
            Keys.TRAITS: cls._as_list(args.get("traits", [])),
            Keys.WAY: cls._as_list(args.get("ways", [])),
        }

        use_prosperity: Optional[bool]
        if "prosperity" in args:
            use_prosperity = args["prosperity"]
        else:
            use_prosperity = None

        return cls(
            paths=paths,
            init_numbers=init_numbers,
            init_cards=init_cards,
            num_stacks=int(args.get("num_stacks", 10)),
            allow_potions=bool(args.get("potions", True)),
            use_prosperity=use_prosperity,
            use_old_cards=bool(args.get("oldcards", False)),
            allow_shelters=bool(args.get("shelters", True)),
            bot=bool(args.get("bot", False)),
            randobot=int(args.get("randobot", 0)),
            random_start_player=bool(args.get("random_start_player", True)),
        )


# EOF
