"""State Snapshot, Rewind, and Branching Clone support for Dominion Game Engine.

Allows deterministic mid-turn or cross-turn undo/rewind (`game.snapshot()` / `game.restore(snap)`)
and independent branching simulation (`game.clone()`) for AI/MCTS lookahead and testing.

UI/IO handles (`PlayerController`, `rich.Console`) and immutable class registries
(`Game.card_mapping`) are excluded from snapshots and preserved across restores.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from dominion.Controller import (
    BigMoneyBotController,
    PlayerController,
    RandoBotController,
    TextController,
)

if TYPE_CHECKING:
    from dominion.Game import Game


###############################################################################
@dataclass
class GameSnapshot:
    """Immutable-by-convention deep snapshot of a Game's complete engine state and RNG."""

    _game: "Game"
    rng_state: tuple[Any, ...]
    game_rng_state: tuple[Any, ...]
    controller_specs: dict[str, tuple[type[PlayerController], dict[str, Any]]] = field(default_factory=dict)


###############################################################################
def _controller_spec(ctrl: PlayerController) -> tuple[type[PlayerController], dict[str, Any]]:
    """Capture minimal metadata to reconstruct a clean controller for clones."""
    if isinstance(ctrl, TextController):
        return (TextController, {"player_number": getattr(ctrl, "player_number", 0)})
    if isinstance(ctrl, BigMoneyBotController):
        return (BigMoneyBotController, {})
    if isinstance(ctrl, RandoBotController):
        return (RandoBotController, {})
    return (type(ctrl), {})


def _make_controller(spec: tuple[type[PlayerController], dict[str, Any]]) -> PlayerController:
    cls, kwargs = spec
    try:
        return cls(**kwargs)
    except Exception:
        return TextController()


###############################################################################
def capture_snapshot(game: "Game") -> GameSnapshot:
    """Capture a complete, isolated snapshot of `game` and Python's PRNG state.

    Excludes non-serializable UI handles (`Player.controller`, `rich.Console`) and
    shares the immutable `game.card_mapping` class registry.
    """
    memo: dict[int, Any] = {}
    if hasattr(game, "card_mapping"):
        memo[id(game.card_mapping)] = game.card_mapping

    controller_specs: dict[str, tuple[type[PlayerController], dict[str, Any]]] = {}
    for uid, plr in game.players.items():
        controller_specs[uid] = _controller_spec(plr.controller)
        memo[id(plr.controller)] = None
        if getattr(plr.controller, "console", None) is not None:
            memo[id(plr.controller.console)] = None
        for stacked in getattr(plr, "_controller_stack", []):
            memo[id(stacked)] = None
            if getattr(stacked, "console", None) is not None:
                memo[id(stacked.console)] = None
        if getattr(plr, "console", None) is not None:
            memo[id(plr.console)] = None

    snap_game = copy.deepcopy(game, memo)
    for plr in snap_game.players.values():
        plr._controller_stack = []

    return GameSnapshot(
        _game=snap_game,
        rng_state=random.getstate(),
        game_rng_state=game.rng.getstate(),
        controller_specs=controller_specs,
    )


###############################################################################
def restore_snapshot(game: "Game", snap: GameSnapshot, restore_rng: bool = True) -> None:
    """Rewind `game` in-place to `snap` without mutating `snap`.

    - Preserves object identity of `game` and every existing `Player` instance.
    - Preserves live `PlayerController`s, controller stacks, and `rich.Console` handles.
    - Rebinds all circular references (`card.game`, `card.player`, `play_area.game`,
      `player.coins.on_add`) to the live `game` and `Player` objects.
    - Can be called repeatedly with the same `snap` for repeated undo / tree search.
    """
    live_players = dict(game.players)
    live_rng_state = game.rng.getstate()
    live_ui: dict[str, tuple[PlayerController, list[PlayerController], Any]] = {
        uid: (
            plr.controller,
            list(getattr(plr, "_controller_stack", [])),
            getattr(plr, "console", None),
        )
        for uid, plr in live_players.items()
    }

    memo: dict[int, Any] = {
        id(snap._game): game,
        id(snap._game.card_mapping): game.card_mapping,
    }
    for uid, snap_plr in snap._game.players.items():
        if uid in live_players:
            memo[id(snap_plr)] = live_players[uid]

    # Step 1: Deep-copy Game's attributes (all Cards/Piles/etc. are cloned once into memo,
    # while references to Game and existing Players resolve to the live objects).
    new_game_dict = copy.deepcopy(snap._game.__dict__, memo)

    # Step 2: Deep-copy each Player's attributes into the live Player objects,
    # sharing all Card instances already placed into `memo` in Step 1.
    for uid, snap_plr in snap._game.players.items():
        if uid in live_players:
            live_plr = live_players[uid]
            new_plr_dict = copy.deepcopy(snap_plr.__dict__, memo)
            live_plr.__dict__.clear()
            live_plr.__dict__.update(new_plr_dict)
            ctrl, stack, console = live_ui[uid]
            live_plr.controller = ctrl
            live_plr._controller_stack = stack
            if console is not None:
                live_plr.console = console
        else:
            cloned_plr = new_game_dict["players"][uid]
            spec = snap.controller_specs.get(uid, (TextController, {}))
            ctrl = _make_controller(spec)
            cloned_plr.controller = ctrl
            cloned_plr._controller_stack = []
            if hasattr(ctrl, "console"):
                cloned_plr.console = ctrl.console

    # Step 3: Commit restored Game attributes in-place.
    game.__dict__.clear()
    game.__dict__.update(new_game_dict)

    # Step 4: Restore RNG state for deterministic replay.
    if restore_rng:
        random.setstate(snap.rng_state)
        game.rng.setstate(snap.game_rng_state)
    else:
        game.rng.setstate(live_rng_state)


###############################################################################
def clone_game(game: "Game", quiet: bool = True) -> "Game":
    """Return an independent deep copy of `game` with fresh controllers.

    Mutations to the returned clone never affect `game`, and vice versa.
    """
    snap = capture_snapshot(game)
    memo: dict[int, Any] = {id(snap._game.card_mapping): game.card_mapping}
    cloned = copy.deepcopy(snap._game, memo)
    cloned.rng = random.Random()
    cloned.rng.setstate(snap.game_rng_state)
    cloned.quiet = quiet
    for uid, plr in cloned.players.items():
        spec = snap.controller_specs.get(uid, (TextController, {}))
        ctrl = _make_controller(spec)
        plr.controller = ctrl
        plr._controller_stack = []
        plr.quiet = quiet
        if hasattr(ctrl, "console"):
            plr.console = ctrl.console
    return cloned


# EOF
