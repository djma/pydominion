"""Dynamic bot discovery for the bots package."""

import importlib
import inspect
import pkgutil
from pathlib import Path

from dominion.Player import Player


def get_all_bots() -> dict[str, type[Player]]:
    """Return a mapping of class name -> class for every Player subclass in this package."""
    bots: dict[str, type[Player]] = {}
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module = importlib.import_module(f"dominion.bots.{module_info.name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Player) and obj is not Player:
                bots[name] = obj
    return bots
