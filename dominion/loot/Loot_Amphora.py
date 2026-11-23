#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Amphora"""
import unittest
from collections import Counter

from dominion import Loot, Card, Game, Piles, Player, OptionKeys


###############################################################################
class Loot_Amphora(Loot.Loot):
    """Amphora"""

    def __init__(self) -> None:
        Loot.Loot.__init__(self)
        self.cardtype = [Card.CardType.LOOT, Card.CardType.TREASURE, Card.CardType.DURATION]
        self.base = Card.CardExpansion.PLUNDER
        self.name = "Amphora"
        self.desc = "Either now or at the start of your next turn: +1 Buy and +$3."
        self.cost = 7
        self.pile = "Loot"
        self._pending = 0
        self.stay_in_play = False

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Choose when to receive the coins and Buy, once per play."""
        if player.plr_choose_options("Amphora: When?", ("Now", True), ("Next turn", False)):
            player.coins.add(3)
            player.buys.add(1)
        else:
            self._pending += 1
        self.stay_in_play = self._pending > 0

    def duration(self, game: Game.Game, player: Player.Player) -> dict[OptionKeys, str]:
        if self._pending:
            self._pending -= 1
            player.coins.add(3)
            player.buys.add(1)
        self.stay_in_play = self._pending > 0
        return {}


###############################################################################
class TestAmphora(unittest.TestCase):
    """Test Amphora."""

    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, traits=["Cursed"], initcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Loot", "Amphora")
        self.plr.add_card(self.card, Piles.HAND)

    def test_now(self) -> None:
        self.plr.test_input = ["Now"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)
        self.plr.end_turn()
        self.assertIn(self.card, self.plr.piles[Piles.DISCARD])

    def test_complete_loot_pile(self) -> None:
        """Pickaxe requires two copies of each of the fifteen Loot cards."""
        game = Game.Game(quiet=True, num_stacks=1, initcards=["Pickaxe"], prosperity=False, shelters=False)
        game.start_game()
        names = {
            "Amphora", "Doubloons", "Endless Chalice", "Figurehead", "Hammer",
            "Insignia", "Jewels", "Orb", "Prize Goat", "Puzzle Box", "Sextant",
            "Shield", "Spell Scroll", "Staff", "Sword",
        }
        pile = game.card_piles["Loot"]
        self.assertEqual(Counter(card.name for card in pile), Counter({name: 2 for name in names}))
        self.assertEqual(len(pile), 30)
        self.assertFalse(pile.in_supply)
        self.assertEqual(len({card.uuid for card in pile}), 30)
        for card in pile:
            self.assertIs(card.game, game)
            self.assertIs(game.cards[card.uuid], card)
            self.assertEqual(card.pile, "Loot")

    def test_next_turn(self) -> None:
        self.plr.test_input = ["Next turn"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 0)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)

    def test_replay_mixed_choices(self) -> None:
        self.plr.test_input = ["Next turn", "Now"]
        self.plr.play_card(self.card)
        self.plr.play_card(self.card, move_card=False)
        self.assertEqual(self.plr.coins.get(), 3)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertEqual(self.plr.buys.get(), 2)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
