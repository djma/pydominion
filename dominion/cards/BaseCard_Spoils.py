#!/usr/bin/env python
""" https://wiki.dominionstrategy.com/index.php/Spoils"""
import unittest

from dominion import Game, Card, Piles, Player


###############################################################################
class Card_Spoils(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = Card.CardType.TREASURE
        self.base = Card.CardExpansion.DARKAGES
        self.desc = "+3 coin. When you play this, return it to the Spoils pile."
        self.basecard = True
        self.purchasable = False
        self.insupply = False
        self.name = "Spoils"
        self.cost = 0
        self.coin = 3
        self.numcards = 15

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """When you play this return it to the spoils pile"""
        if self.location != Piles.PLAYED or self not in player.piles[Piles.PLAYED]:
            return
        player.remove_card(self)
        game.card_piles["Spoils"].add_to_top(self)
        self.player = None
        self.location = Piles.CARDPILE


###############################################################################
class TestSpoils(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(quiet=True, numplayers=1, initcards=["Bandit Camp"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]

    def test_play(self) -> None:
        num_spoils = len(self.g.card_piles["Spoils"])
        spoils = self.g.get_card_from_pile("Spoils")
        self.plr.add_card(spoils, Piles.HAND)
        self.plr.play_card(spoils)
        self.assertEqual(self.plr.coins.get(), 3)
        self.assertTrue(self.plr.piles[Piles.PLAYED].is_empty())
        self.assertEqual(len(self.g.card_piles["Spoils"]), num_spoils)

    def test_replay_does_not_return_twice(self):
        spoils = self.g.get_card_from_pile("Spoils")
        self.plr.add_card(spoils, Piles.HAND)
        self.plr.play_card(spoils)
        self.plr.play_card(spoils, cost_action=False, move_card=False)
        self.assertEqual(self.plr.coins.get(), 6)
        self.assertEqual(sum(c is spoils for c in self.g.card_piles["Spoils"]), 1)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
