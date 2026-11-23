#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Scepter"""
import unittest

from dominion import Game, Card, Piles, Player


###############################################################################
class Card_Scepter(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.TREASURE, Card.CardType.COMMAND]
        self.base = Card.CardExpansion.RENAISSANCE
        self.desc = "Choose one: +$2; or replay a non-Command Action card you played this turn that's still in play."
        self.name = "Scepter"
        self.cost = 5

    def special(self, game: Game.Game, player: Player.Player) -> None:
        acts = [
            c
            for c in player.played_this_turn
            if c.location == Piles.PLAYED and c in player.piles[Piles.PLAYED] and c.isAction() and not c.isCommand()
        ]
        if acts:
            get_coin = player.plr_choose_options("Choose one? ", ("2 Coin", True), ("Replay an action card", False))
        else:
            get_coin = True
            player.output("No suitable cards - gaining coin")
        if get_coin:
            player.coins.add(2)
        else:
            if cards := player.card_sel(cardsrc=acts):
                player.play_card(cards[0], cost_action=False, move_card=False, throne_card=self)


###############################################################################
class Test_Scepter(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Scepter", "Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Scepter")
        self.plr.add_card(self.card, Piles.HAND)

    def test_play_coin(self) -> None:
        self.plr.test_input = ["2 Coin"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 2)

    def test_play_replay(self) -> None:
        self.plr.piles[Piles.PLAYED].set("Moat")
        self.plr.played_this_turn.append(self.plr.piles[Piles.PLAYED]["Moat"])
        self.plr.test_input = ["Replay", "Moat"]
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 0)
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 5 + 2)

    def test_previous_turn_action_is_not_eligible(self):
        self.plr.piles[Piles.PLAYED].set("Moat")
        self.plr.play_card(self.card)
        self.assertEqual(self.plr.coins.get(), 2)
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 5)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
