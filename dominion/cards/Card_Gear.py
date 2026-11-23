#!/usr/bin/env python

import unittest

from dominion import Game, Piles, Player, Card

GEAR = "gear"


###############################################################################
class Card_Gear(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.ADVENTURE
        self.desc = "+2 Cards; Set aside up to 2 cards from your hand. Pick up next turn"
        self.name = "Gear"
        self.cards = 2
        self.cost = 3
        self._play_batches: list[list[Card.Card]] = []

    def _mat_key(self) -> str:
        return f"gear:{self.uuid}"

    def special(self, game: Game.Game, player: Player.Player) -> None:
        """Set aside up to 2 cards from your hand face down..."""
        area = player.get_set_aside_area(self._mat_key())
        player.specials[GEAR] = area
        cards = player.card_sel(
            num=2,
            cardsrc=Piles.HAND,
            prompt="Set aside up to 2 cards from your hand to be put back next turn",
            verbs=("Set", "Unset"),
        )
        for card in cards:
            player.remove_card(card)
            area.add(card)
            card.location = Piles.SPECIAL
        if cards:
            self._play_batches.append(list(cards))
        self.stay_in_play = bool(self._play_batches)

    def duration(self, game: Game.Game, player: Player.Player) -> None:
        """... At the start of your next turn, put them into your hand"""
        area = player.get_set_aside_area(self._mat_key())
        batch = self._play_batches.pop(0) if self._play_batches else []
        for card in batch:
            if card not in area:
                continue
            player.output(f"Pulling {card} reserved by Gear")
            area.remove(card)
            player.add_card(card, Piles.HAND)
        self.stay_in_play = bool(self._play_batches)


###############################################################################
class Test_Gear(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Gear", "Throne Room"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Gear")

    def test_play_card(self) -> None:
        """Play a gear"""
        self.plr.piles[Piles.HAND].set("Duchy", "Silver", "Gold")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["set silver", "set gold", "finish"]
        self.plr.play_card(self.card)
        try:
            self.assertEqual(self.plr.piles[Piles.HAND].size(), 1 + 2)  # Duchy + 2 picked up
            self.assertIn("Duchy", self.plr.piles[Piles.HAND])
            self.assertEqual(self.plr.piles[Piles.PLAYED].size(), 1)
            self.plr.end_turn()
            self.assertEqual(self.plr.piles[Piles.DURATION].size(), 1)
            self.plr.start_turn()
            self.assertEqual(self.plr.piles[Piles.DURATION].size(), 0)
            self.assertEqual(self.plr.piles[Piles.PLAYED].size(), 1)
            self.assertEqual(self.plr.piles[Piles.PLAYED][-1].name, "Gear")
            self.assertIn("Silver", self.plr.piles[Piles.HAND])
            self.assertIn("Gold", self.plr.piles[Piles.HAND])
        except AssertionError:  # pragma: no cover
            self.g.print_state()
            raise

    def test_lost_throne_returns_only_first_batch(self):
        throne = self.g.get_card_from_pile("Throne Room")
        self.plr.piles[Piles.HAND].set("Gold", "Silver")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.add_card(throne, Piles.HAND)
        self.plr.test_input = ["Gear", "Set Gold", "Finish", "Set Silver", "Finish"]
        self.plr.play_card(throne)
        area = self.plr.get_set_aside_area(self.card._mat_key())
        self.assertEqual(len(area), 2)
        self.plr.move_card(throne, Piles.DISCARD)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertIn("Gold", self.plr.piles[Piles.HAND])
        self.assertIn("Silver", area)
        self.assertNotIn("Silver", self.plr.piles[Piles.HAND])
        self.assertIn(area[0], self.plr.all_cards())


    def test_empty_play_does_not_delay_later_batch(self):
        from unittest.mock import patch

        self.plr.piles[Piles.HAND].set("Gold")
        gold = self.plr.piles[Piles.HAND]["Gold"]
        with patch.object(self.plr, "card_sel", side_effect=[[], [gold]]):
            self.card.special(self.g, self.plr)
            self.card.special(self.g, self.plr)
        self.assertEqual(len(self.card._play_batches), 1)
        self.card.duration(self.g, self.plr)
        self.assertIn(gold, self.plr.piles[Piles.HAND])
        self.assertFalse(self.card.stay_in_play)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
