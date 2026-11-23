#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Procession"""
import unittest

from dominion import Game, Card, Piles, Player


###############################################################################
class Card_Procession(Card.Card):
    """Procession"""

    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = Card.CardType.ACTION
        self.base = Card.CardExpansion.DARKAGES
        self.desc = """You may play a non-Duration Action card from your
            hand twice. Trash it. Gain an Action
            card costing exactly 1 more than it."""
        self.name = "Procession"
        self.cost = 4

    def special(self, game: Game.Game, player: Player.Player) -> None:
        action_cards = [_ for _ in player.piles[Piles.HAND] if _.isAction() and not _.isDuration()]
        if not action_cards:
            player.output("No suitable action cards")
            return
        cards = player.card_sel(prompt="Select a card to play twice, then trash", cardsrc=action_cards)
        if not cards:
            return
        card = cards[0]
        player.play_card_multiple(card, times=2, source_card=self, trash_after=True)
        target_cost = player.effective_cost(card).with_coin_delta(1)
        candidates = [
            c for _, pile in game.get_supply_piles() if pile
            if (c := pile.get_top_card()) is not None
            and c.isAction() and player.effective_cost(c) == target_cost
        ]
        if candidates:
            player.plr_gain_card(target_cost.coins, cardsrc=candidates, force=True)


###############################################################################
class TestProcession(unittest.TestCase):
    """Test Procession"""

    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Procession", "Moat", "Hermit"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Procession")

    def test_play(self) -> None:
        """Play procession to trash moat and buy a witch"""
        self.plr.piles[Piles.HAND].set("Moat")
        self.plr.add_card(self.card, Piles.HAND)
        self.plr.test_input = ["Moat", "Hermit"]
        self.plr.play_card(self.card)
        self.assertIn("Moat", self.g.trash_pile)
        self.assertEqual(self.plr.piles[Piles.HAND].size(), 4)
        self.assertIn("Hermit", self.plr.piles[Piles.DISCARD])

    def test_preserves_potion_and_debt_costs(self):
        from unittest.mock import patch

        for component, value in (("potcost", True), ("debtcost", 2)):
            with self.subTest(component=component):
                target = self.g.get_card_from_pile("Moat")
                eligible = self.g.card_piles["Hermit"].get_top_card()
                setattr(target, component, value)
                setattr(eligible, component, value)
                self.plr.add_card(target, Piles.HAND)
                with patch.object(self.plr, "card_sel", return_value=[target]), patch.object(
                    self.plr, "play_card_multiple"
                ), patch.object(self.plr, "plr_gain_card") as gain:
                    self.card.special(self.g, self.plr)
                candidates = gain.call_args.kwargs["cardsrc"]
                self.assertIn(eligible, candidates)
                expected = self.plr.effective_cost(target).with_coin_delta(1)
                self.assertTrue(all(self.plr.effective_cost(c) == expected for c in candidates))
                setattr(eligible, component, False if component == "potcost" else 0)


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
