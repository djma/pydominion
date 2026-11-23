#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Spell_Scroll"""
import unittest
from unittest.mock import patch

from dominion import Loot, Card, Game, Piles, Player


###############################################################################
class Loot_SpellScroll(Loot.Loot):
    """Spell Scroll"""

    def __init__(self):
        Loot.Loot.__init__(self)
        self.cardtype = [
            Card.CardType.ACTION,
            Card.CardType.LOOT,
            Card.CardType.TREASURE,
        ]
        self.base = Card.CardExpansion.PLUNDER
        self.desc = "Trash this to gain a cheaper card. If it's an Action or Treasure, you may play it."
        self.name = "Spell Scroll"
        self.cost = 7
        self.pile = "Loot"

    def special(self, game: "Game.Game", player: "Player.Player") -> None:
        """Trash this to gain a cheaper card. If it's an Action or Treasure, you may play it."""
        if self.location not in (Piles.PLAYED, Piles.DURATION):
            return
        cost = player.effective_cost(self)
        player.trash_card(self)
        candidates = player.cards_matching_cost(lambda candidate: candidate.is_less_than(cost))
        if not candidates:
            return
        if gained_card := player.plr_gain_card(cost=cost.coins, cardsrc=candidates, force=True):
            if gained_card.isAction() or gained_card.isTreasure():
                if player.plr_choose_options(f"Play {gained_card}?", ("Play", True), ("Discard", False)):
                    player.play_card(gained_card, cost_action=False)


###############################################################################
class TestSpellScroll(unittest.TestCase):
    """Test Spell Scroll"""

    def setUp(self):
        self.g = Game.TestGame(quiet=True, numplayers=1, num_stacks=1, traits=["Cursed"], initcards=["Witch"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        # Remove all other cards from loot pile, so we know what we will draw
        mods = 1
        while mods > 0:
            mods = 0
            for loot in self.g.card_piles["Loot"]:
                if loot.name != "Spell Scroll":
                    self.g.card_piles["Loot"].remove(loot.name)
                    mods += 1

    def test_playing(self):
        """Test playing a spell scroll"""
        spell = self.g.get_card_from_pile("Loot", "Spell Scroll")
        self.plr.add_card(spell, Piles.HAND)
        self.plr.test_input = ["Get Gold", "Play"]
        self.plr.play_card(spell)
        self.assertIn("Spell Scroll", self.g.trash_pile)
        self.assertIn("Gold", self.plr.piles[Piles.PLAYED])
        self.assertEqual(self.plr.coins.get(), 3)

    def test_reduced_cost(self):
        """A cost reduction also reduces the gain limit."""
        spell = self.g.get_card_from_pile("Loot", "Spell Scroll")
        self.plr.add_card(spell, Piles.HAND)
        self.plr.turn_effects.add_cost_modifier("test", -2, card_filter="treasure")
        self.plr.test_input = ["Get Silver", "Discard"]
        with patch.object(self.plr, "plr_gain_card", wraps=self.plr.plr_gain_card) as gain:
            self.plr.play_card(spell)
        self.assertNotIn("Witch", [card.name for card in gain.call_args.kwargs["cardsrc"]])
        self.assertIn("Spell Scroll", self.g.trash_pile)
        self.assertIn("Silver", self.plr.piles[Piles.DISCARD])


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
