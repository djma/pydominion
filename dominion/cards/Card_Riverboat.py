#!/usr/bin/env python
"""https://wiki.dominionstrategy.com/index.php/Riverboat"""
import random
import unittest
from unittest.mock import patch
from typing import Any

from dominion import Game, Card, Player, OptionKeys, game_setup, Keys, Piles

RIVERBOAT = "riverboat"


###############################################################################
class Card_Riverboat(Card.Card):
    def __init__(self) -> None:
        Card.Card.__init__(self)
        self.cardtype = [Card.CardType.ACTION, Card.CardType.DURATION]
        self.base = Card.CardExpansion.RISING_SUN
        self.name = "Riverboat"
        self.cost = 3
        self.desc = "At the start of your next turn, play the set aside card, leaving it there. Setup: Set aside an unused non-Duration Action card costing $5."

    def dynamic_description(self, player: "Player.Player") -> str:
        card_name = player.game.specials[RIVERBOAT]
        desc = player.game.card_instances[card_name].description(player)
        return f"At the start of your next turn, play {card_name}, leaving it there. ({card_name}: {desc})"

    def duration(self, game: Game.Game, player: Player.Player) -> dict[OptionKeys, Any]:
        """Play the set aside card, leaving it there."""
        player.output(f"Playing {game.specials[RIVERBOAT]} from Riverboat")
        player.card_benefits(game.card_instances[game.specials[RIVERBOAT]])
        # Try and handle cards that move themselves when played
        if game.card_instances[game.specials[RIVERBOAT]].location:
            player.remove_card(game.card_instances[game.specials[RIVERBOAT]])
            game.card_instances[game.specials[RIVERBOAT]].location = None
        return {}

    def setup(self, game: Game.Game) -> None:
        """Set aside an unused non-Duration Action card costing $5."""
        if RIVERBOAT in game.specials:
            return
        card = self.pick_card(game)
        self.setup_riverboat(game, card)

    def pick_card(self, game: Game.Game) -> str:
        """Pick suitable card to set aside"""
        actions = []
        for klass in game.card_mapping["Card"].values():
            card = klass()

            if card.name in game.card_piles.keys():
                continue
            if hasattr(card, "cardpile_setup") or not card.desc.strip():
                continue
            if not card.insupply or not card.purchasable:
                continue
            if card.name in game.config.init_cards[Keys.BAD_CARDS]:
                continue
            if card.isDuration() or not card.isAction():
                continue
            if card.cost != 5 or card.potcost or card.debtcost:
                continue
            actions.append(card.name)
        set_aside = random.choice(actions)
        return set_aside

    def setup_riverboat(self, game: Game.Game, set_aside: str):
        """Separate function so we can call it with named card for testing"""
        game.specials[RIVERBOAT] = set_aside
        card = game.card_mapping["Card"][set_aside]()
        card.game = game
        card.location = Piles.SPECIAL
        game.card_instances[set_aside] = card
        game.cards[card.uuid] = card
        game_setup.check_card_requirement(game, game.card_instances[set_aside])
        game.output(f"Using {set_aside} as the card for Riverboat")


###############################################################################
class Test_Riverboat(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Game.TestGame(numplayers=1, initcards=["Riverboat"], badcards=["Moat"])
        self.g.start_game()
        self.plr = self.g.player_list()[0]
        self.card = self.g.get_card_from_pile("Riverboat")
        self.card.setup_riverboat(self.g, "Moat")  # type: ignore

    def test_pick_card(self) -> None:
        card_name = self.card.pick_card(self.g)  # type: ignore
        card = self.g.card_mapping["Card"][card_name]()
        self.assertEqual(card.cost, 5)
        self.assertFalse(card.isDuration())
        self.assertTrue(card.isAction())

    def test_no_placeholders(self) -> None:
        """All candidates must be playable single cards."""
        with patch("random.choice", side_effect=lambda cards: cards[0]) as choose:
            self.card.pick_card(self.g)
        candidates = choose.call_args.args[0]
        self.assertNotIn("Knights", candidates)
        for name in candidates:
            card = self.g.card_mapping["Card"][name]()
            self.assertTrue(card.desc.strip())
            self.assertFalse(hasattr(card, "cardpile_setup"))

    def test_baker_setup(self) -> None:
        """The set-aside card's setup runs once after players exist."""
        game = Game.TestGame(numplayers=2, num_stacks=1, initcards=["Riverboat"])
        with patch.object(Card_Riverboat, "pick_card", return_value="Baker"):
            game.start_game()
        baker = game.card_instances["Baker"]
        self.assertIs(baker.game, game)
        self.assertIs(game.cards[baker.uuid], baker)
        self.assertEqual(baker.location, Piles.SPECIAL)
        self.assertEqual([player.coffers.get() for player in game.player_list()], [1, 1])
        game_setup.card_setup(game)
        self.assertEqual([player.coffers.get() for player in game.player_list()], [1, 1])

    def test_proportional_checks_use_different_cards(self) -> None:
        """Shelters and Prosperity draw different Kingdom randomizers."""
        game = Game.TestGame(num_stacks=2, initcards=["Moat", "Village"], prosperity=None, shelters=True)
        with patch("dominion.game_setup._proportional_card", wraps=game_setup._proportional_card) as choose:
            game.start_game()
        self.assertEqual(choose.call_count, 2)
        used = choose.call_args_list[0].args[1]
        self.assertEqual(len(used), 2)
        self.assertEqual(len(set(used)), 2)

    def test_tournament_prizes(self) -> None:
        """Tournament uses one of each Prize outside the Supply."""
        game = Game.TestGame(num_stacks=1, initcards=["Tournament"], oldcards=True)
        game.start_game()
        prizes = [pile for _, pile in game.get_all_piles() if not pile.in_supply]
        self.assertEqual(len(prizes), 5)
        self.assertEqual([len(pile) for pile in prizes], [1] * 5)

    def test_play(self) -> None:
        """Play card"""
        self.plr.play_card(self.card)
        self.plr.end_turn()
        self.plr.start_turn()
        self.assertEqual(len(self.plr.piles[Piles.HAND]), 5 + 2)  # 2 for Moat


###############################################################################
if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# EOF
