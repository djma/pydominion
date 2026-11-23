"""Comprehensive rules conformance test suite verifying fixes from the Dominion Complete Rules audit."""

import unittest

from dominion import Card, Game, Phase, Piles, Prompt
from dominion.Card import CardType
from dominion.Cost import Cost


###############################################################################
class TestSetupAndPilesAudit(unittest.TestCase):
    """Domain 1: Setup, Supply vs Non-Supply, Pile Sizes, Special Cards."""

    def test_curses_always_in_supply_scaled_by_players(self) -> None:
        for num_players, expected in ((2, 10), (3, 20), (4, 30)):
            g = Game.TestGame(numplayers=num_players, initcards=["Moat"])
            g.start_game()
            self.assertIn("Curse", g.card_piles)
            self.assertTrue(g.card_piles["Curse"].in_supply)
            self.assertEqual(len(g.card_piles["Curse"]), expected)

    def test_victory_kingdom_pile_sizes_8_vs_12(self) -> None:
        g2 = Game.TestGame(numplayers=2, initcards=["Gardens", "Castles"])
        g2.start_game()
        self.assertEqual(len(g2.card_piles["Gardens"]), 8)
        self.assertEqual(len(g2.card_piles["Castles"]), 8)

        g3 = Game.TestGame(numplayers=3, initcards=["Gardens", "Castles"])
        g3.start_game()
        self.assertEqual(len(g3.card_piles["Gardens"]), 12)
        self.assertEqual(len(g3.card_piles["Castles"]), 12)

    def test_shelters_estate_supply_count(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Moat", "Shelters"])
        g.start_game()
        # 2-player Estate Supply pile has 8 Estates (starting Estates replaced by Shelters)
        self.assertEqual(len(g.card_piles["Estate"]), 8)
        self.assertNotIn("Shelters", g.card_piles)
        for plr in g.player_list():
            cards = plr.get_cards()
            self.assertEqual(cards.get("Estate", 0), 0)
            self.assertEqual(cards.get("Hovel", 0), 1)
            self.assertEqual(cards.get("Necropolis", 0), 1)
            self.assertEqual(cards.get("Overgrown Estate", 0), 1)

    def test_potion_16_and_willowisp_12(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Alchemist", "Pixie"])
        g.start_game()
        self.assertEqual(len(g.card_piles["Potion"]), 16)
        self.assertIn("Will-o'-Wisp", g.card_piles)
        self.assertEqual(len(g.card_piles["Will-o'-Wisp"]), 12)
        self.assertFalse(g.card_piles["Will-o'-Wisp"].in_supply)

    def test_heirlooms_dealt_and_no_pile_left(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Pixie"])
        g.start_game()
        self.assertNotIn("Goat", g.card_piles)
        for plr in g.player_list():
            cards = plr.get_cards()
            self.assertEqual(cards.get("Goat", 0), 1)
            self.assertEqual(cards.get("Copper", 0), 6)

    def test_ruins_purchasable_and_zombies_only_in_trash(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Cultist", "Necromancer"])
        g.start_game()
        self.assertIn("Ruins", g.card_piles)
        self.assertTrue(g.card_piles["Ruins"].in_supply)
        self.assertTrue(g.card_piles["Ruins"].get_top_card().purchasable)
        for z in ("Zombie Apprentice", "Zombie Mason", "Zombie Spy"):
            self.assertNotIn(z, g.card_piles)
            self.assertIn(z, g.trash_pile)

    def test_travellers_loaded_per_line(self) -> None:
        g_page = Game.TestGame(numplayers=2, initcards=["Page"], badcards=["Peasant"])
        g_page.start_game()
        self.assertIn("Champion", g_page.card_piles)
        self.assertFalse(g_page.card_piles["Champion"].in_supply)
        self.assertNotIn("Teacher", g_page.card_piles)

        g_peasant = Game.TestGame(numplayers=2, initcards=["Peasant"], badcards=["Page"])
        g_peasant.start_game()
        self.assertIn("Teacher", g_peasant.card_piles)
        self.assertFalse(g_peasant.card_piles["Teacher"].in_supply)
        self.assertNotIn("Champion", g_peasant.card_piles)

    def test_charlatan_makes_curse_a_treasure_globally(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Charlatan"])
        g.start_game()
        plr = g.player_list()[0]
        curse = g.card_piles["Curse"].get_top_card()
        self.assertTrue(curse.isCurse())
        self.assertTrue(curse.isTreasure())
        c = g.get_card_from_pile("Curse")
        plr.add_card(c, Piles.HAND)
        plr.play_card(c)
        self.assertEqual(plr.coins.get(), 1)


###############################################################################
class TestGameEndAndSupplyAudit(unittest.TestCase):
    """Domain 2: Game End Conditions & Supply vs Non-Supply."""

    def test_colony_empty_ends_game_even_if_province_remaining(self) -> None:
        g = Game.TestGame(numplayers=2, prosperity=True, initcards=["Moat"])
        g.start_game()
        self.assertFalse(g.isGameOver())
        while not g.card_piles["Colony"].is_empty():
            c = g.get_card_from_pile("Colony")
            g.trash_pile.add(c)
            c.location = Piles.TRASH
        self.assertFalse(g.card_piles["Province"].is_empty())
        self.assertTrue(g.isGameOver())

    def test_empty_supply_threshold_3_for_4p_and_4_for_5p(self) -> None:
        g5 = Game.TestGame(numplayers=5, initcards=["Moat", "Village", "Smithy", "Market"])
        g5.start_game()
        for name in ("Moat", "Village", "Smithy"):
            while not g5.card_piles[name].is_empty():
                g5.get_card_from_pile(name)
        self.assertFalse(g5.isGameOver())  # 5 players requires 4 empty Supply piles
        while not g5.card_piles["Market"].is_empty():
            g5.get_card_from_pile("Market")
        self.assertTrue(g5.isGameOver())

    def test_non_supply_empty_piles_do_not_end_game(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Page", "Moat", "Village"])
        g.start_game()
        # Empty 2 Supply piles + 3 non-Supply Traveller piles -> still NOT game over
        for name in ("Moat", "Village", "Treasure Hunter", "Warrior", "Hero"):
            while not g.card_piles[name].is_empty():
                g.get_card_from_pile(name)
        self.assertFalse(g.isGameOver())


###############################################################################
class TestCostAndMultiTypeAudit(unittest.TestCase):
    """Domain 3: Multi-dimensional Cost, Potions, Cost Modifiers, Multi-type selection."""

    def test_cost_partial_ordering(self) -> None:
        c5 = Cost(coins=5)
        c2p = Cost(coins=2, potions=1)
        c3 = Cost(coins=3)
        c3p = Cost(coins=3, potions=1)
        c8d = Cost(coins=0, debt=8)
        self.assertFalse(c2p.is_up_to(c5))  # Potion cannot be <= coin-only
        self.assertFalse(c8d.is_up_to(c5))  # Debt cannot be <= coin-only
        self.assertTrue(c3.is_up_to(c3p))   # $3 <= $3+P
        self.assertTrue(c2p.is_up_to(c3p))  # $2+P <= $3+P

    def test_buy_card_spends_potions(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Alchemist"])
        g.start_game()
        plr = g.player_list()[0]
        plr.coins.set(3)
        plr.potions.set(1)
        plr.buys.set(1)
        bought = plr.buy_card("Alchemist")
        self.assertIsNotNone(bought)
        self.assertEqual(plr.potions.get(), 0)
        self.assertEqual(plr.coins.get(), 0)

    def test_events_ignore_bridge_cost_reduction(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Bridge"], events=["Mission"])
        g.start_game()
        plr = g.player_list()[0]
        bridge = g.get_card_from_pile("Bridge")
        plr.add_card(bridge, Piles.HAND)
        plr.play_card(bridge)
        # Mission costs printed $4; player has $3 (Bridge gives +$1 -> set to $3 total)
        plr.coins.set(3)
        plr.buys.set(1)
        event_opts, _ = Prompt.event_selection(plr, 0)
        self.assertEqual(len(event_opts), 1)
        self.assertEqual(event_opts[0]["selector"], "-")  # Unaffordable: printed $4 ignores Bridge
        self.assertIn("4 Coins", event_opts[0]["details"])
        plr.coins.set(4)
        event_opts_ok, _ = Prompt.event_selection(plr, 0)
        self.assertNotEqual(event_opts_ok[0]["selector"], "-")

    def test_throne_room_bridge_stacks_and_inventor_order(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Throne Room", "Bridge", "Inventor", "Gardens"])
        g.start_game()
        plr = g.player_list()[0]
        tr = g.get_card_from_pile("Throne Room")
        br = g.get_card_from_pile("Bridge")
        plr.piles[Piles.HAND].set()
        plr.add_card(tr, Piles.HAND)
        plr.add_card(br, Piles.HAND)
        plr.test_input = ["Bridge"]
        plr.play_card(tr)
        # Throne Room + Bridge => -$2 total
        self.assertEqual(plr.card_cost(g.card_piles["Gold"].get_top_card()), 4)

        # Inventor applies -$1 AFTER gaining a card costing up to $4
        g2 = Game.TestGame(numplayers=1, initcards=["Inventor", "Market", "Gardens"])
        g2.start_game()
        p2 = g2.player_list()[0]
        inv = g2.get_card_from_pile("Inventor")
        p2.add_card(inv, Piles.HAND)
        # Market costs $5 -> must NOT be affordable during Inventor's own gain!
        self.assertEqual(p2.card_cost(g2.card_piles["Market"].get_top_card()), 5)
        p2.test_input = ["Get Gardens"]
        p2.play_card(inv)
        self.assertEqual(p2.card_cost(g2.card_piles["Market"].get_top_card()), 4)

    def test_bridge_troll_cost_reduction_persists_next_turn(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Bridge Troll"])
        g.start_game()
        plr, _ = g.player_list()
        bt = g.get_card_from_pile("Bridge Troll")
        plr.add_card(bt, Piles.HAND)
        plr.play_card(bt)
        gold = g.card_piles["Gold"].get_top_card()
        self.assertEqual(plr.card_cost(gold), 5)
        plr.end_turn()
        plr.start_turn()
        self.assertEqual(plr.card_cost(gold), 5)  # Still -$1 on next turn!
        plr.end_turn()
        plr.start_turn()
        self.assertEqual(plr.card_cost(gold), 6)  # Expired on turn after

    def test_peddler_buy_phase_only_and_counts_duration_actions(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Peddler", "Wharf"])
        g.start_game()
        plr = g.player_list()[0]
        wharf = g.get_card_from_pile("Wharf")
        plr.add_card(wharf, Piles.DURATION)
        peddler = g.card_piles["Peddler"].get_top_card()
        plr.phase = Phase.ACTION
        self.assertEqual(plr.card_cost(peddler), 8)
        plr.phase = Phase.BUY
        self.assertEqual(plr.card_cost(peddler), 6)

    def test_select_by_type_multi_type_cards(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Nobles"])
        g.start_game()
        plr = g.player_list()[0]
        nobles = g.get_card_from_pile("Nobles")
        self.assertTrue(plr.select_by_type(nobles, {CardType.ACTION: True}))
        self.assertTrue(plr.select_by_type(nobles, {CardType.VICTORY: True}))
        self.assertFalse(plr.select_by_type(nobles, {CardType.ACTION: False}))


###############################################################################
class TestTurnFlowAndTokensAudit(unittest.TestCase):
    """Domain 4: Buy Phase Parts, Tokens, Batch Discard/Trash, Exchange, Exile, 2nd-ed Shuffle."""

    def test_two_part_buy_phase_blocks_treasures_after_buy(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Moat"])
        g.start_game()
        plr = g.player_list()[0]
        plr.piles[Piles.HAND].set("Copper", "Silver")
        plr.phase = Phase.BUY
        plr.buy_phase_part = 1
        opts_part1 = Prompt.spendable_selection(plr)
        self.assertTrue(len(opts_part1) > 0)
        plr.buys.set(2)
        plr.coins.set(3)
        plr.buy_card("Silver")
        self.assertEqual(plr.buy_phase_part, 2)
        opts_part2 = Prompt.spendable_selection(plr)
        self.assertEqual(opts_part2, [])

    def test_minus_one_card_and_coin_tokens_intercept_all_sources(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Moat"])
        g.start_game()
        plr = g.player_list()[0]
        plr.piles[Piles.HAND].set()
        plr.piles[Piles.DECK].set("Copper", "Silver", "Gold")
        plr.card_token = True
        drawn = plr.pickup_cards(2)
        self.assertEqual(len(drawn), 1)
        self.assertFalse(plr.card_token)

        plr.coin_token = True
        plr.coins.add(3)
        self.assertEqual(plr.coins.get(), 2)
        self.assertFalse(plr.coin_token)

    def test_exchange_does_not_count_as_gain(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Page"])
        g.start_game()
        plr = g.player_list()[0]
        page = g.get_card_from_pile("Page")
        plr.add_card(page, Piles.PLAYED)
        plr.stats["gained"] = []
        plr.test_input = ["Replace"]
        plr.replace_traveller(page, "Treasure Hunter")
        self.assertIn("Treasure Hunter", plr.piles[Piles.DISCARD])
        self.assertEqual(plr.stats["gained"], [])

    def test_second_edition_shuffle_goes_under_remaining_deck(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Moat"])
        g.start_game()
        plr = g.player_list()[0]
        plr.piles[Piles.HAND].set()
        plr.piles[Piles.DECK].set("Province")  # Top of deck is Province
        plr.piles[Piles.DISCARD].set("Copper", "Copper", "Copper")
        drawn = plr.pickup_cards(2)
        # Province MUST be the first card drawn before any reshuffled Copper!
        self.assertEqual(drawn[0].name, "Province")
        self.assertEqual(drawn[1].name, "Copper")


###############################################################################
class TestPlayThroneDurationReactionAudit(unittest.TestCase):
    """Domain 5 & 6: Unified Play, Throne-Room + Duration, Stop-Moving, Reactions, Ways."""

    def test_throne_room_duration_stays_in_play_and_triggers_twice(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Throne Room", "Wharf"])
        g.start_game()
        plr = g.player_list()[0]
        tr = g.get_card_from_pile("Throne Room")
        wharf = g.get_card_from_pile("Wharf")
        plr.piles[Piles.HAND].set()
        plr.add_card(tr, Piles.HAND)
        plr.add_card(wharf, Piles.HAND)
        plr.test_input = ["Wharf"]
        plr.play_card(tr)
        # During turn played, both are in PLAYED
        self.assertIn("Throne Room", plr.piles[Piles.PLAYED])
        self.assertIn("Wharf", plr.piles[Piles.PLAYED])
        self.assertEqual(plr.buys.get(), 3)  # 1 base + 2 from Wharf played twice
        # Clean-up: BOTH stay in play (moved to DURATION)
        plr.end_turn()
        self.assertIn("Throne Room", plr.piles[Piles.DURATION])
        self.assertIn("Wharf", plr.piles[Piles.DURATION])
        # Start of next turn: Wharf triggers TWICE (+4 Cards, +2 Buys)
        plr.piles[Piles.DECK].set("Copper", "Copper", "Copper", "Copper", "Copper")
        plr.start_turn()
        self.assertEqual(plr.buys.get(), 3)  # 1 base + 2 from 2x Wharf duration
        self.assertEqual(len(plr.piles[Piles.HAND]), 5 + 4)

    def test_stop_moving_throne_room_horse(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Throne Room", "Sleigh"])
        g.start_game()
        plr = g.player_list()[0]
        tr = g.get_card_from_pile("Throne Room")
        horse = g.get_card_from_pile("Horse")
        plr.piles[Piles.HAND].set()
        plr.add_card(tr, Piles.HAND)
        plr.add_card(horse, Piles.HAND)
        horses_before = len(g.card_piles["Horse"])
        plr.test_input = ["Horse"]
        plr.play_card(tr)
        # Horse returned to its pile on play 1, stayed there (Stop-Moving!), and still gave 2x +2 Cards +1 Action
        self.assertEqual(len(g.card_piles["Horse"]), horses_before + 1)
        self.assertNotIn("Horse", plr.piles[Piles.PLAYED])
        self.assertEqual(plr.actions.get(), 2)  # 1 - 1 (TR) + 1 + 1

    def test_lighthouse_defends_only_in_play_and_watchtower_only_in_hand(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Lighthouse", "Watchtower", "Militia"])
        g.start_game()
        p0, p1 = g.player_list()
        lh = g.get_card_from_pile("Lighthouse")
        p1.piles[Piles.HAND].set()
        p1.add_card(lh, Piles.HAND)
        self.assertFalse(p1.has_defense(p0))  # In hand -> does NOT defend!
        p1.move_card(lh, Piles.DURATION)
        self.assertTrue(p1.has_defense(p0))   # In play -> DOES defend!

        # Watchtower in PLAYED must NOT trigger on gain
        wt = g.get_card_from_pile("Watchtower")
        p0.add_card(wt, Piles.PLAYED)
        p0.gain_card("Copper")  # No test_input needed because Watchtower should not prompt
        self.assertIn("Copper", p0.piles[Piles.DISCARD])

    def test_tactician_does_not_stay_in_play_if_nothing_discarded(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Tactician"])
        g.start_game()
        plr = g.player_list()[0]
        tac = g.get_card_from_pile("Tactician")
        plr.piles[Piles.HAND].set()
        plr.add_card(tac, Piles.HAND)
        plr.test_input = ["Keep"]
        plr.play_card(tac)
        plr.end_turn()
        self.assertNotIn("Tactician", plr.piles[Piles.DURATION])

    def test_way_skips_duration_stay(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Wharf"], ways=["Way of the Pig"])
        g.start_game()
        plr = g.player_list()[0]
        wharf = g.get_card_from_pile("Wharf")
        plr.add_card(wharf, Piles.HAND)
        plr.play_card(wharf, way=g.ways["Way of the Pig"])
        self.assertEqual(plr.buys.get(), 1)  # Wharf's +1 Buy did not happen
        self.assertEqual(plr.actions.get(), 1)  # Way of the Pig gave +1 Action (net 0)
        plr.end_turn()
        self.assertNotIn("Wharf", plr.piles[Piles.DURATION])


###############################################################################
class TestTurnQueueScoringBoonsHexesAudit(unittest.TestCase):
    """Domain 7 & 8: TurnQueue, Consecutive Cap, Scoring Tie-Breaker, Set-Aside Scoring, Boons, Hexes."""

    def test_turn_queue_2023_two_consecutive_turns_cap(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Outpost"])
        g.start_game()
        p0, p1 = g.player_list()
        # Turn 1: p0 plays regular turn and schedules extra turn
        e1 = g.turn_queue.pop_next_turn()
        self.assertEqual(e1.player_uuid, p0.uuid)
        self.assertTrue(g.schedule_extra_turn(p0, source="Outpost"))
        g.turn_queue.finish_current_turn()

        # Turn 2: p0 takes extra turn; trying to schedule a 3rd turn in a row MUST fail!
        e2 = g.turn_queue.pop_next_turn()
        self.assertEqual(e2.player_uuid, p0.uuid)
        self.assertTrue(e2.is_extra)
        self.assertFalse(g.schedule_extra_turn(p0, source="Outpost"))
        g.turn_queue.finish_current_turn()

        # Turn 3: p1 takes regular turn
        e3 = g.turn_queue.pop_next_turn()
        self.assertEqual(e3.player_uuid, p1.uuid)

    def test_determine_winners_tiebreaker_fewest_regular_turns(self) -> None:
        g = Game.TestGame(numplayers=2, initcards=["Moat"])
        g.start_game()
        p0, p1 = g.player_list()
        # Both start with 3 VP; p0 took 5 regular turns, p1 took 4 -> p1 wins tie-break!
        p0.turn_number = 5
        p1.turn_number = 4
        self.assertEqual(g.determine_winners(), [p1])
        # Equal turns -> shared victory
        p1.turn_number = 5
        self.assertEqual(set(g.determine_winners()), {p0, p1})

    def test_set_aside_victory_card_is_scored(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Haven"])
        g.start_game()
        plr = g.player_list()[0]
        prov = g.get_card_from_pile("Province")
        haven = g.get_card_from_pile("Haven")
        plr.piles[Piles.HAND].set()
        plr.add_card(prov, Piles.HAND)
        plr.add_card(haven, Piles.HAND)
        plr.test_input = ["select province"]
        plr.play_card(haven)
        # Province is set aside under Haven, NOT in hand/deck/discard, yet MUST count for 6 VP!
        self.assertNotIn("Province", plr.piles[Piles.HAND])
        self.assertEqual(plr.get_score_details().get("Province"), 6)

    def test_cleanup_boons_does_not_prematurely_reshuffle(self) -> None:
        g = Game.TestGame(numplayers=1, initcards=["Druid"])
        g.start_game()
        initial_len = len(g.boons)
        boon = g.receive_boon()
        self.assertIsNotNone(boon)
        boon.retain_boon = True
        g.discard_boon(boon)
        self.assertEqual(len(g.retained_boons), 1)
        g.cleanup_boons()
        self.assertEqual(len(g.retained_boons), 0)
        self.assertEqual(len(g.discarded_boons), 1)
        self.assertEqual(len(g.boons), initial_len - 1)  # NOT reshuffled back!

    def test_distribute_hex_draws_single_shared_hex(self) -> None:
        g = Game.TestGame(numplayers=3, initcards=["Skulk"])
        g.start_game()
        p0, p1, p2 = g.player_list()
        # Put Delusion on top of Hex deck
        for h in list(g.hexes):
            if h.name == "Delusion":
                g.hexes.remove(h)
                g.hexes.append(h)
                break
        hexes_before = len(g.hexes)
        g.distribute_hex(p0, [p1, p2])
        # Only ONE Hex drawn and discarded once, yet BOTH victims received Delusion!
        self.assertEqual(len(g.hexes), hexes_before - 1)
        self.assertEqual(len(g.discarded_hexes), 1)
        self.assertTrue(p1.has_state("Deluded"))
        self.assertTrue(p2.has_state("Deluded"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
