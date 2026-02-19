#!/usr/bin/env python3
"""Run a BigMoney bot vs a Randobot and print the spectator log."""

from dominion import Game


def main() -> None:
    g = Game.Game(numplayers=2, quiet=True, bot=True, randobot=1)
    g.start_game()

    turn = 0
    while not g.game_over:
        turn += 1
        g.turn()
        if turn > 400:
            break

    g.whoWon()

    print("=" * 60)
    print(f"SPECTATOR LOG  ({len(g.spectator_log)} entries, {turn} turns)")
    print("=" * 60)
    for line in g.spectator_log:
        print(line)


if __name__ == "__main__":
    main()
