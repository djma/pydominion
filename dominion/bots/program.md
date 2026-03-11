# dominion bot autoresearch

This is an experiment to have the LLM recursively write and improve a Dominion bot.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these files for full context:
   - `dominion/bots/BigMoney.py` — the control bot (opponent). Do not modify.
   - `dominion/bots/matchup.py` — the evaluation harness. Do not modify.
   - `dominion/bots/__init__.py` — bot auto-discovery. Do not modify.
   - `dominion/Player.py` — base Player class. Read for context (don't modify).
   - `dominion/Card.py` — Card class. Read for context (don't modify).
   - `dominion/__init__.py` — enums (Action, Piles, Phase). Read for context.
4. **Read the kingdom cards**: Read the card source files for every card in the kingdom to understand their effects. The cards are in `dominion/cards/` as `Card_<Name>.py` (e.g. `Card_Chapel.py`).
5. **Create ExperimentBot.py**: Create a new `dominion/bots/ExperimentBot.py` which inherits Player.py. This is the file you will iterate on.
6. **Verify the matchup runs**: Run the matchup command once to confirm everything works:
   ```
   uv run python -m dominion.bots.matchup --control BigMoney --experiment ExperimentBot \
       --kingdom Patron Scavenger "Wandering Minstrel" Sculptor "Hunting Grounds" Ducat Lackeys Vagrant "Market Square" Feodum --project Guildhall
   ```
7. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is a matchup of your `ExperimentBot` against the current control bot on the fixed kingdom. The matchup runs adaptively (no `--games` flag), stopping when the 95% CI converges. Each run typically takes 1-3 minutes. The control starts as `BigMoney` but gets replaced when ExperimentBot is promoted (see "Control promotion" below).

Run command (substitute `<CURRENT_CONTROL>` with the active control bot name):

```
uv run python -m dominion.bots.matchup --control <CURRENT_CONTROL> --experiment ExperimentBot \
    --kingdom Patron Scavenger "Wandering Minstrel" Sculptor "Hunting Grounds" Ducat Lackeys Vagrant "Market Square" Feodum --project Guildhall
```

**What you CAN do:**

- Modify `dominion/bots/ExperimentBot.py` — this is the ONLY file you edit. Everything is fair game: buy priorities, action play order, card selection heuristics, trashing/discarding strategy, special-case logic for specific kingdom cards, card synergy awareness, deck composition tracking, game-phase adaptation, etc.

**What you CANNOT do:**

- Modify any other file in the repo. The game engine, card definitions, matchup harness, and other bots are all read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness. The matchup framework is the ground truth metric.

**The goal is simple: maximize win % against the current control bot.** The initial control is BigMoney, a pure treasure-buying strategy that ignores kingdom cards — it just buys the most expensive treasure/victory card it can afford. As ExperimentBot improves and gets promoted, the control becomes a copy of the previous best ExperimentBot, creating a self-play ladder of increasing difficulty.

## Output format

The matchup prints a summary like this:

```
Games: 42  (W 30 / L 10 / D 2)
Win %:            73.8%
Draw %:           4.8%
Win % as 1st plr: 71.4%
Avg turns/game:   14.2
Coins/turn (exp): 1.4, 2.8, 3.5, 4.2, ...
Coins/turn (ctl): 1.2, 2.5, 3.8, 4.5, ...
Actions/turn (exp): 0.0, 0.5, 1.2, 1.8, ...
VP/turn (exp):    3.0, 3.0, 3.0, 6.0, ...
```

You can extract the key metric:

```
grep "^Win %:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	win_pct	games	status	control	description
```

1. git commit hash (short, 7 chars)
2. win_pct achieved (e.g. 73.8) — use 0.0 for crashes
3. total games played (e.g. 42) — use 0 for crashes
4. status: `keep`, `discard`, `crash`, or `promote`
5. control bot name used for this run (e.g. `BigMoney`, `ControlBot_v1`)
6. short text description of what this experiment tried

Example:

```
commit	win_pct	games	status	control	description
b2c3d4e	68.5	44	keep	BigMoney	add Chapel trashing of Coppers and Estates
c3d4e5f	48.1	40	discard	BigMoney	overly aggressive trashing
d4e5f6g	0.0	0	crash	BigMoney	syntax error in card_sel
e5f6g7h	78.2	36	promote	BigMoney	add Witch priority + Village/Smithy engine
f6g7h8i	55.1	42	keep	ControlBot_v1	tweak Village buy priority vs promoted control
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar11`).

Track the current control bot name in a variable (starts as `BigMoney`) and a version counter (starts at 0).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Think about strategy: read the card files if needed, analyze what the current control does poorly, think about Dominion strategy (engine building, trashing, attacks, greening timing)
3. Tune `ExperimentBot.py` with an experimental idea by directly hacking the code
4. git commit
5. Run the experiment: `uv run python -m dominion.bots.matchup --control <CURRENT_CONTROL> --experiment ExperimentBot --kingdom Patron Scavenger "Wandering Minstrel" Sculptor "Hunting Grounds" Ducat Lackeys Vagrant "Market Square" Feodum --project Guildhall > run.log 2>&1`
6. Read out the results: `grep "^Win %:\|^Games:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If win_pct improved (higher), you "advance" the branch, keeping the git commit
10. If win_pct is equal or worse, you git reset back to where you started
11. **Promotion check** (see below): if win_pct > 66%, promote the current ExperimentBot to become the new control

The idea is that you are a completely autonomous Dominion strategist trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck, you can try more radical changes but avoid rewinding to much earlier states.

## Control promotion

When `ExperimentBot` achieves **> 66% win rate** against the current control, it has clearly surpassed it. At that point:

1. Log the result in `results.tsv` with status `promote`
2. Increment the version counter (e.g. 0 → 1)
3. Copy `ExperimentBot.py` to `ControlBot_v<N>.py` (e.g. `ControlBot_v1.py`), renaming the class inside to match (e.g. `ControlBot_v1`)
4. git commit this new control file with message like "promote ExperimentBot to ControlBot_v1"
5. Update your current control variable to the new class name (e.g. `ControlBot_v1`)
6. Now `ExperimentBot` continues to be the file you edit, but all future matchups run against the newly promoted control
7. The win_pct tracking resets — you're now trying to beat the stronger opponent

This creates an iterative self-play ladder: each time the bot gets good enough (>66%), it graduates to become its own opponent, and the experiment continues trying to beat it. The bar keeps rising.

**Important**: The promotion threshold is **strictly greater than 66%**. A result of exactly 66.0% does not trigger promotion — only 66.1% or higher does.

**After promotion**: The experiment bot starts from whatever state it was in when promoted (it's already the same code as the new control). You'll need to find new improvements to beat yourself. This is where creativity matters most — the easy wins against BigMoney are gone, and you need subtler strategic edges.

**Timeout**: Each matchup should take 1-3 minutes typically. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import, wrong method signature), fix it and re-run. If the idea itself is fundamentally broken (e.g. trying to access game state that doesn't exist), just skip it, log "crash", and move on.

**Statistical noise**: Win % has natural variance. A change from 72.3% to 73.1% might just be noise. Pay attention to the 95% CI — if the CIs overlap heavily, the change might not be real. Prefer changes that show clear, meaningful improvement (several percentage points). When in doubt, you can re-run the same commit to see if results are stable.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working _indefinitely_ until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the card files for new angles, try combining previous near-misses, try more radical strategic changes, study what the timeseries data tells you about where your bot is weak. The loop runs until the human interrupts you, period.
