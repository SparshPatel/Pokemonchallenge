"""Diagnose *how* our agent loses to the tough opponents (tempo, strong).

For each game we record, from our agent's perspective:

* outcome (win / loss / draw)
* whether we moved first or second
* battle length in engine selections (proxy for fast race vs grind)
* our remaining prizes and the opponent's remaining prizes at the terminal state
  (a loss with many of our prizes still up = we got blown out / raced;
   a loss with 1 prize left = a close game decided at the wire)

Run::

    $env:PYTHONPATH="src;submission"
    python -m tools.diagnose_losses --decks 16 --games 8 --seed 7 --opponents tempo strong

This is a *dev-only* instrument; it does not touch the shipped agent.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg_agent.deckgen import DeckPool  # noqa: E402
from ptcg_agent.evaluate import (  # noqa: E402
    _ModuleOpponent,
    _load_opponents,
    _our_agent,
    _read_deck_file,
)


def _prizes_remaining(player: dict) -> int | None:
    """Best-effort count of prizes still unclaimed for a player.

    ``prize`` is a list of Card|None. The engine keeps taken prizes out of the
    list (length shrinks) in some builds, or marks them; we count truthy-or-None
    entries as "still a prize card present". Returns None if unavailable.
    """
    if not isinstance(player, dict):
        return None
    prize = player.get("prize")
    if not isinstance(prize, list):
        return None
    # Each element is a Card dict (face-up to owner) or None (face-down). Both
    # represent an *unclaimed* prize slot still on the board.
    return len(prize)


def _play_and_record(game, ours, opp, our_deck, opp_deck, n_games, stats):
    """Mirror run_match_lowlevel but capture terminal diagnostics for `ours`."""
    for g in range(n_games):
        a_first = g % 2 == 0
        # ours is always agent_a; map engine index -> agent
        our_index = 0 if a_first else 1
        if a_first:
            agents = {0: ours, 1: opp}
            da, db = our_deck, opp_deck
        else:
            agents = {0: opp, 1: ours}
            da, db = opp_deck, our_deck

        try:
            obs, _ = game.battle_start(da, db)
        except Exception:
            continue

        turns = 0
        final_state = None
        for _ in range(5000):
            if not isinstance(obs, dict):
                break
            state = obs.get("current")
            yi = 0
            if isinstance(state, dict) and isinstance(state.get("yourIndex"), int):
                yi = state["yourIndex"]
                result = state.get("result")
                if isinstance(result, int) and result >= 0:
                    final_state = state
                    final_state["_result"] = result
                    break
            try:
                choice = agents[yi].agent(obs)
            except Exception:
                choice = [0]
            try:
                obs = game.battle_select(choice)
            except Exception:
                break
            turns += 1

        try:
            game.battle_finish()
        except Exception:
            pass

        key = "second" if not a_first else "first"
        if final_state is None:
            stats["unknown"] += 1
            continue

        result = final_state["_result"]
        players = final_state.get("players")
        our_prizes = opp_prizes = None
        if isinstance(players, list) and len(players) == 2:
            our_prizes = _prizes_remaining(players[our_index])
            opp_prizes = _prizes_remaining(players[1 - our_index])

        if result == our_index:
            outcome = "win"
        elif result in (0, 1):
            outcome = "loss"
        else:
            outcome = "draw"

        stats["games"] += 1
        stats[outcome] += 1
        stats[f"{outcome}_{key}"] += 1
        stats["turns_all"].append(turns)
        stats[f"turns_{outcome}"].append(turns)
        if outcome == "loss" and our_prizes is not None:
            stats["loss_our_prizes_left"].append(our_prizes)
        if outcome == "win" and opp_prizes is not None:
            stats["win_opp_prizes_left"].append(opp_prizes)


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decks", type=int, default=16)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opponents", nargs="*", default=["tempo", "strong"])
    ap.add_argument("--our-deck", default=None)
    args = ap.parse_args()

    import cg.game as game

    pool = DeckPool()
    decks = pool.sample(args.decks, seed=args.seed)
    panel = _load_opponents(args.opponents)
    our_path = args.our_deck or os.path.join(_ROOT, "submission", "deck.csv")
    our_deck = _read_deck_file(our_path)

    print(f"our deck: {our_path} ({len(our_deck)} cards) | "
          f"pool {len(decks)} decks (seed {args.seed}) | {args.games} games each")

    for label, mod in panel:
        stats = defaultdict(int)
        for k in ("turns_all", "turns_win", "turns_loss", "turns_draw",
                  "loss_our_prizes_left", "win_opp_prizes_left"):
            stats[k] = []
        for opp_deck in decks:
            ours = _our_agent(our_deck)
            opp = _ModuleOpponent(mod, opp_deck)
            _play_and_record(game, ours, opp, our_deck, opp_deck, args.games, stats)

        g = stats["games"] or 1
        print(f"\n=== vs {label} ===")
        print(f"  games {stats['games']} | win {stats['win']} "
              f"({stats['win']/g:.3f}) | loss {stats['loss']} | draw {stats['draw']}")
        print(f"  going first : win {stats['win_first']:3d} / loss {stats['loss_first']:3d}")
        print(f"  going second: win {stats['win_second']:3d} / loss {stats['loss_second']:3d}")
        print(f"  turns  avg(win) {_avg(stats['turns_win']):.0f}  "
              f"avg(loss) {_avg(stats['turns_loss']):.0f}")
        if stats["loss_our_prizes_left"]:
            lp = stats["loss_our_prizes_left"]
            print(f"  on LOSS, our prizes left: avg {_avg(lp):.2f}  "
                  f"min {min(lp)}  max {max(lp)}  "
                  f"(blowouts >=4 left: {sum(1 for x in lp if x >= 4)}/{len(lp)})")
        if stats["win_opp_prizes_left"]:
            wp = stats["win_opp_prizes_left"]
            print(f"  on WIN,  opp prizes left: avg {_avg(wp):.2f}")


if __name__ == "__main__":
    main()
