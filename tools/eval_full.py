"""Discriminating evaluation with the FULL runtime Policy (planner + supervisor).

The stock ``ptcg_agent.evaluate`` pits our *rules-only* ``WeightedAgent`` against
the weak baselines, which saturates near 1.0 and cannot measure play-quality
changes. This harness instead loads the real ``submission/main.py`` ``agent``
(the exact competition entry point, planner + supervisor + rules) for our side
and any baseline ``main.py`` (or our own agent, for a self-play mirror) for the
opponent, then plays paired games across a generated deck pool.

Usage::

    PYTHONPATH=_sample_sub:src:submission python -m tools.eval_full \
        --opp strong --decks 8 --games 4 --our-deck submission/deck.csv
    PYTHONPATH=_sample_sub:src:submission python -m tools.eval_full \
        --opp self --decks 8 --games 4          # planner-vs-planner mirror
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "submission"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg_agent.deckgen import DeckPool  # noqa: E402
from ptcg_agent.harness import _load_agent_module, run_match_lowlevel  # noqa: E402

_OPP_DIRS = {
    "random": "baselines/random_agent",
    "greedy": "baselines/greedy_agent",
    "tempo": "baselines/tempo_agent",
    "strong": "baselines/strong_agent",
    "pivot": "baselines/pivot_wall",
    "self": "submission",
}


def _read_deck(path: str) -> list[int]:
    return [int(l) for l in open(path) if l.strip().lstrip("-").isdigit()]


class _Mod:
    def __init__(self, module, deck):
        self._m = module
        self.deck = list(deck)

    def agent(self, obs):
        return self._m.agent(obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="strong", choices=list(_OPP_DIRS))
    ap.add_argument("--decks", type=int, default=8)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--our-deck", default=os.path.join(_ROOT, "submission", "deck.csv"))
    ap.add_argument("--same-deck", action="store_true",
                    help="opponent pilots OUR deck too (isolates skill, not deck)")
    ap.add_argument("--opp-deck", default=None,
                    help="fixed deck.csv the opponent pilots (e.g. a wall deck)")
    args = ap.parse_args()

    our_deck = _read_deck(args.our_deck)
    ours = _load_agent_module(os.path.join(_ROOT, "submission"))
    opp_mod = _load_agent_module(os.path.join(_ROOT, _OPP_DIRS[args.opp]))

    if args.opp_deck:
        pool = [_read_deck(args.opp_deck)] * args.decks
    elif args.same_deck:
        pool = [our_deck] * args.decks
    else:
        pool = DeckPool().sample(args.decks, seed=args.seed)

    t0 = time.monotonic()
    wins = total = errs = 0
    for opp_deck in pool:
        a = _Mod(ours, our_deck)
        b = _Mod(opp_mod, opp_deck)
        res = run_match_lowlevel(a, b, our_deck, opp_deck, args.games, swap_each=True)
        wins += res.wins_a
        total += res.games
        errs += res.errors_a + res.errors_b
    dt = time.monotonic() - t0
    wr = wins / total if total else 0.0
    print(f"opp={args.opp} decks={args.decks} games/deck={args.games} "
          f"-> {wins}/{total} = {wr:.3f}  (errors {errs}, {dt:.1f}s, {total/dt:.1f} g/s)")


if __name__ == "__main__":
    main()
