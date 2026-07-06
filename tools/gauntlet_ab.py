"""Controlled A/B gauntlet: planner ON vs planner OFF, identical decks/opponents.

Runs our agent piloting the submission deck against the full opponent panel
(each piloting pool decks), once with the engine planner enabled and once with
pure rules, on the *same* deck pool and seeds so the comparison is controlled.
This is the keep-if-better gate for the planner.

Writes to tools/_gauntlet_out.txt.
Run:  python tools/gauntlet_ab.py [--decks N] [--games G] [--think S]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
SRC = os.path.join(ROOT, "src")
for p in (SRC, SUB):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gauntlet_out.txt")


def log(msg):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")
    print(msg, flush=True)


def read_deck_file(path):
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", type=int, default=5)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--think", type=float, default=0.10)
    ap.add_argument("--our-deck", default=os.path.join(SUB, "deck.csv"))
    ap.add_argument("--mirror-deck", default=None,
                    help="if set, BOTH sides pilot this deck (skill test) vs each "
                         "baseline; measures play skill on that archetype")
    args = ap.parse_args()

    open(OUT, "w").close()
    mode = "mirror" if args.mirror_deck else "field"
    log(f"A/B gauntlet [{mode}]: decks={args.decks} games={args.games} "
        f"seed={args.seed} think={args.think}s "
        f"deck={os.path.basename(args.mirror_deck or args.our_deck)}")

    from ptcg_agent.deckgen import DeckPool
    from ptcg_agent.evaluate import _load_opponents, _ModuleOpponent
    from ptcg_agent.harness import run_match_lowlevel
    from ptcg_agent.selfplay import WeightedAgent

    panel = _load_opponents(None)

    if args.mirror_deck:
        mirror_deck = read_deck_file(args.mirror_deck)
        log(f"panel: {', '.join(l for l, _ in panel)} | MIRROR both sides pilot deck")

        def run_arm(use_planner):
            results = {}
            for label, mod in panel:
                ours = WeightedAgent(mirror_deck, None, use_planner=use_planner,
                                     planner_think_s=args.think)
                opp = _ModuleOpponent(mod, mirror_deck)
                res = run_match_lowlevel(ours, opp, mirror_deck, mirror_deck,
                                         args.games, swap_each=True)
                results[label] = (res.wins_a, res.games,
                                  res.errors_a + res.errors_b)
            return results
    else:
        pool = DeckPool()
        decks = pool.sample(args.decks, seed=args.seed)
        our_deck = read_deck_file(args.our_deck)
        log(f"panel: {', '.join(l for l, _ in panel)} | pool {len(decks)} decks")

        def run_arm(use_planner):
            results = {}
            for label, mod in panel:
                wins = total = errs = 0
                for opp_deck in decks:
                    ours = WeightedAgent(our_deck, None, use_planner=use_planner,
                                         planner_think_s=args.think)
                    opp = _ModuleOpponent(mod, opp_deck)
                    res = run_match_lowlevel(ours, opp, our_deck, opp_deck,
                                             args.games, swap_each=True)
                    wins += res.wins_a
                    total += res.games
                    errs += res.errors_a + res.errors_b
                results[label] = (wins, total, errs)
            return results

    log("\n--- arm B: planner OFF (pure rules) ---")
    t0 = time.perf_counter()
    off = run_arm(False)
    log(f"(off arm wall {time.perf_counter()-t0:.1f}s)")

    log("\n--- arm A: planner ON ---")
    t0 = time.perf_counter()
    on = run_arm(True)
    log(f"(on arm wall {time.perf_counter()-t0:.1f}s)")

    log("\n=== A/B RESULT (winrate: planner_ON vs planner_OFF) ===")
    go_w = go_t = ff_w = ff_t = 0
    for label, _ in panel:
        ow, ot, oe = on[label]
        fw, ft, fe = off[label]
        owr = ow / ot if ot else 0.0
        fwr = fw / ft if ft else 0.0
        delta = owr - fwr
        log(f"  vs {label:8s}: ON {owr:.3f} ({ow}/{ot})  OFF {fwr:.3f} ({fw}/{ft})  "
            f"delta {delta:+.3f}  errs ON/OFF {oe}/{fe}")
        go_w += ow
        go_t += ot
        ff_w += fw
        ff_t += ft
    o_all = go_w / go_t if go_t else 0.0
    f_all = ff_w / ff_t if ff_t else 0.0
    log(f"  {'OVERALL':11s}: ON {o_all:.3f}  OFF {f_all:.3f}  delta {o_all-f_all:+.3f}")
    log("DONE")


if __name__ == "__main__":
    main()
