"""Targeted deck A/B: our PLANNER agent piloting deck_v2 vs deck.csv, each played
against a FIXED wall opponent (strong_agent piloting a high-HP Mega deck). This
isolates whether the deck change helps break the walls we lose to on the ladder
(the mirror/H2H can't test this — an aggro deck has no walls).

Run: python tools/_deck_vs_wall.py --games 80 --opp-deck decks_archive/deck_DA.csv
"""
from __future__ import annotations
import argparse, math, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)


def read_deck(path):
    return [int(l) for l in open(path) if l.strip().lstrip("-").isdigit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--think", type=float, default=0.10)
    ap.add_argument("--opp-deck", default=os.path.join(ROOT, "decks_archive", "deck_DA.csv"))
    ap.add_argument("--opp", default="strong_agent")
    args = ap.parse_args()

    from ptcg_agent.evaluate import _load_agent_module, _ModuleOpponent
    from ptcg_agent.harness import run_match_lowlevel
    from ptcg_agent.selfplay import WeightedAgent

    opp_deck = read_deck(args.opp_deck)
    opp_dir = os.path.join(ROOT, "baselines", args.opp)
    opp_mod = _load_agent_module(opp_dir)

    decks = {
        "v2 (deck_v2.csv)": read_deck(os.path.join(ROOT, "submission", "deck_v2.csv")),
        "v1 (deck.csv)":    read_deck(os.path.join(ROOT, "submission", "deck.csv")),
    }
    print(f"opponent = {args.opp} piloting {os.path.basename(args.opp_deck)} "
          f"| games={args.games} think={args.think}s")
    for label, deck in decks.items():
        ours = WeightedAgent(deck, None, use_planner=True, planner_think_s=args.think)
        opp = _ModuleOpponent(opp_mod, opp_deck)
        t0 = time.perf_counter()
        res = run_match_lowlevel(ours, opp, deck, opp_deck, args.games, swap_each=True)
        wall = time.perf_counter() - t0
        n = res.games
        wr = res.wins_a / n if n else 0.0
        se = math.sqrt(wr * (1 - wr) / n) if n else 0.0
        z = (wr - 0.5) / se if se > 0 else 0.0
        print(f"  {label:22s} win {res.wins_a:3d}/{n} = {wr:.3f} +-{se:.3f}  "
              f"z_vs_0.5={z:+.2f}  (draws {res.draws}, {wall/max(1,n):.2f}s/g)")


if __name__ == "__main__":
    main()
