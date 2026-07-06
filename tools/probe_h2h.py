"""Head-to-head agent comparison — the low-variance measurement tool.

Comparing two agents by their win rates against a *third* baseline is dominated
by engine-shuffle noise (the engine does not seed its shuffles, so N=10 carries
+-0.16 variance). A *direct* head-to-head — agent A vs agent B piloting the same
deck, playing each other — is a paired comparison with far less noise and is the
right way to decide "is variant A better than variant B".

Configs (``--a`` / ``--b``):
  rules        pure rule policy
  planner      planner, static end-of-turn eval   (PTCG_PLANNER_OPP=0)
  planner_opp  planner + one opponent-response ply (PTCG_PLANNER_OPP=1)

Usage:
  python tools/probe_h2h.py --a planner --b rules --deck submission/deck.csv --games 60
  python tools/probe_h2h.py --a planner_opp --b planner --deck data/sim_sample/deck.csv --games 60

Writes to tools/_h2h_out.txt.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
SRC = os.path.join(ROOT, "src")
for p in (SRC, SUB):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_h2h_out.txt")


def log(msg):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")
    print(msg, flush=True)


def read_deck(path):
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def make_agent(kind, deck, gd, think, beam=4, depth=6, k=2, eval_path=None, max_nodes=300, use_vnet=False):
    from agent import rules
    from agent.adapter import extract_select, is_deck_phase
    from agent.planner import TurnPlanner

    eval_weights = None
    if eval_path:
        import json
        with open(eval_path, "r", encoding="utf-8") as fh:
            eval_weights = json.load(fh)

    planner = None
    if kind in ("planner", "planner_opp"):
        planner = TurnPlanner(
            None, gamedata=gd, your_deck_ids=deck,
            max_think_s=think,
            beam_width=beam, max_depth=depth, n_determinizations=k,
            opp_response=(kind == "planner_opp"),
            eval_weights=eval_weights,
            max_nodes=max_nodes,
            use_value_net=use_vnet,
        )

    def agent(obs):
        try:
            if is_deck_phase(obs):
                return list(deck)
            sel = extract_select(obs)
            if sel is None:
                return list(deck)
            rules.set_weights({})
            if planner is not None and sel.select_type == 0:
                try:
                    ch = planner.choose(obs, sel, time.monotonic() + 8.0)
                    if ch and _valid(ch, sel):
                        return ch
                except Exception:
                    pass
            ch = rules.choose(obs, sel, gd)
            if _valid(ch, sel):
                return ch
            n = len(sel.options)
            return list(range(max(0, min(sel.min_count, n))))
        except Exception:
            return [0]

    return agent


def _valid(choice, sel):
    if not isinstance(choice, list) or len(choice) != len(set(choice)):
        return False
    n = len(sel.options)
    if not (sel.min_count <= len(choice) <= sel.max_count):
        return False
    return all(isinstance(i, int) and 0 <= i < n for i in choice)


class _Ag:
    def __init__(self, fn, deck):
        self.agent = fn
        self.deck = list(deck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="planner")
    ap.add_argument("--b", default="rules")
    ap.add_argument("--deck", default=os.path.join(SUB, "deck.csv"))
    ap.add_argument("--a-deck", default=None, help="deck override for arm A (default: --deck)")
    ap.add_argument("--b-deck", default=None, help="deck override for arm B (default: --deck)")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--think", type=float, default=0.12)
    ap.add_argument("--a-think", type=float, default=None, help="think time override for arm A")
    ap.add_argument("--b-think", type=float, default=None, help="think time override for arm B")
    ap.add_argument("--a-beam", type=int, default=4)
    ap.add_argument("--a-depth", type=int, default=6)
    ap.add_argument("--a-k", type=int, default=2)
    ap.add_argument("--b-beam", type=int, default=4)
    ap.add_argument("--b-depth", type=int, default=6)
    ap.add_argument("--b-k", type=int, default=2)
    ap.add_argument("--a-eval", default=None, help="JSON eval-weights file for arm A")
    ap.add_argument("--b-eval", default=None, help="JSON eval-weights file for arm B")
    ap.add_argument("--a-nodes", type=int, default=300, help="max_nodes for arm A")
    ap.add_argument("--b-nodes", type=int, default=300, help="max_nodes for arm B")
    ap.add_argument("--a-vnet", action="store_true", help="use learned value net for arm A")
    ap.add_argument("--b-vnet", action="store_true", help="use learned value net for arm B")
    args = ap.parse_args()

    open(OUT, "w").close()
    from agent.gamedata import GameData
    from ptcg_agent.harness import run_match_lowlevel

    gd = GameData.load()
    deck = read_deck(args.deck)
    a_deck = read_deck(args.a_deck) if args.a_deck else deck
    b_deck = read_deck(args.b_deck) if args.b_deck else deck
    a_think = args.a_think if args.a_think is not None else args.think
    b_think = args.b_think if args.b_think is not None else args.think
    log(f"H2H: A={args.a}(b{args.a_beam}/d{args.a_depth}/k{args.a_k}/t{a_think}) vs "
        f"B={args.b}(b{args.b_beam}/d{args.b_depth}/k{args.b_k}/t{b_think}) | "
        f"deck_a={os.path.basename(args.a_deck or args.deck)} "
        f"deck_b={os.path.basename(args.b_deck or args.deck)} | games={args.games}")

    a_fn = make_agent(args.a, a_deck, gd, a_think, args.a_beam, args.a_depth, args.a_k, args.a_eval, args.a_nodes, args.a_vnet)
    b_fn = make_agent(args.b, b_deck, gd, b_think, args.b_beam, args.b_depth, args.b_k, args.b_eval, args.b_nodes, args.b_vnet)
    A = _Ag(a_fn, a_deck)
    B = _Ag(b_fn, b_deck)

    t0 = time.perf_counter()
    res = run_match_lowlevel(A, B, a_deck, b_deck, args.games, swap_each=True)
    wall = time.perf_counter() - t0

    n = res.games
    wr = res.wins_a / n if n else 0.0
    se = math.sqrt(wr * (1 - wr) / n) if n else 0.0
    # Two-sided z vs 0.5 for "A better than B".
    z = (wr - 0.5) / se if se > 0 else 0.0
    log(f"A wins {res.wins_a}/{n} = {wr:.3f} +-{se:.3f}  (draws {res.draws})  z_vs_0.5={z:+.2f}")
    log(f"wall={wall:.1f}s ({wall/max(1,n):.2f}s/game)")
    log("DONE")


if __name__ == "__main__":
    main()
