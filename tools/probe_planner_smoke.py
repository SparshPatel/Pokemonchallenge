"""Smoke test: exercise the TurnPlanner in real games and measure its effect.

Runs our planner-enabled agent vs the pure-rules agent on the submission deck
(mirror), instrumenting the planner to count invocations / hits / errors and the
per-decision think time, and reports the head-to-head win rate. Confirms the
planner is actually firing (not silently falling back) and whether it helps.

Writes to tools/_smoke_out.txt.

Run:  python tools/probe_planner_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
SRC = os.path.join(ROOT, "src")
for p in (SUB, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_out.txt")


def log(msg):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def read_deck():
    ids = []
    with open(os.path.join(SUB, "deck.csv"), encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def main():
    open(OUT, "w").close()
    log("start")
    from agent.gamedata import GameData
    from agent.adapter import extract_select, is_deck_phase
    from agent import rules
    from agent.planner import TurnPlanner
    from ptcg_agent.harness import run_match_lowlevel

    deck = read_deck()
    gd = GameData.load()
    log(f"gd ok={gd.ok}")

    stats = {"calls": 0, "hits": 0, "errs": 0, "t": 0.0, "tmax": 0.0}

    class PlannerAgent:
        def __init__(self, deck):
            self.deck = list(deck)
            self.pl = TurnPlanner(None, gamedata=gd, your_deck_ids=self.deck,
                                  max_think_s=0.15)
            log(f"planner available={self.pl.available()}")

        def agent(self, obs):
            try:
                if is_deck_phase(obs):
                    return list(self.deck)
                sel = extract_select(obs)
                if sel is None:
                    return list(self.deck)
                rules.set_weights({})
                if sel.select_type == 0:  # MAIN
                    stats["calls"] += 1
                    t0 = time.perf_counter()
                    try:
                        choice = self.pl.choose(obs, sel, time.monotonic() + 8.0)
                    except Exception as e:
                        stats["errs"] += 1
                        log(f"planner err: {type(e).__name__}: {e}")
                        choice = None
                    dt = time.perf_counter() - t0
                    stats["t"] += dt
                    stats["tmax"] = max(stats["tmax"], dt)
                    if choice and _valid(choice, sel):
                        stats["hits"] += 1
                        return choice
                return rules.choose(obs, sel, gd)
            except Exception:
                return [0]

    class RulesAgent:
        def __init__(self, deck):
            self.deck = list(deck)

        def agent(self, obs):
            try:
                if is_deck_phase(obs):
                    return list(self.deck)
                sel = extract_select(obs)
                if sel is None:
                    return list(self.deck)
                rules.set_weights({})
                return rules.choose(obs, sel, gd)
            except Exception:
                return [0]

    def _valid(choice, sel):
        if not isinstance(choice, list) or len(choice) != len(set(choice)):
            return False
        n = len(sel.options)
        if not (sel.min_count <= len(choice) <= sel.max_count):
            return False
        return all(isinstance(i, int) and 0 <= i < n for i in choice)

    n_games = 20
    a = PlannerAgent(deck)
    b = RulesAgent(deck)
    log(f"playing {n_games} games: planner(A) vs rules(B), mirror deck")
    t0 = time.perf_counter()
    res = run_match_lowlevel(a, b, deck, deck, n_games, swap_each=True)
    wall = time.perf_counter() - t0

    log(f"RESULT: A(planner) wins={res.wins_a} B(rules) wins={res.wins_b} "
        f"draws={res.draws} winrate_A={res.winrate_a:.3f}")
    log(f"planner: calls={stats['calls']} hits={stats['hits']} errs={stats['errs']} "
        f"avg_think={1000*stats['t']/max(1,stats['calls']):.2f}ms "
        f"max_think={1000*stats['tmax']:.1f}ms")
    log(f"wall={wall:.1f}s ({wall/n_games:.2f}s/game)")
    log("DONE")


if __name__ == "__main__":
    main()
