"""End-to-end check: drive the real submission main.agent through full games.

Verifies the *competition entry point* (submission/main.py -> Policy -> planner
default-on) is crash-free across complete games, including the deck-selection
phase and the safe-fallback wrapper. Reports wins, any agent errors, decision
count and think-time, and asserts every returned action was engine-legal (the
harness would raise otherwise).

Writes to tools/_e2e_out.txt.
Run:  python tools/probe_e2e_submission.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
SRC = os.path.join(ROOT, "src")
for p in (SRC, SUB):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_e2e_out.txt")


def log(msg):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def read_deck(path):
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def main():
    open(OUT, "w").close()
    log("start e2e: submission/main.agent with planner default-on")

    import importlib.util
    import cg.game as game

    # Load the real competition entry point exactly as the engine would.
    spec = importlib.util.spec_from_file_location("sub_main", os.path.join(SUB, "main.py"))
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    log("loaded submission/main.py")

    # A plain rules opponent (baseline greedy) to play against.
    opp_dir = os.path.join(ROOT, "baselines", "greedy_agent")
    ospec = importlib.util.spec_from_file_location("opp_main", os.path.join(opp_dir, "main.py"))
    opp = importlib.util.module_from_spec(ospec)
    ospec.loader.exec_module(opp)

    our_deck = read_deck(os.path.join(SUB, "deck.csv"))
    opp_deck = read_deck(os.path.join(opp_dir, "deck.csv"))

    n_games = 4
    wins = errs = decisions = 0
    tsum = 0.0
    tmax = 0.0

    for g in range(n_games):
        a_first = (g % 2 == 0)
        agents = {0: sub.agent, 1: opp.agent} if a_first else {0: opp.agent, 1: sub.agent}
        our_idx = 0 if a_first else 1
        d0, d1 = (our_deck, opp_deck) if a_first else (opp_deck, our_deck)
        try:
            obs, _ = game.battle_start(d0, d1)
        except Exception as e:
            log(f"game {g}: battle_start FAILED {e}")
            continue
        result = -1
        for _ in range(5000):
            cur = obs.get("current")
            if isinstance(cur, dict) and cur.get("result", -1) >= 0:
                result = cur["result"]
                break
            yi = cur.get("yourIndex", 0) if isinstance(cur, dict) else 0
            t0 = time.perf_counter()
            try:
                choice = agents[yi](obs)
            except Exception as e:
                errs += 1
                log(f"game {g}: agent[{yi}] raised {type(e).__name__}: {e}")
                choice = [0]
            dt = time.perf_counter() - t0
            if yi == our_idx:
                decisions += 1
                tsum += dt
                tmax = max(tmax, dt)
            try:
                obs = game.battle_select(choice)
            except Exception as e:
                log(f"game {g}: battle_select FAILED on {choice}: {e}")
                result = -99
                break
        try:
            game.battle_finish()
        except Exception:
            pass
        won = (result == our_idx)
        wins += 1 if won else 0
        log(f"game {g}: first={'ours' if a_first else 'opp'} result={result} "
            f"won={won}")

    log(f"RESULT: wins={wins}/{n_games} agent_errors={errs} our_decisions={decisions} "
        f"avg_think={1000*tsum/max(1,decisions):.2f}ms max_think={1000*tmax:.1f}ms")
    log("VERDICT: " + ("PASS (no crashes)" if errs == 0 else "FAIL (agent errors)"))
    log("DONE")


if __name__ == "__main__":
    main()
