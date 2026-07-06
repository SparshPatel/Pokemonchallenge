"""Probe: confirm the rewritten PIMC actually runs real search rollouts.

Reaches a mid-game MAIN decision, builds a PIMCSearcher, and checks that
determinizations are sampled and ``_evaluate`` returns real (non-None) values
through ``search_begin`` -> rollout -> terminal scoring.

Run:  python tools/probe_pimc.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

import cg.game as game  # noqa: E402
from agent.adapter import extract_select  # noqa: E402
from agent.cards import CardStats  # noqa: E402
from agent.pimc import PIMCSearcher  # noqa: E402


def read_deck():
    ids = []
    with open(os.path.join(SUB, "deck.csv"), encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def legal(sel):
    n = len(sel.get("option") or [])
    mn = sel.get("minCount") or 0
    k = max(1, mn) if sel.get("type") == 0 else mn
    return list(range(min(max(k, 0), n)))


def main():
    deck = read_deck()
    obs, _ = game.battle_start(deck, deck)
    target = None
    for _ in range(60):
        cur, sel = obs.get("current"), obs.get("select")
        if sel is None:
            obs = game.battle_select(list(range(60)))
            continue
        if isinstance(cur, dict) and cur.get("result", -1) >= 0:
            break
        if (isinstance(cur, dict) and sel.get("type") == 0
                and cur.get("turn", 0) >= 1 and obs.get("search_begin_input")):
            target = obs
            break
        obs = game.battle_select(legal(sel))

    if target is None:
        print("no decision point")
        game.battle_finish()
        return

    obs = target
    cards = CardStats.load(os.path.join(SUB, "cards.json"))
    searcher = PIMCSearcher(cards, your_deck_ids=deck, n_determinizations=3,
                            max_think_s=3.0, seed=1)
    print("available:", searcher.available())

    select = extract_select(obs)
    searcher._update_belief(obs)
    dets = searcher._build_determinizations(obs)
    print("determinizations:", len(dets), "first is None:", dets[0] is None)

    actions = searcher._candidate_actions(select)[:5]
    deadline = time.monotonic() + 5.0
    real = 0
    for a in actions:
        for d in dets:
            v = searcher._evaluate(obs, a, d, deadline)
            if v is not None:
                real += 1
                print(f"  action={a} value={v:+.3f}")
                break
    print(f"non-None evaluations: {real}/{len(actions)}")

    t0 = time.monotonic()
    choice = searcher.choose(obs, select, time.monotonic() + 3.0)
    print(f"choose -> {choice} in {time.monotonic()-t0:.2f}s")

    # Deep rollout (NOT runtime-viable) to confirm terminal scoring works.
    import cg.api as api  # noqa: E402
    obs_cls = api.to_observation_class(obs)
    args = searcher._search_args(obs, dets[0])
    if args is not None:
        yd, yp, od, op, oh, oa = args
        ss = api.search_begin(obs_cls, yd, yp, od, op, oh, oa, False)
        ss = api.search_step(ss.searchId, actions[0])
        yi = obs["current"]["yourIndex"]
        steps = 0
        val = None
        import dataclasses
        from agent import rules
        while steps < 2000:
            o = ss.observation
            cur = o.current
            if cur is None:
                break
            if isinstance(cur.result, int) and cur.result >= 0:
                val = 1.0 if cur.result == yi else (0.0 if cur.result == 2 else -1.0)
                break
            if o.select is None:
                break
            od_ = dataclasses.asdict(o)
            sl = extract_select(od_)
            ch = rules.choose(od_, sl, searcher.gamedata) if sl and sl.options else [0]
            ss = api.search_step(ss.searchId, ch)
            steps += 1
        api.search_end()
        print(f"deep rollout: steps={steps} terminal_value={val}")
    game.battle_finish()


if __name__ == "__main__":
    main()
