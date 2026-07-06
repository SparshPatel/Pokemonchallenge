"""Probe: verify the cabt search tree branching semantics.

The planner's beam search assumes a PERSISTENT search tree: from a parent
``searchId`` we can call ``search_step(parent_id, [a])`` and then
``search_step(parent_id, [b])`` to obtain two *sibling* states, exploring
alternative actions from the same node. This probe verifies that assumption.

It:
1. Advances a self-play game to a MAIN decision with >=2 options.
2. Opens a search, records the root searchId + options.
3. Steps option A (records child A id + resulting select).
4. Steps option B *from the same root id* (records child B id + select).
5. Re-steps option A again from root (should reproduce child A deterministically).

Writes findings to tools/_branch_out.txt.

Run:  python tools/probe_branch.py
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_branch_out.txt")


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


def legal_choice(sel):
    n = len(sel.get("option") or [])
    mn = sel.get("minCount") or 0
    k = max(1, mn) if sel.get("type") == 0 else mn
    k = min(max(k, 0), n)
    return list(range(k)) if k else []


def card_id(c):
    if isinstance(c, dict):
        return c.get("cardId", c.get("id"))
    return None


def visible_self(me):
    out = []
    for c in (me.get("hand") or []):
        v = card_id(c)
        if v is not None:
            out.append(v)
    for c in (me.get("discard") or []):
        v = card_id(c)
        if v is not None:
            out.append(v)
    for area in ("active", "bench"):
        for mon in (me.get(area) or []):
            if isinstance(mon, dict):
                if isinstance(mon.get("id"), int):
                    out.append(mon["id"])
                for c in (mon.get("energyCards") or []):
                    v = card_id(c)
                    if v is not None:
                        out.append(v)
    return out


def sel_summary(sel):
    if sel is None:
        return "None"
    opts = sel.get("option") or []
    return f"type={sel.get('type')} ctx={sel.get('context')} nopt={len(opts)} " \
           f"types={[o.get('type') for o in opts][:8]}"


def main():
    open(OUT, "w").close()
    log("start")
    import cg.api as api
    import cg.game as game

    deck = read_deck()
    rng = random.Random(1)
    obs, _ = game.battle_start(deck, deck)

    # Advance to a MAIN decision with >=2 options and a real board.
    target = None
    for _ in range(200):
        cur = obs.get("current")
        sel = obs.get("select")
        if sel is None:
            obs = game.battle_select(list(range(60)))
            continue
        if isinstance(cur, dict) and cur.get("result", -1) >= 0:
            break
        if (
            sel.get("type") == 0
            and len(sel.get("option") or []) >= 2
            and isinstance(cur, dict)
            and cur.get("turn", 0) >= 2
            and obs.get("search_begin_input")
        ):
            target = obs
            break
        obs = game.battle_select(legal_choice(sel))

    if target is None:
        log("no suitable MAIN decision found")
        game.battle_finish()
        return

    obs = target
    cur = obs["current"]
    yi = cur["yourIndex"]
    players = cur["players"]
    me, opp = players[yi], players[1 - yi]
    log(f"decision turn={cur.get('turn')} yi={yi} {sel_summary(obs['select'])}")

    # Determinize hidden state (mirror opponent).
    unseen = Counter(deck)
    for v in visible_self(me):
        if unseen.get(v, 0) > 0:
            unseen[v] -= 1
    pool = list(unseen.elements())
    rng.shuffle(pool)
    mdn = int(me.get("deckCount") or 0)
    mpn = len(me.get("prize") or [])
    your_deck = (pool[:mdn] + [deck[0]] * mdn)[:mdn]
    your_prize = (pool[mdn:mdn + mpn] + [deck[0]] * mpn)[:mpn]

    opp_pool = deck[:]
    rng.shuffle(opp_pool)
    odn = int(opp.get("deckCount") or 0)
    opn = len(opp.get("prize") or [])
    ohn = int(opp.get("handCount") or 0)
    opp_deck = (opp_pool[:odn] + [deck[0]] * odn)[:odn]
    opp_prize = (opp_pool[odn:odn + opn] + [deck[0]] * opn)[:opn]
    opp_hand = (opp_pool[odn + opn:odn + opn + ohn] + [deck[0]] * ohn)[:ohn]
    opp_active = []
    oa = opp.get("active") or []
    if oa and oa[0] is None:
        opp_active = [deck[0]]

    obs_cls = api.to_observation_class(obs)
    try:
        root = api.search_begin(
            obs_cls, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active, False
        )
    except Exception as e:
        log(f"search_begin FAILED: {type(e).__name__}: {e}")
        game.battle_finish()
        return

    root_id = root.searchId
    root_sel = root.observation.select
    n_opts = len(getattr(root_sel, "option", []) or [])
    log(f"root: id={root_id} n_opts={n_opts}")
    if n_opts < 2:
        log("root has <2 options after begin; abort")
        api.search_end()
        game.battle_finish()
        return

    import dataclasses

    def step_and_report(tag, parent_id, choice):
        try:
            ss = api.search_step(parent_id, choice)
        except Exception as e:
            log(f"  {tag}: step({parent_id},{choice}) FAILED {type(e).__name__}: {e}")
            return None
        d = dataclasses.asdict(ss.observation)
        log(f"  {tag}: parent={parent_id} choice={choice} -> child_id={ss.searchId} "
            f"sel={sel_summary(d.get('select'))}")
        return ss

    # Branch A from root.
    a1 = step_and_report("A1", root_id, [0])
    # Branch B from the SAME root id.
    b1 = step_and_report("B1", root_id, [1])
    # Re-branch A from root again (determinism / persistence check).
    a2 = step_and_report("A2(re-root)", root_id, [0])

    # If A1 succeeded, try to go deeper from child A, then branch back at root.
    if a1 is not None:
        a1_sel = dataclasses.asdict(a1.observation).get("select")
        if a1_sel and len(a1_sel.get("option") or []) >= 1:
            step_and_report("A1->child", a1.searchId, legal_choice(a1_sel))
        # After descending A's subtree, can we still branch a NEW option at root?
        if n_opts >= 3:
            step_and_report("C1(re-root after deep)", root_id, [2])

    log("branch verdict: compare A1 vs A2 child_id/sel; B1 must succeed for beam.")
    api.search_end()
    game.battle_finish()
    log("DONE")


if __name__ == "__main__":
    main()
