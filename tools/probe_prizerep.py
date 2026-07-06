"""Probe: inspect the SEARCH-state representation of hidden zones.

The planner's leaf eval counts remaining prizes as face-down (None) entries.
But search_begin is *given* concrete prize/deck IDs, so the search observation
may represent them as KNOWN (not None), which would zero-out the prize-race term.
This probe dumps, for a real obs and its search_begin result, the prize array
shape and drives our turn forward to see how taking a prize changes the count.

Writes to tools/_prize_out.txt.
Run:  python tools/probe_prizerep.py
"""
from __future__ import annotations

import dataclasses
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prize_out.txt")


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


def describe(tag, player):
    prize = player.get("prize") or []
    hand = player.get("hand")
    nnone = sum(1 for p in prize if p is None)
    sample = [card_id(p) for p in prize[:3]]
    log(f"  {tag}: prize len={len(prize)} none={nnone} sample_ids={sample} "
        f"deckCount={player.get('deckCount')} handCount={player.get('handCount')} "
        f"hand={'None' if hand is None else len(hand)}")


def main():
    open(OUT, "w").close()
    log("start")
    import cg.api as api
    import cg.game as game

    deck = read_deck()
    rng = random.Random(2)
    obs, _ = game.battle_start(deck, deck)

    target = None
    for _ in range(200):
        cur = obs.get("current")
        sel = obs.get("select")
        if sel is None:
            obs = game.battle_select(list(range(60)))
            continue
        if isinstance(cur, dict) and cur.get("result", -1) >= 0:
            break
        if (sel.get("type") == 0 and cur.get("turn", 0) >= 3
                and obs.get("search_begin_input")):
            target = obs
            break
        obs = game.battle_select(legal_choice(sel))

    if target is None:
        log("no decision found")
        game.battle_finish()
        return

    obs = target
    cur = obs["current"]
    yi = cur["yourIndex"]
    players = cur["players"]
    me, opp = players[yi], players[1 - yi]
    log(f"REAL obs turn={cur.get('turn')} yi={yi}")
    describe("REAL me ", me)
    describe("REAL opp", opp)

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
        ss = api.search_begin(
            obs_cls, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active, False
        )
    except Exception as e:
        log(f"search_begin FAILED: {type(e).__name__}: {e}")
        game.battle_finish()
        return

    sd = dataclasses.asdict(ss.observation)
    scur = sd.get("current") or {}
    splayers = scur.get("players") or []
    log(f"SEARCH obs turn={scur.get('turn')} yourIndex={scur.get('yourIndex')}")
    if len(splayers) >= 2:
        describe("SRCH me ", splayers[yi])
        describe("SRCH opp", splayers[1 - yi])

    # Drive our turn forward with legal picks; log prize-count evolution.
    from agent.adapter import extract_select
    from agent.gamedata import GameData
    from agent import rules
    gd = GameData.load()
    prev = None
    for i in range(60):
        d = dataclasses.asdict(ss.observation)
        c = d.get("current") or {}
        if c.get("result", -1) >= 0:
            log(f"  step {i}: result={c.get('result')}")
            break
        sp = c.get("players") or []
        if len(sp) >= 2:
            mp = sp[yi]
            op = sp[1 - yi]
            key = (len(mp.get("prize") or []),
                   sum(1 for p in (mp.get("prize") or []) if p is None),
                   len(op.get("prize") or []))
            if key != prev:
                log(f"  step {i}: my_prize_len={key[0]} my_none={key[1]} "
                    f"opp_prize_len={key[2]} acting={c.get('yourIndex')}")
                prev = key
        s = d.get("select")
        if s is None:
            break
        sel_obj = extract_select(d)
        try:
            ch = rules.choose(d, sel_obj, gd) if sel_obj and sel_obj.options else [0]
        except Exception:
            ch = [0]
        try:
            ss = api.search_step(ss.searchId, ch)
        except Exception as e:
            log(f"  step {i}: search_step failed {e}")
            break

    api.search_end()
    game.battle_finish()
    log("DONE")


if __name__ == "__main__":
    main()
