"""Probe: measure per-decision timing and engine search cost.

Establishes the *search budget headroom* for the planner (P0):

1. Full games with the shipped rules agent -> per-decision time distribution and
   decisions-per-game (how many replans a per-decision planner would trigger).
2. Cost of a single ``search_begin`` (determinization overhead).
3. Cost of a single ``search_step`` (one simulated ply).
4. Cost of a bounded turn expansion (begin + K steps + end).

Run:  python tools/probe_timing.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

import cg.api as api  # noqa: E402
import cg.game as game  # noqa: E402

from agent.adapter import extract_select  # noqa: E402
from agent.gamedata import GameData  # noqa: E402
from agent import rules  # noqa: E402


def read_deck() -> list[int]:
    ids = []
    with open(os.path.join(SUB, "deck.csv"), encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids


def legal_choice(sel) -> list[int]:
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


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_timing_out.txt")


def log(msg):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def main():
    open(OUT, "w").close()
    log("start: importing + loading GameData")
    deck = read_deck()
    gd = GameData.load()
    log(f"GameData loaded ok={gd.ok} attacks={len(gd.attacks)} cards={len(gd.card_type)}")
    rng = random.Random(0)

    n_games = 2
    per_decision_ms = []
    decisions_per_game = []
    begin_ms = []
    step_ms = []
    turn_expand_ms = []

    for g in range(n_games):
        obs, _ = game.battle_start(deck, deck)
        dcount = 0
        probed_this_game = 0
        log(f"game {g+1}/{n_games} started")
        for _ in range(1200):
            cur = obs.get("current")
            sel = obs.get("select")
            if isinstance(cur, dict) and cur.get("result", -1) >= 0:
                break
            if sel is None:
                obs = game.battle_select(list(range(60)))
                continue
            select = extract_select(obs)
            # Time the rules decision.
            t0 = time.perf_counter()
            try:
                choice = rules.choose(obs, select, gd)
            except Exception:
                choice = legal_choice(sel)
            dt = (time.perf_counter() - t0) * 1000.0
            per_decision_ms.append(dt)
            dcount += 1

            # On a handful of MAIN decisions per game, measure engine search cost.
            if (
                probed_this_game < 3
                and sel.get("type") == 0
                and isinstance(cur, dict)
                and cur.get("turn", 0) >= 2
                and obs.get("search_begin_input")
            ):
                _measure_search(
                    obs, cur, deck, gd, rng, begin_ms, step_ms, turn_expand_ms
                )
                probed_this_game += 1

            obs = game.battle_select(choice)
        decisions_per_game.append(dcount)
        try:
            game.battle_finish()
        except Exception:
            pass
        log(
            f"  game {g+1}: decisions={dcount} begin_samples={len(begin_ms)} "
            f"step_samples={len(step_ms)}"
        )

    per_decision_ms.sort()
    begin_ms.sort()
    step_ms.sort()
    turn_expand_ms.sort()

    def summ(name, arr):
        if not arr:
            log(f"  {name:22s}: (no samples)")
            return
        log(
            f"  {name:22s}: n={len(arr):5d}  mean={sum(arr)/len(arr):7.3f}ms  "
            f"p50={pct(arr,0.5):7.3f}  p95={pct(arr,0.95):7.3f}  max={arr[-1]:7.3f}"
        )

    log(f"=== Timing over {n_games} games ===")
    summ("rules per-decision", per_decision_ms)
    log(
        f"  decisions/game        : mean={sum(decisions_per_game)/len(decisions_per_game):.1f} "
        f"min={min(decisions_per_game)} max={max(decisions_per_game)}"
    )
    summ("search_begin", begin_ms)
    summ("search_step", step_ms)
    summ("turn_expand(begin+16step)", turn_expand_ms)
    log("DONE")


def _measure_search(obs, cur, deck, gd, rng, begin_ms, step_ms, turn_expand_ms):
    yi = cur["yourIndex"]
    players = cur["players"]
    me, opp = players[yi], players[1 - yi]

    unseen = Counter(deck)
    for v in visible_self(me):
        if unseen.get(v, 0) > 0:
            unseen[v] -= 1
    pool = list(unseen.elements())
    rng.shuffle(pool)
    my_deck_n = int(me.get("deckCount") or 0)
    my_prize_n = len(me.get("prize") or [])
    your_deck = pool[:my_deck_n]
    your_prize = pool[my_deck_n:my_deck_n + my_prize_n]
    while len(your_deck) < my_deck_n:
        your_deck.append(deck[0])
    while len(your_prize) < my_prize_n:
        your_prize.append(deck[0])

    opp_pool = deck[:]
    rng.shuffle(opp_pool)
    odn = int(opp.get("deckCount") or 0)
    opn = len(opp.get("prize") or [])
    ohn = int(opp.get("handCount") or 0)
    opp_deck = opp_pool[:odn]
    opp_prize = opp_pool[odn:odn + opn]
    opp_hand = opp_pool[odn + opn:odn + opn + ohn]
    while len(opp_deck) < odn:
        opp_deck.append(deck[0])
    while len(opp_prize) < opn:
        opp_prize.append(deck[0])
    while len(opp_hand) < ohn:
        opp_hand.append(deck[0])
    opp_active = []
    oa = opp.get("active") or []
    if oa and oa[0] is None:
        for cid in opp_hand + opp_deck:
            if gd.is_basic_pokemon(cid):
                opp_active = [cid]
                break

    try:
        obs_cls = api.to_observation_class(obs)
    except Exception:
        return

    t0 = time.perf_counter()
    try:
        ss = api.search_begin(
            obs_cls, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active, False
        )
    except Exception:
        return
    begin_ms.append((time.perf_counter() - t0) * 1000.0)

    # Drive up to 16 legal steps forward with the rules policy; time each step.
    n_steps = 0
    try:
        for _ in range(16):
            nxt = ss.observation
            ncur = getattr(nxt, "current", None)
            if ncur is None or getattr(ncur, "result", -1) >= 0:
                break
            nsel = getattr(nxt, "select", None)
            if nsel is None:
                break
            import dataclasses

            odict = dataclasses.asdict(nxt)
            nselect = extract_select(odict)
            if nselect is None or not nselect.options:
                ch = list(range(int(getattr(nsel, "minCount", 0) or 0)))
            else:
                try:
                    ch = rules.choose(odict, nselect, gd)
                except Exception:
                    ch = list(range(max(1, int(getattr(nsel, "minCount", 0) or 0))))
            ts = time.perf_counter()
            ss = api.search_step(ss.searchId, ch)
            step_ms.append((time.perf_counter() - ts) * 1000.0)
            n_steps += 1
    except Exception:
        pass
    finally:
        try:
            api.search_end()
        except Exception:
            pass
    turn_expand_ms.append((time.perf_counter() - t0) * 1000.0)


if __name__ == "__main__":
    main()
