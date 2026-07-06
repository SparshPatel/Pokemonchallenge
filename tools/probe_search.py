"""Probe: validate the real cabt search API (search_begin/search_step) shape.

Plays a few self-play selections with a trivial agent to reach a mid-game agent
decision, then attempts a determinized ``search_begin`` + ``search_step`` using a
mirror opponent model, printing success or the exact engine error. This confirms
PIMC is feasible from our harness before committing to the full rewrite.

Run:  python tools/probe_search.py
"""
from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
sys.path.insert(0, SUB)

import cg.api as api  # noqa: E402
import cg.game as game  # noqa: E402


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
    mx = sel.get("maxCount") or n
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


def main():
    deck = read_deck()
    rng = random.Random(0)
    obs, _ = game.battle_start(deck, deck)

    # Step forward until we hit an agent decision with a real board (~turn 1+).
    steps = 0
    target_obs = None
    while steps < 60:
        cur = obs.get("current")
        sel = obs.get("select")
        if sel is None:
            obs = game.battle_select(list(range(60)))  # deck phase (won't happen lowlevel)
            steps += 1
            continue
        if isinstance(cur, dict) and cur.get("result", -1) >= 0:
            print("game ended before probe")
            return
        # Once we have a MAIN decision with a populated board, probe here.
        if (
            isinstance(cur, dict)
            and sel.get("type") == 0
            and cur.get("turn", 0) >= 1
            and obs.get("search_begin_input")
        ):
            target_obs = obs
            break
        obs = game.battle_select(legal_choice(sel))
        steps += 1

    if target_obs is None:
        print("no suitable decision point found")
        return

    obs = target_obs
    cur = obs["current"]
    yi = cur["yourIndex"]
    players = cur["players"]
    me, opp = players[yi], players[1 - yi]
    print(f"probe at turn={cur.get('turn')} yi={yi} sbi_len={len(obs['search_begin_input'])}")

    # --- our hidden cards: deck list minus visible, split into deck/prize ----
    from collections import Counter
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
    # Pad if our reconstruction is short.
    while len(your_deck) < my_deck_n:
        your_deck.append(deck[0])
    while len(your_prize) < my_prize_n:
        your_prize.append(deck[0])

    # --- opponent mirror determinization ------------------------------------
    opp_pool = deck[:]
    rng.shuffle(opp_pool)
    opp_deck_n = int(opp.get("deckCount") or 0)
    opp_prize_n = len(opp.get("prize") or [])
    opp_hand_n = int(opp.get("handCount") or 0)
    opp_deck = opp_pool[:opp_deck_n]
    opp_prize = opp_pool[opp_deck_n:opp_deck_n + opp_prize_n]
    opp_hand = opp_pool[opp_deck_n + opp_prize_n:opp_deck_n + opp_prize_n + opp_hand_n]
    while len(opp_deck) < opp_deck_n:
        opp_deck.append(deck[0])
    while len(opp_prize) < opp_prize_n:
        opp_prize.append(deck[0])
    while len(opp_hand) < opp_hand_n:
        opp_hand.append(deck[0])

    opp_active_field = opp.get("active") or []
    opp_active = []
    if opp_active_field and opp_active_field[0] is None:
        opp_active = [979]  # a Basic Pokémon id from our deck

    obs_cls = api.to_observation_class(obs)
    print(
        f"counts: your_deck={len(your_deck)}/{my_deck_n} your_prize={len(your_prize)}/{my_prize_n} "
        f"opp_deck={len(opp_deck)}/{opp_deck_n} opp_prize={len(opp_prize)}/{opp_prize_n} "
        f"opp_hand={len(opp_hand)}/{opp_hand_n} opp_active={opp_active}"
    )
    try:
        ss = api.search_begin(
            obs_cls, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active, False
        )
        print(f"search_begin OK: searchId={ss.searchId} next_select_type="
              f"{getattr(ss.observation.select, 'type', None)}")
        choice = legal_choice(obs["select"])
        ss2 = api.search_step(ss.searchId, choice)
        nxt = ss2.observation
        print(f"search_step OK: searchId={ss2.searchId} "
              f"result={getattr(nxt.current, 'result', None) if nxt.current else None} "
              f"next_type={getattr(nxt.select, 'type', None) if nxt.select else None}")
        api.search_end()
        print("search_end OK")
    except Exception as e:
        print(f"SEARCH FAILED: {type(e).__name__}: {e}")
    finally:
        game.battle_finish()


if __name__ == "__main__":
    main()
