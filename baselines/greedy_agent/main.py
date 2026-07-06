"""Naive greedy-attacker baseline.

A type-priority agent with no board awareness: it attacks as soon as an attack
is legal, otherwise uses abilities / evolves / attaches / plays in a fixed
order. This is the classic "naive aggression" baseline — attacking before
finishing setup — and is the reference our setup-first policy should beat.

Self-contained (stdlib only, no shared ``agent`` package).
"""
from __future__ import annotations

import os

# OptionType priority (attack first — deliberately naive).
_PRIORITY = {
    13: 100,  # ATTACK
    10: 90,   # ABILITY
    9: 80,    # EVOLVE
    8: 70,    # ATTACH
    7: 60,    # PLAY
    1: 55,    # YES
    12: 20,   # RETREAT
    11: 10,   # DISCARD
    14: -100, # END
}

_DECK_CACHE: list[int] | None = None


def _deck() -> list[int]:
    global _DECK_CACHE
    if _DECK_CACHE is not None:
        return _DECK_CACHE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    ids: list[int] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.lstrip("-").isdigit():
                    ids.append(int(s))
    except Exception:
        pass
    _DECK_CACHE = ids
    return ids


def agent(obs):
    if not isinstance(obs, dict):
        return [0]
    sel = obs.get("select")
    if sel is None:
        return _deck()

    options = sel.get("option") or []
    n = len(options)
    mn = sel.get("minCount")
    mx = sel.get("maxCount")
    mn = 0 if mn is None else int(mn)
    mx = n if mx is None else int(mx)
    if n == 0:
        return list(range(max(0, mn)))
    mn = max(0, min(mn, n))
    mx = max(mn, min(mx, n))

    stype = sel.get("type")
    if stype == 0:  # MAIN — take the single highest-priority option.
        best_i, best_s = 0, float("-inf")
        for i, opt in enumerate(options):
            t = opt.get("type") if isinstance(opt, dict) else None
            s = _PRIORITY.get(t, 0)
            if s > best_s:
                best_s, best_i = s, i
        return [best_i]

    if stype == 9:  # YES_NO — say YES.
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and opt.get("type") == 1:
                return [i]
        return [0]

    # Other selects: take the minimum required (lowest indices).
    k = max(mn, 1) if mx >= 1 else mn
    k = min(k, n)
    return list(range(k))
