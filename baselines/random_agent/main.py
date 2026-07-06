"""Random-legal baseline agent (the canonical floor).

Mirrors the competition's sample agent: during the deck phase it returns the
fixed 60-card deck; otherwise it picks a random legal number of random option
indices. Self-contained (stdlib only, no shared ``agent`` package) so the
harness can load it alongside the real submission without import clashes.
"""
from __future__ import annotations

import os
import random

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
    k = random.randint(mn, mx)
    if k == 0 and sel.get("type") == 0:  # MAIN must take an action
        k = 1
    k = min(k, n)
    return sorted(random.sample(range(n), k))
