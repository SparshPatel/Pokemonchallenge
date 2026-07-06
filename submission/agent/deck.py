"""Deck loading for the runtime agent.

``deck.csv`` is one Card ID per line (60 lines). A header line is tolerated.
"""
from __future__ import annotations

import os


def load_deck(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"deck.csv not found at {path}")
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                # Header or non-numeric line — skip.
                continue
    return ids
