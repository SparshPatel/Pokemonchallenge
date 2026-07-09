"""Deck loading for the runtime agent.
``deck.csv`` is one Card ID per line (60 lines). A header line is tolerated.
"""
from __future__ import annotations
import os
def load_deck(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"deck.csv not found at {path}")
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            tok = line.strip().split(",")[0].strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                if lineno == 1:
                    # tolerate header
                    continue
                raise ValueError(
                    f"Invalid card id '{tok}' on line {lineno}"
                )
    if len(ids) != 60:
        raise ValueError(
            f"Expected 60 cards, found {len(ids)}."
        )
    return ids