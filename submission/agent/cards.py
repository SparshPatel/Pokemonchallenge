"""Compact, dependency-free card stats for the runtime agent.
At runtime we cannot rely on pandas being importable inside the engine sandbox,
so card stats are precomputed offline into ``cards.json`` (see
``tools/export_cards.py``) and loaded here with the standard library only.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass

@dataclass
class CardStat:
    card_id: int
    name: str
    stage_type: str
    rule: str
    hp: int
    types: list[str]
    weakness: list[str]
    retreat: int
    best_damage: int
    is_pokemon: bool
    is_basic: bool
    is_energy: bool
    is_basic_energy: bool
    is_trainer: bool
    is_ex: bool

class CardStats:
    def __init__(self, by_id: dict[int, CardStat]):
        self._by_id = by_id

    def __contains__(self, cid: int) -> bool:
        return cid in self._by_id

    def get(self, cid: int) -> CardStat | None:
        return self._by_id.get(cid)

    @classmethod
    def load(cls, path: str) -> "CardStats":
        if not os.path.exists(path):
            return cls({})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls({})
        by_id = {}
        for row in data:
            try:
                by_id[int(row["card_id"])] = CardStat(**row)
            except Exception:
                continue
        return cls(by_id)