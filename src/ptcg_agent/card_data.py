"""Card data loading and feature extraction for the Pokémon TCG card pool.

The competition ships ``EN_Card_Data.csv`` with one row per move/ability, so a
single card with two attacks spans two rows. This module collapses those rows
into one :class:`Card` per ``Card ID`` and parses the energy/cost notation used
throughout the file.

Energy notation
---------------
* ``{G}``, ``{R}``, ``{W}``, ``{L}``, ``{P}``, ``{F}``, ``{D}``, ``{M}``,
  ``{C}`` are the typed energy symbols (Grass, Fire, Water, Lightning, Psychic,
  Fighting, Darkness, Metal, Colorless).
* ``●`` is a loose colorless energy requirement in a move cost.
* ``{A}`` and ``{Team Rocket}`` appear on a handful of special cards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"

# Map raw type tokens to short codes used internally.
TYPE_TOKENS = {
    "{G}": "G", "{R}": "R", "{W}": "W", "{L}": "L", "{P}": "P",
    "{F}": "F", "{D}": "D", "{M}": "M", "{C}": "C", "{A}": "A",
    "竜": "N",  # Dragon
}

_ENERGY_TOKEN_RE = re.compile(r"\{([^}]+)\}|(●)")


def parse_energy(text: str) -> dict[str, int]:
    """Parse an energy string like ``{W}{W}●`` into ``{'W': 2, 'C': 1}``.

    Loose colorless symbols (``●``) and ``{C}`` both count as colorless (``C``).
    Returns an empty dict for blank / ``n/a`` values.
    """
    if not text or text in ("n/a", "nan"):
        return {}
    counts: dict[str, int] = {}
    for braced, dot in _ENERGY_TOKEN_RE.findall(text):
        if dot:
            counts["C"] = counts.get("C", 0) + 1
            continue
        token = "{" + braced + "}"
        code = TYPE_TOKENS.get(token)
        if code is None:
            # Unknown / multi-word token (e.g. Team Rocket) — bucket as special.
            code = braced
        counts[code] = counts.get(code, 0) + 1
    return counts


def _parse_int(text: str) -> int | None:
    if not text or text in ("n/a", "nan"):
        return None
    m = re.match(r"\d+", text)
    return int(m.group()) if m else None


@dataclass
class Move:
    name: str
    cost: dict[str, int]
    cost_total: int
    damage: int
    damage_variable: bool  # True if damage uses ×, +, or other modifiers
    effect: str
    is_ability: bool


@dataclass
class Card:
    card_id: int
    name: str
    expansion: str
    collection_no: str
    category: str          # raw Category column (archetype tag), often empty
    stage_type: str        # e.g. "Basic Pokémon", "Item", "Special Energy"
    rule: str              # "Pokémon ex", "ACE SPEC", "Mega Pokémon ex", ""
    previous_stage: str
    hp: int | None
    types: list[str]       # Pokémon type codes (usually one)
    weakness: list[str]
    resistance: list[str]
    retreat: int | None
    moves: list[Move] = field(default_factory=list)
    text: str = ""            # rules text for Trainers / Special Energy

    # --- convenience predicates -------------------------------------------
    @property
    def is_pokemon(self) -> bool:
        return self.stage_type.endswith("Pokémon")

    @property
    def is_basic(self) -> bool:
        return self.stage_type == "Basic Pokémon"

    @property
    def is_energy(self) -> bool:
        return "Energy" in self.stage_type

    @property
    def is_basic_energy(self) -> bool:
        return self.stage_type == "Basic Energy"

    @property
    def is_trainer(self) -> bool:
        return self.stage_type in ("Item", "Supporter", "Pokémon Tool", "Stadium")

    @property
    def is_ex(self) -> bool:
        return "ex" in self.rule.lower()

    @property
    def is_ace_spec(self) -> bool:
        return self.rule == "ACE SPEC"

    @property
    def stage_number(self) -> int:
        """0 for Basic, 1 for Stage 1, 2 for Stage 2; -1 for non-Pokémon."""
        if self.stage_type == "Basic Pokémon":
            return 0
        if self.stage_type == "Stage 1 Pokémon":
            return 1
        if self.stage_type == "Stage 2 Pokémon":
            return 2
        return -1

    @property
    def best_attack_damage(self) -> int:
        dmgs = [m.damage for m in self.moves if not m.is_ability]
        return max(dmgs) if dmgs else 0


class CardDB:
    """In-memory database of all cards, keyed by ``Card ID``."""

    def __init__(self, cards: dict[int, Card]):
        self.cards = cards

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, card_id: int) -> Card:
        return self.cards[card_id]

    def __contains__(self, card_id: int) -> bool:
        return card_id in self.cards

    def get(self, card_id: int) -> Card | None:
        return self.cards.get(card_id)

    def all(self) -> list[Card]:
        return list(self.cards.values())

    def filter(self, predicate) -> list[Card]:
        return [c for c in self.cards.values() if predicate(c)]

    @classmethod
    def from_csv(cls, path: str | Path) -> "CardDB":
        df = pd.read_csv(path, encoding="utf-8", dtype=str).fillna("")
        cards: dict[int, Card] = {}
        for _, row in df.iterrows():
            cid = _parse_int(row["Card ID"])
            if cid is None:
                continue
            move = _row_to_move(row)
            if cid in cards:
                if move is not None:
                    cards[cid].moves.append(move)
                continue
            cards[cid] = Card(
                card_id=cid,
                name=row["Card Name"].strip(),
                expansion=row["Expansion"].strip(),
                collection_no=row["Collection No."].strip(),
                category=row["Category"].strip(),
                stage_type=row[STAGE_COL].strip(),
                rule=_clean(row["Rule"]),
                previous_stage=_clean(row["Previous stage"]),
                hp=_parse_int(row["HP"]),
                types=_parse_types(row["Type"]),
                weakness=_parse_types(row["Weakness"]),
                resistance=_parse_types(row["Resistance (Type)"]),
                retreat=_parse_int(row["Retreat"]),
                moves=[move] if move is not None else [],
                text=_clean(row["Effect Explanation"]) if move is None else "",
            )
        return cls(cards)


def _clean(text: str) -> str:
    return "" if text.strip() in ("n/a", "nan") else text.strip()


def _parse_types(text: str) -> list[str]:
    energy = parse_energy(text)
    return [code for code, n in energy.items() for _ in range(n)]


def _row_to_move(row: pd.Series) -> Move | None:
    name = row["Move Name"].strip()
    if not name or name in ("n/a", "nan"):
        return None
    cost = parse_energy(row["Cost"])
    damage_raw = row["Damage"].strip()
    damage = _parse_int(damage_raw) or 0
    return Move(
        name=name,
        cost=cost,
        cost_total=sum(cost.values()),
        damage=damage,
        damage_variable=bool(re.search(r"[×x+\-]", damage_raw)),
        effect=_clean(row["Effect Explanation"]),
        is_ability=name.startswith("[Ability]") or name.startswith("[Tera]"),
    )


def load_default(lang: str = "EN") -> CardDB:
    """Load the bundled card database for the given language (``EN`` or ``JP``)."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "raw" / f"{lang}_Card_Data.csv"
    return CardDB.from_csv(path)
