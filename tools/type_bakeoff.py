"""Build the strongest Basic-aggro deck for each energy type and (a few) two-type
fusions, then write them as deck.csv files for the field bake-off (dev-only).

Motivation: our shipped list is mono-Fighting "Ancient Box". The open question is
whether a *different* type — or a two-type fusion that shares an energy base —
yields a stronger fixed deck against the opponent field. Rather than guess, we
build one principled deck per type from the actual card pool and let
``ptcg_agent.evaluate --mode field`` measure them head-to-head against D1.

Deck shape mirrors the proven D1 structure:
  * a high-damage finisher core (top attackers, ex welcome),
  * one durable single-prize **wall** (high HP, cheap attack) to buy turns,
  * the same type-agnostic Trainer staple package,
  * Basic Energy of the deck's type(s).

Energy payability is respected: an attacker is only considered if its cheapest
damaging attack can be paid by the deck's chosen energy base (its colored
symbols are a subset of the base; Colorless pips are wild). Colorless attackers
are therefore eligible in *every* deck — the real "fusion" lever.

Run::
    python -m tools.type_bakeoff                # build all type decks + fusions
    python -m tools.type_bakeoff --list         # just print what would be built
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from ptcg_agent.card_data import Card, CardDB, load_default  # noqa: E402
from ptcg_agent.deckgen import BASIC_ENERGY_BY_TYPE, _staple_package  # noqa: E402

DECK_SIZE = 60
MAX_COPIES = 4
OUT_DIR = os.path.join(_ROOT, "artifacts", "deck_candidates", "types")

# Human labels for the single-letter type codes.
TYPE_NAME = {
    "G": "Grass", "R": "Fire", "W": "Water", "L": "Lightning",
    "P": "Psychic", "F": "Fighting", "D": "Darkness", "M": "Metal",
    "C": "Colorless", "N": "Dragon",
}


def _cheapest_damaging(card: Card) -> tuple[int, dict[str, int], int] | None:
    """(cost_total, colored-cost dict, damage) of the cheapest damaging move."""
    best = None
    for m in card.moves:
        if m.is_ability or m.damage <= 0:
            continue
        if best is None or m.cost_total < best[0]:
            best = (m.cost_total, m.cost, m.damage)
    return best


def _payable(colored: dict[str, int], base: set[str]) -> bool:
    """True if every *colored* (non-Colorless) pip is covered by the energy base."""
    for sym, n in colored.items():
        if sym in ("C", "Colorless", ""):
            continue
        if sym not in base:
            return False
    return True


def _score(card: Card, cd: tuple[int, dict[str, int], int]) -> float:
    """Rank attackers: reward damage and HP, penalise heavy energy cost."""
    cost_total, _colored, dmg = cd
    hp = card.hp or 0
    return dmg - 14.0 * cost_total + 0.12 * hp


def _attackers(db: CardDB, base: set[str]) -> list[tuple[Card, tuple]]:
    """Basic attackers whose cheapest damaging attack is payable by ``base``."""
    out = []
    for c in db.all():
        if not (c.is_basic and c.is_pokemon):
            continue
        cd = _cheapest_damaging(c)
        if cd is None:
            continue
        if not _payable(cd[1], base):
            continue
        out.append((c, cd))
    out.sort(key=lambda x: _score(x[0], x[1]), reverse=True)
    return out


def _best_wall(db: CardDB, base: set[str]) -> Card | None:
    """Highest-HP single-prize Basic with a cheap (<=1 colored) damaging attack."""
    best = None
    for c in db.all():
        if not (c.is_basic and c.is_pokemon) or c.is_ex:
            continue
        cd = _cheapest_damaging(c)
        if cd is None or cd[0] > 1 or not _payable(cd[1], base):
            continue
        if (c.hp or 0) < 110:
            continue
        if best is None or (c.hp or 0) > (best.hp or 0):
            best = c
    return best


def build_deck(db: CardDB, base: list[str], staples: list[tuple[int, int]]) -> list[int] | None:
    """Build one 60-card aggro deck on energy ``base`` (1 or 2 type codes)."""
    base_set = set(base)
    attackers = _attackers(db, base_set)
    if len(attackers) < 3:
        return None

    counts: Counter[int] = Counter()
    for cid, n in staples:
        counts[cid] += n
    trainers = sum(counts.values())

    energy_target = 17
    pokemon_target = DECK_SIZE - trainers - energy_target

    # Wall first (single-prize durability), then the strongest finishers.
    chosen: list[Card] = []
    wall = _best_wall(db, base_set)
    if wall is not None:
        chosen.append(wall)
    for c, _cd in attackers:
        if len(chosen) >= 6:
            break
        if c.card_id not in [x.card_id for x in chosen]:
            chosen.append(c)

    # Fill the Pokémon slots: wall gets 3, finishers up to 4 each, best first.
    remaining = pokemon_target
    plan: list[tuple[int, int]] = []
    if wall is not None:
        plan.append((wall.card_id, 3))
        remaining -= 3
    for c in chosen:
        if c is wall:
            continue
        n = min(MAX_COPIES, remaining)
        if n <= 0:
            break
        plan.append((c.card_id, n))
        remaining -= n
    if remaining > 0 and plan:  # top up the best finisher
        # add to the highest-damage non-wall species not yet at MAX
        for i, (cid, n) in enumerate(plan):
            if cid == (wall.card_id if wall else None):
                continue
            add = min(MAX_COPIES - n, remaining)
            if add > 0:
                plan[i] = (cid, n + add)
                remaining -= add
            if remaining <= 0:
                break

    for cid, n in plan:
        counts[cid] += n

    # Energy split across the base types.
    energy_left = DECK_SIZE - sum(counts.values())
    if energy_left <= 0:
        return None
    per = energy_left // len(base)
    for i, t in enumerate(base):
        n = per + (energy_left - per * len(base) if i == 0 else 0)
        counts[BASIC_ENERGY_BY_TYPE[t]] += n

    deck = [cid for cid, n in counts.items() for _ in range(n)]
    if len(deck) != DECK_SIZE:
        return None
    return deck


def _write(path: str, deck: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for cid in deck:
            fh.write(f"{cid}\n")


# Two-type fusions worth trying (share-energy decks). Each base must be a real
# Basic-Energy type; Colorless attackers are auto-eligible in every base, so a
# "Type + Colorless" fusion is simply that type's deck (no separate entry needed).
FUSIONS: list[tuple[str, list[str]]] = [
    ("fusion_lightning_metal", ["L", "M"]),     # Zacian/Pikachu shells overlap
    ("fusion_water_lightning", ["W", "L"]),     # classic Dragon-support pair
    ("fusion_fire_fighting", ["R", "F"]),       # Ho-Oh / Gouging Fire + Fighting
    ("fusion_psychic_darkness", ["P", "D"]),    # Mewtwo / Yveltal control
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print decks, don't simulate")
    args = ap.parse_args()

    db = load_default("EN")
    staples = _staple_package(db)

    built: list[tuple[str, list[int]]] = []
    # Mono-type decks (Colorless excluded as a base: it has no basic energy).
    for t in ("G", "R", "W", "L", "P", "F", "D", "M"):
        deck = build_deck(db, [t], staples)
        if deck is None:
            print(f"  (skip {TYPE_NAME[t]}: not enough attackers)")
            continue
        name = f"mono_{TYPE_NAME[t].lower()}"
        path = os.path.join(OUT_DIR, f"{name}.csv")
        _write(path, deck)
        built.append((name, deck))

    for name, base in FUSIONS:
        deck = build_deck(db, base, staples)
        if deck is None:
            print(f"  (skip {name}: not enough attackers)")
            continue
        path = os.path.join(OUT_DIR, f"{name}.csv")
        _write(path, deck)
        built.append((name, deck))

    print(f"\nBuilt {len(built)} decks under {OUT_DIR}:")
    for name, deck in built:
        counts = Counter(deck)
        pkmn = sorted(
            ((cid, n) for cid, n in counts.items()
             if (c := db.get(cid)) and c.is_pokemon),
            key=lambda x: -(db.get(x[0]).best_attack_damage),
        )
        head = ", ".join(
            f"{db.get(cid).name}×{n}" for cid, n in pkmn[:4]
        )
        print(f"  {name:28s} | {head}")


if __name__ == "__main__":
    main()
