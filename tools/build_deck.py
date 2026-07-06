"""Build the curated competitive deck and write ``submission/deck.csv``.

Archetype: **Mono-Fighting "Ancient Box" aggro** built around Basic ex attackers
and an *ability-driven* consistency/acceleration engine. The agent is piloted by
a heuristic (or belief-gated PIMC) policy that selects by option *type*, so the
deck deliberately leans on Pokémon **abilities** — which the policy triggers
reliably — rather than on trainers whose text it cannot read.

Engine / acceleration chain (all free, ability-based):
  * Drilbur — Dig Dig Dig: search up to 3 Basic {F} Energy and discard them,
    loading the discard pile.
  * Regirock ex — Regi Charge: attach up to 2 Basic {F} Energy from the discard
    pile (recovers what Drilbur dug), then swing Giant Rock (140, +140 vs Stage 2).
  * Bloodmoon Ursaluna — Battle-Hardened: attach up to 2 Basic {F} Energy from
    hand on bench-play; Mad Bite scales with the opponent's damage counters.
  * Fezandipiti ex — Flip the Script: draw 3 after a KO (comeback draw engine;
    ability is free, no Energy needed).

Attackers:
  * Koraidon ex — Impact Blow 200 (FFC), HP230 — primary beater.
  * Mega Zygarde ex — Gaia Wave 200 (FFF), HP310 — tanky closer.
  * Cornerstone Mask Ogerpon ex — Demolish 140 (FCC), ignores Weakness/effects;
    Tera ability walls ability-based attackers on the bench.
  * Regirock ex / Bloodmoon Ursaluna double as mid attackers.

Legality (standard PTCG, enforced by ``validate``):
  * exactly 60 cards,
  * at most 4 copies of any card except Basic Energy,
  * at most 1 ACE SPEC card.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_agent.card_data import CardDB, load_default  # noqa: E402

DECK_SIZE = 60
MAX_COPIES = 4
BASIC_F_ENERGY = 6  # Basic {F} Energy card ID

# (card_id, count). Documented above. 60 cards total.
CURATED_DECK: list[tuple[int, int]] = [
    # --- Pokémon (18) ---
    (979, 3),   # Koraidon ex            — main attacker, Impact Blow 200 (FFC)
    (1056, 2),  # Mega Zygarde ex        — closer, Gaia Wave 200 (FFF), HP310
    (117, 2),   # Cornerstone Mask Ogerpon ex — Demolish 140 (FCC), Tera wall
    (447, 2),   # Regirock ex            — Regi Charge accel + Giant Rock 140
    (140, 2),   # Fezandipiti ex         — Flip the Script draw 3 (free)
    (135, 4),   # Bloodmoon Ursaluna     — Battle-Hardened accel + Mad Bite
    (81, 3),    # Drilbur                — Dig Dig Dig: load discard with {F}
    # --- Trainers (24) ---
    (1121, 4),  # Ultra Ball             — search any Pokémon
    (1086, 4),  # Buddy-Buddy Poffin     — search small Basics (Drilbur, etc.)
    (1122, 4),  # Pokégear 3.0           — dig for Supporters
    (1123, 3),  # Switch                 — mobility
    (1097, 3),  # Night Stretcher        — recover Pokémon / Energy
    (1182, 3),  # Boss's Orders          — gust the target
    (1119, 2),  # Energy Search          — find Basic {F} Energy
    (1088, 1),  # Prime Catcher [ACE SPEC] — gust + switch (only ACE SPEC)
    # --- Energy (18) ---
    (BASIC_F_ENERGY, 18),  # Basic {F} Energy
]


def build_deck() -> list[int]:
    deck: list[int] = []
    for cid, n in CURATED_DECK:
        deck.extend([cid] * n)
    return deck


def validate(deck: list[int], db: CardDB) -> None:
    assert len(deck) == DECK_SIZE, f"deck has {len(deck)} cards, need {DECK_SIZE}"
    counts = Counter(deck)
    ace = 0
    for cid, n in counts.items():
        card = db.get(cid)
        assert card is not None, f"unknown card id {cid}"
        if not card.is_basic_energy:
            assert n <= MAX_COPIES, f"{card.name}: {n} copies > {MAX_COPIES}"
        if card.is_ace_spec:
            ace += n
    assert ace <= 1, f"{ace} ACE SPEC cards, max 1"
    # Need at least one Basic Pokémon to start.
    assert any(db[cid].is_basic and db[cid].is_pokemon for cid in counts), (
        "deck has no Basic Pokémon"
    )


def main() -> None:
    db = load_default("EN")
    deck = build_deck()
    validate(deck, db)

    out = ROOT / "submission" / "deck.csv"
    with out.open("w", encoding="utf-8") as f:
        for cid in deck:
            f.write(f"{cid}\n")

    counts = Counter(deck)
    print(f"Wrote {len(deck)} cards to {out}  ({len(counts)} unique)\n")
    pokemon = [c for c in counts if db[c].is_pokemon]
    trainers = [c for c in counts if db[c].is_trainer]
    energy = [c for c in counts if db[c].is_energy]
    for label, ids in (("POKÉMON", pokemon), ("TRAINERS", trainers), ("ENERGY", energy)):
        total = sum(counts[c] for c in ids)
        print(f"{label} ({total}):")
        for cid in sorted(ids, key=lambda c: -counts[c]):
            print(f"  {counts[cid]}x [{cid}] {db[cid].name}")
        print()


if __name__ == "__main__":
    main()
