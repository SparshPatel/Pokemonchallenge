"""Procedural generator for a pool of legal, playable training decks.

The offline trainer (:mod:`ptcg_agent.train`) tunes the agent's decision weights
by **self-play across many decks** so the resulting policy is deck-agnostic and
robust, not overfit to our one curated list. To do that it needs a large, varied
supply of decks that are (a) legal and (b) actually playable by a type-driven
heuristic agent.

Strategy: build **mono-type Basic-attacker aggro** decks. Basic-only attacker
lines mean no evolution dependencies, so any reasonable agent can pilot them; a
shared package of type-agnostic staple Items/Supporters provides consistency;
the remainder is filled with the matching Basic Energy. Randomizing the species
selection, copy counts and energy count yields effectively unlimited distinct —
yet sensible — decks.

Decks are produced as ``list[int]`` of 60 Card IDs (the engine's deck format).
"""
from __future__ import annotations

import random
from collections import Counter

from .card_data import CardDB, Card, load_default

DECK_SIZE = 60
MAX_COPIES = 4

# Basic Energy Card ID per internal type code (from EN_Card_Data.csv).
BASIC_ENERGY_BY_TYPE: dict[str, int] = {
    "G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8,
}
ATTACK_TYPES = tuple(BASIC_ENERGY_BY_TYPE.keys())

# Type-agnostic consistency staples, resolved by name at load time. Each entry is
# (list-of-candidate-names, copies). The first name that resolves to a non-ACE
# card of the expected kind is used; unresolved entries are simply skipped, so
# the generator degrades gracefully if the pool differs.
_STAPLE_SPEC: list[tuple[tuple[str, ...], int]] = [
    (("Ultra Ball",), 4),
    (("Buddy-Buddy Poffin",), 4),
    (("Pokégear 3.0", "Pokegear 3.0"), 2),
    (("Switch",), 2),
    (("Night Stretcher",), 2),
    (("Boss’s Orders", "Boss's Orders"), 2),
    (("Cheren", "Urbain"), 4),  # unconditional "Draw 3" Supporter engine
]


def _resolve_staple(db: CardDB, names: tuple[str, ...]) -> int | None:
    """Return the Card ID of the first matching, non-ACE Trainer for ``names``."""
    for name in names:
        target = name.lower()
        for c in db.all():
            if c.is_trainer and not c.is_ace_spec and c.name.lower() == target:
                return c.card_id
        # Fall back to a substring match if no exact hit.
        for c in db.all():
            if c.is_trainer and not c.is_ace_spec and target in c.name.lower():
                return c.card_id
    return None


def _staple_package(db: CardDB) -> list[tuple[int, int]]:
    pkg: list[tuple[int, int]] = []
    for names, copies in _STAPLE_SPEC:
        cid = _resolve_staple(db, names)
        if cid is not None:
            pkg.append((cid, copies))
    return pkg


def _attackers_by_type(db: CardDB) -> dict[str, list[Card]]:
    """Basic Pokémon with a real attack, grouped by primary type, best first."""
    by_type: dict[str, list[Card]] = {t: [] for t in ATTACK_TYPES}
    for c in db.all():
        if not (c.is_basic and c.is_pokemon):
            continue
        if c.best_attack_damage <= 0 or not c.types:
            continue
        t = c.types[0]
        if t in by_type:
            by_type[t].append(c)
    for t in by_type:
        by_type[t].sort(key=lambda c: c.best_attack_damage, reverse=True)
    return by_type


class DeckPool:
    """Lazily-built supply of legal training decks."""

    def __init__(self, db: CardDB | None = None):
        self.db = db or load_default("EN")
        self.staples = _staple_package(self.db)
        self.attackers = _attackers_by_type(self.db)
        # Only keep types that have enough strong Basic attackers to build around.
        self.usable_types = [
            t for t in ATTACK_TYPES if len(self.attackers[t]) >= 3
        ]

    def generate(self, rng: random.Random) -> list[int]:
        """Build one legal 60-card mono-type aggro deck."""
        for _ in range(20):  # retry until a legal deck falls out
            deck = self._try_generate(rng)
            if deck is not None:
                return deck
        # Extremely unlikely fallback: a minimal legal mono-Fighting list.
        return self._fallback(rng)

    def _try_generate(self, rng: random.Random) -> list[int] | None:
        t = rng.choice(self.usable_types)
        pool = self.attackers[t]
        # Draw species from the stronger half of the type's attackers.
        top = pool[: max(8, len(pool) // 2)]

        counts: Counter[int] = Counter()
        for cid, n in self.staples:
            counts[cid] += n
        trainers = sum(counts.values())

        # Realistic proportions: a fixed Energy target, the rest is Pokémon —
        # rather than letting Energy absorb a huge remainder.
        energy_target = rng.randint(13, 18)
        pokemon_target = DECK_SIZE - trainers - energy_target
        if pokemon_target < 8:
            return None

        # Enough species to reach the Pokémon target at <= MAX_COPIES each.
        need_species = pokemon_target // MAX_COPIES + 2
        species = rng.sample(top, min(len(top), max(3, need_species)))
        remaining = pokemon_target
        while remaining > 0:
            progressed = False
            for c in species:
                if remaining <= 0:
                    break
                if counts[c.card_id] < MAX_COPIES:
                    counts[c.card_id] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                return None  # ran out of species capacity

        counts[BASIC_ENERGY_BY_TYPE[t]] += energy_target

        deck = [cid for cid, n in counts.items() for _ in range(n)]
        if len(deck) == DECK_SIZE and self.validate(deck):
            rng.shuffle(deck)
            return deck
        return None

    def _fallback(self, rng: random.Random) -> list[int]:
        counts: Counter[int] = Counter()
        for c in self.attackers["F"][:3]:
            counts[c.card_id] += 4
        for cid, n in self.staples:
            counts[cid] += n
        used = sum(counts.values())
        counts[BASIC_ENERGY_BY_TYPE["F"]] += max(0, DECK_SIZE - used)
        deck = [cid for cid, n in counts.items() for _ in range(n)]
        deck = deck[:DECK_SIZE]
        while len(deck) < DECK_SIZE:
            deck.append(BASIC_ENERGY_BY_TYPE["F"])
        return deck

    def validate(self, deck: list[int]) -> bool:
        if len(deck) != DECK_SIZE:
            return False
        counts = Counter(deck)
        ace = 0
        has_basic_pokemon = False
        for cid, n in counts.items():
            card = self.db.get(cid)
            if card is None:
                return False
            if not card.is_basic_energy and n > MAX_COPIES:
                return False
            if card.is_ace_spec:
                ace += n
            if card.is_basic and card.is_pokemon:
                has_basic_pokemon = True
        return ace <= 1 and has_basic_pokemon

    def sample(self, n: int, seed: int = 0) -> list[list[int]]:
        """Return ``n`` distinct legal decks reproducibly from ``seed``."""
        rng = random.Random(seed)
        return [self.generate(rng) for _ in range(n)]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Preview generated training decks.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = DeckPool()
    print(f"usable types: {pool.usable_types}")
    print(f"staples: {[(cid, n) for cid, n in pool.staples]}")
    for i, deck in enumerate(pool.sample(args.n, args.seed)):
        counts = Counter(deck)
        names = sorted(counts.items(), key=lambda kv: -kv[1])
        summary = ", ".join(
            f"{n}x {pool.db[cid].name}" for cid, n in names[:6]
        )
        print(f"deck {i}: {len(deck)} cards, {len(counts)} unique | {summary} ...")
