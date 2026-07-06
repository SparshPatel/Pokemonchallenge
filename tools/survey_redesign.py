"""Survey the card pool for deck-redesign candidates (dev-only, read-only).

Surfaces cards that fit the roles in the "tank wall + finisher" / evolution
scenario, so a deck redesign rests on the actual pool, not guesswork:

* ROLE A — cheap durable WALL: Basic Pokémon with a damaging 1-Energy attack,
  high HP, low retreat (buys turns while finishers power up). Single-prize
  preferred (non-ex).
* ROLE B — FINISHER: high-damage Basic attackers (ex welcome) to power on the
  Bench and bring up for the KO.
* ROLE C — EVOLUTION payoff: Stage 1/2 lines whose evolved form hits hard, with
  the Basic it evolves from (the "weak now, strong later" idea).
* ROLE D — HEALING / defensive Trainers (text scan).

Run::
    python -m tools.survey_redesign
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from ptcg_agent.card_data import load_default  # noqa: E402


def _cheapest_damaging(card):
    """(cost_total, damage) of the cheapest move that deals > 0 damage, or None."""
    best = None
    for m in card.moves:
        if m.is_ability or m.damage <= 0:
            continue
        if best is None or m.cost_total < best[0]:
            best = (m.cost_total, m.damage)
    return best


def main() -> None:
    db = load_default("EN")
    cards = db.all()

    # ROLE A — cheap durable walls: basic, 1-energy damaging attack, HP >= 110.
    print("=== ROLE A: cheap durable WALLS (Basic, 1-energy attack, HP>=110) ===")
    walls = []
    for c in cards:
        if not (c.is_basic and c.is_pokemon):
            continue
        cd = _cheapest_damaging(c)
        if cd is None or cd[0] > 1:
            continue
        if (c.hp or 0) < 110:
            continue
        walls.append((c, cd))
    walls.sort(key=lambda x: (-(x[0].hp or 0), x[0].retreat or 9))
    for c, cd in walls[:25]:
        ex = " ex" if c.is_ex else ""
        print(f"  {c.card_id:4d} {c.name[:26]:26s}{ex:3s} HP{c.hp:>3} "
              f"retreat{c.retreat if c.retreat is not None else '?'} "
              f"1E-dmg {cd[1]:>3} type {','.join(c.types)}")

    # ROLE B — finishers: basic, high best damage.
    print("\n=== ROLE B: FINISHERS (Basic, best damage >= 150) ===")
    fins = [c for c in cards if c.is_basic and c.is_pokemon and c.best_attack_damage >= 150]
    fins.sort(key=lambda c: -c.best_attack_damage)
    for c in fins[:25]:
        cd = _cheapest_damaging(c)
        ex = " ex" if c.is_ex else ""
        print(f"  {c.card_id:4d} {c.name[:26]:26s}{ex:3s} HP{c.hp:>3} "
              f"best {c.best_attack_damage:>3} cheap {cd[0] if cd else '?'}E "
              f"type {','.join(c.types)}")

    # ROLE C — evolution payoffs: Stage 1/2 with high damage + the basic it needs.
    print("\n=== ROLE C: EVOLUTION payoffs (Stage1/2, damage >= 160) ===")
    evos = [c for c in cards if c.is_pokemon and c.stage_number >= 1
            and c.best_attack_damage >= 160]
    evos.sort(key=lambda c: -c.best_attack_damage)
    for c in evos[:20]:
        cd = _cheapest_damaging(c)
        ex = " ex" if c.is_ex else ""
        print(f"  {c.card_id:4d} {c.name[:24]:24s}{ex:3s} S{c.stage_number} HP{c.hp:>3} "
              f"best {c.best_attack_damage:>3} cheap {cd[0] if cd else '?'}E "
              f"<- {c.previous_stage[:18]:18s} type {','.join(c.types)}")

    # ROLE D — healing / defensive trainers (text scan).
    print("\n=== ROLE D: HEALING / defensive Trainers ===")
    kws = ("heal", "damage from", "prevent", "reduce", "less damage")
    seen = set()
    for c in cards:
        if not c.is_trainer:
            continue
        t = (c.text or "").lower()
        if any(k in t for k in kws) and c.name not in seen:
            seen.add(c.name)
            print(f"  {c.card_id:4d} {c.name[:26]:26s} [{c.stage_type}] "
                  f"{(c.text or '')[:70]}")


if __name__ == "__main__":
    main()
