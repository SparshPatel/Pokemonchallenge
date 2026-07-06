"""Export a compact ``cards.json`` for the runtime agent from the full CSV.

Run from the project root::

    python tools/export_cards.py

Writes ``submission/cards.json`` containing one record per card with only the
fields the runtime policy needs (no pandas dependency at battle time).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_agent.card_data import load_default  # noqa: E402


def main() -> None:
    db = load_default("EN")
    rows = []
    for c in db.all():
        rows.append(
            {
                "card_id": c.card_id,
                "name": c.name,
                "stage_type": c.stage_type,
                "rule": c.rule,
                "hp": c.hp or 0,
                "types": c.types,
                "weakness": c.weakness,
                "retreat": c.retreat if c.retreat is not None else 0,
                "best_damage": c.best_attack_damage,
                "is_pokemon": c.is_pokemon,
                "is_basic": c.is_basic,
                "is_energy": c.is_energy,
                "is_basic_energy": c.is_basic_energy,
                "is_trainer": c.is_trainer,
                "is_ex": c.is_ex,
            }
        )
    out = ROOT / "submission" / "cards.json"
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} cards to {out}")


if __name__ == "__main__":
    main()
