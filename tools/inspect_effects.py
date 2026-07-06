"""Inspect effect text for specific card IDs. Usage: python tools/inspect_effects.py 313 979 ..."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ptcg_agent.card_data import load_default

db = load_default("EN")
ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else []

for cid in ids:
    c = db.get(cid)
    if c is None:
        print(f"{cid}: NOT FOUND"); continue
    print(f"\n[{cid}] {c.name}  ({c.stage_type}{', '+c.rule if c.rule else ''})  HP={c.hp} types={c.types} retreat={c.retreat}")
    if c.text:
        print(f"   TEXT: {c.text}")
    for m in c.moves:
        kind = "ABILITY" if m.is_ability else f"ATK {m.damage}{'*' if m.damage_variable else ''} cost={''.join(k*v for k,v in m.cost.items()) or '-'}"
        print(f"   - {m.name} [{kind}]")
        if m.effect:
            print(f"       {m.effect}")
