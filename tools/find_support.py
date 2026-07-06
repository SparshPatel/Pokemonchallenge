"""Find Basic Pokemon with consistency abilities (draw/search/attach energy)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ptcg_agent.card_data import load_default

db = load_default("EN")
KW = ("draw", "search", "attach", "look at the top", "into your hand")

rows = []
for c in db.all():
    if not c.is_basic:
        continue
    for m in c.moves:
        if m.is_ability and any(k in m.effect.lower() for k in KW):
            rows.append((c, m))
            break

rows.sort(key=lambda r: (not r[0].is_ex, -(r[0].hp or 0)))
for c, m in rows:
    tag = "ex" if c.is_ex else "  "
    print(f"{c.card_id:>4} {tag} {c.name:<28} {''.join(c.types) or '-':<3} HP{c.hp}")
    print(f"       {m.name}: {m.effect[:140]}")
