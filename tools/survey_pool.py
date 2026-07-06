"""Survey the card pool to inform deck construction. Run: python tools/survey_pool.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ptcg_agent.card_data import load_default

db = load_default("EN")


def show(label, cards):
    print(f"\n=== {label} ({len(cards)}) ===")
    for c in cards:
        rule = f" [{c.rule}]" if c.rule else ""
        print(f"  {c.card_id}: {c.name}{rule}")


show("SUPPORTERS", [c for c in db.all() if c.stage_type == "Supporter"])
show("ITEMS", [c for c in db.all() if c.stage_type == "Item"])
show("POKEMON TOOLS", [c for c in db.all() if c.stage_type == "Pokémon Tool"])
show("STADIUMS", [c for c in db.all() if c.stage_type == "Stadium"])
show("SPECIAL ENERGY", [c for c in db.all() if c.stage_type == "Special Energy"])

# Strong Basic ex attackers: basic, ex, decent damage, payable cost.
basic_ex = [
    c for c in db.all()
    if c.is_basic and c.is_ex and not c.is_ace_spec and c.best_attack_damage >= 100
]
basic_ex.sort(key=lambda c: c.best_attack_damage, reverse=True)
print(f"\n=== BASIC ex ATTACKERS (dmg>=100) ({len(basic_ex)}) ===")
for c in basic_ex[:40]:
    costs = "; ".join(
        f"{m.name}={m.damage}{'*' if m.damage_variable else ''}({''.join(k*v for k,v in m.cost.items())})"
        for m in c.moves if not m.is_ability
    )
    ab = "; ".join(m.name for m in c.moves if m.is_ability)
    print(f"  {c.card_id}: {c.name} HP{c.hp} types={c.types} retreat={c.retreat} | {costs}" + (f" | AB:{ab}" if ab else ""))
