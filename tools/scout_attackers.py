"""Rank Basic Pokemon attackers by damage, HP, and energy efficiency.

Used to scout alternative deck archetypes. The field opponents pilot mono-type
Basic aggro built from top-damage Basics, so we want to know the whole landscape:
best OHKO threats, best tanks, and cheapest attackers per type.
"""
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'submission')
import cg.api as api

ETYPE = {0: 'C', 1: 'G', 2: 'R', 3: 'W', 4: 'L', 5: 'P', 6: 'F', 7: 'D', 8: 'M', 9: 'N', 10: 'A'}

cards = {c.cardId: c for c in api.all_card_data()}
atks = {a.attackId: a for a in api.all_attack()}


def best_attack(c):
    """Return (dmg, cost_len, cost_str, name) of the highest-damage attack."""
    best = (0, 99, '', '')
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a is None:
            continue
        cost = a.energies or []
        cs = ''.join(ETYPE.get(x, '?') for x in cost)
        if a.damage > best[0]:
            best = (a.damage, len(cost), cs, a.name)
    return best


# Collect all Basic Pokemon with an attack
rows = []
for cid, c in cards.items():
    if not c.basic or c.cardType != 0:
        continue
    dmg, clen, cs, aname = best_attack(c)
    if dmg <= 0:
        continue
    ptype = ETYPE.get(c.energyType, '?')
    prize = 3 if c.megaEx else (2 if c.ex else 1)
    eff = dmg / max(1, clen)  # damage per energy
    rows.append({
        'cid': cid, 'name': c.name, 'type': ptype, 'hp': c.hp, 'dmg': dmg,
        'cost': cs, 'clen': clen, 'eff': eff, 'prize': prize,
        'ex': c.ex, 'mega': c.megaEx, 'retreat': c.retreatCost, 'atk': aname,
    })

print("=" * 100)
print("TOP 30 BASIC ATTACKERS BY RAW DAMAGE")
print("=" * 100)
for r in sorted(rows, key=lambda x: -x['dmg'])[:30]:
    print(f"  [{r['cid']:4d}] {r['name'][:28]:28s} {r['type']} hp={r['hp']:3d} "
          f"dmg={r['dmg']:3d} cost={r['cost']:5s} eff={r['eff']:5.1f} "
          f"prize={r['prize']} rt={r['retreat']}  {r['atk'][:24]}")

print()
print("=" * 100)
print("TOP 25 BY DAMAGE-PER-ENERGY (efficiency) — min 60 dmg")
print("=" * 100)
for r in sorted([x for x in rows if x['dmg'] >= 60], key=lambda x: -x['eff'])[:25]:
    print(f"  [{r['cid']:4d}] {r['name'][:28]:28s} {r['type']} hp={r['hp']:3d} "
          f"dmg={r['dmg']:3d} cost={r['cost']:5s} eff={r['eff']:5.1f} "
          f"prize={r['prize']} rt={r['retreat']}  {r['atk'][:24]}")

print()
print("=" * 100)
print("TOP 20 SINGLE-ENERGY ATTACKERS (1 energy, best damage) — snappy setup")
print("=" * 100)
for r in sorted([x for x in rows if x['clen'] == 1], key=lambda x: -x['dmg'])[:20]:
    print(f"  [{r['cid']:4d}] {r['name'][:28]:28s} {r['type']} hp={r['hp']:3d} "
          f"dmg={r['dmg']:3d} cost={r['cost']:5s} prize={r['prize']} rt={r['retreat']}  {r['atk'][:24]}")

print()
print("=" * 100)
print("TOP 20 TANKS (highest HP Basic) with a real attack")
print("=" * 100)
for r in sorted(rows, key=lambda x: -x['hp'])[:20]:
    print(f"  [{r['cid']:4d}] {r['name'][:28]:28s} {r['type']} hp={r['hp']:3d} "
          f"dmg={r['dmg']:3d} cost={r['cost']:5s} prize={r['prize']} rt={r['retreat']}  {r['atk'][:24]}")
