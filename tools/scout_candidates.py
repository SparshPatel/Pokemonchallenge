"""Print attack details (text/drawbacks) for candidate alternative attackers."""
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'submission')
import cg.api as api

ETYPE = {0: 'C', 1: 'G', 2: 'R', 3: 'W', 4: 'L', 5: 'P', 6: 'F', 7: 'D', 8: 'M', 9: 'N', 10: 'A'}
cards = {c.cardId: c for c in api.all_card_data()}
atks = {a.attackId: a for a in api.all_attack()}

CANDIDATES = [
    (46, 'Gouging Fire ex'), (1062, 'Yveltal ex'), (336, 'Zacian ex'),
    (313, 'Miraidon ex'), (328, 'Pikachu ex'), (756, 'Mega Kangaskhan ex'),
    (139, 'Munkidori ex'), (75, 'Iron Leaves ex'), (1002, 'Zangoose ex'),
    (176, 'Terapagos ex'), (138, 'Okidogi ex'), (886, 'Mega Hawlucha ex'),
    (37, 'Iron Thorns ex'), (431, "TR Mewtwo ex"), (806, 'Rotom ex'),
    (369, 'Dondozo ex'),
]

def cost_str(e):
    return ''.join(ETYPE.get(x, '?') for x in e) if e else 'free'

for cid, label in CANDIDATES:
    c = cards.get(cid)
    if not c:
        print(f"[{cid}] {label}: NOT FOUND")
        continue
    print(f"[{cid}] {c.name} — hp={c.hp} type={ETYPE.get(c.energyType,'?')} "
          f"weak={ETYPE.get(c.weakness,'?')} retreat={c.retreatCost} "
          f"prize={3 if c.megaEx else (2 if c.ex else 1)}")
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a:
            print(f"    -> {a.name}: {a.damage}dmg [{cost_str(a.energies)}]  {(a.text or '').strip()}")
    # abilities
    for sk in (c.skills or []):
        print(f"    [ABILITY] {sk.name}: {(sk.text or '').strip()}")
    print()
