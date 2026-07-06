"""Survey card pool for adversarial archetype pieces."""
import sys
sys.path.insert(0,'submission'); sys.path.insert(0,'src')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}
atks  = {a.attackId: a for a in api.all_attack()}

ETYPE = {0:'C',1:'G',2:'R',3:'W',4:'L',5:'P',6:'F',7:'D',8:'M',9:'N',10:'A'}
ENAME = {0:'Colorless',1:'Grass',2:'Fire',3:'Water',4:'Lightning',
         5:'Psychic',6:'Fighting',7:'Darkness',8:'Metal',9:'Dragon',10:'Rainbow'}

def cost_str(energies):
    return ''.join(ETYPE.get(e,'?') for e in energies) if energies else 'free'

def stage(c):
    if c.megaEx: return 'Mega'
    if c.ex:     return 'ex'
    if c.stage2: return 'S2'
    if c.stage1: return 'S1'
    return 'Ba'

def best_atk(c):
    best = None
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a and (best is None or a.damage > best.damage):
            best = a
    return best

def cheap_atk(c):
    """Attack with fewest energy pips (ties broken by damage desc)."""
    best = None
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a and a.damage > 0:
            if best is None or len(a.energies) < len(best.energies) or \
               (len(a.energies)==len(best.energies) and a.damage > best.damage):
                best = a
    return best

# ------------------------------------------------------------------ #
print('=== GRASS attackers (kryptonite: 5/7 of our Pokémon weak to Grass) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or c.energyType != 1: continue
    a = best_atk(c)
    if a and a.damage >= 80:
        rows.append((c, a))
rows.sort(key=lambda x: x[1].damage, reverse=True)
for c,a in rows[:16]:
    print(f'  {c.cardId:5} {c.name:<36} {stage(c):4} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}]')

print()
print('=== PSYCHIC attackers (Koraidon ex + Great Tusk both weak to Psychic) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or c.energyType != 5: continue
    a = best_atk(c)
    if a and a.damage >= 80:
        rows.append((c, a))
rows.sort(key=lambda x: x[1].damage, reverse=True)
for c,a in rows[:16]:
    print(f'  {c.cardId:5} {c.name:<36} {stage(c):4} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}]')

print()
print('=== FAST (<=2 energy, dmg>=60) BASICS — any type (race archetype) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or not c.basic: continue
    a = cheap_atk(c)
    if a and a.damage >= 60 and len(a.energies) <= 2:
        rows.append((c, a))
rows.sort(key=lambda x: (-x[1].damage, len(x[1].energies)))
for c,a in rows[:16]:
    print(f'  {c.cardId:5} {c.name:<36} {stage(c):4} type={ETYPE.get(c.energyType,"?")} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}]')

print()
print('=== BENCH SNIPERS (attack text mentions bench/benched) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or not c.basic: continue
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a and a.text and ('bench' in a.text.lower()) and a.damage >= 30:
            rows.append((c, a))
            break
rows.sort(key=lambda x: x[1].damage, reverse=True)
for c,a in rows[:16]:
    print(f'  {c.cardId:5} {c.name:<36} {stage(c):4} type={ETYPE.get(c.energyType,"?")} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}] "{a.text[:45]}"')

print()
print('=== STAGE-1 EVOLUTION payoffs (dmg>=130, any type) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or not c.stage1: continue
    a = best_atk(c)
    if a and a.damage >= 130:
        pre = next((x for x in cards.values() if x.name == c.evolvesFrom), None)
        rows.append((c, a, pre))
rows.sort(key=lambda x: x[1].damage, reverse=True)
for c,a,pre in rows[:16]:
    prename = pre.name if pre else c.evolvesFrom
    print(f'  {c.cardId:5} {c.name:<32} {stage(c):4} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}] <- {prename}({pre.cardId if pre else "?"})')

print()
print('=== HIGH-HP WALLS single-prize (hp>=160, any type) ===')
rows = []
for c in cards.values():
    if c.cardType != 0 or not c.basic or c.ex: continue
    a = best_atk(c)
    if c.hp >= 160 and a and a.damage > 0:
        rows.append((c, a))
rows.sort(key=lambda x: x[0].hp, reverse=True)
for c,a in rows[:16]:
    print(f'  {c.cardId:5} {c.name:<36} {stage(c):4} type={ETYPE.get(c.energyType,"?")} hp={c.hp:<4} {a.damage}dmg [{cost_str(a.energies)}] retreat={c.retreatCost}')
