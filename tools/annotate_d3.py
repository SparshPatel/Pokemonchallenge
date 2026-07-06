"""Build annotated CSV of D3 deck with all card attributes."""
import sys, csv
sys.path.insert(0, 'submission')
sys.path.insert(0, 'src')
from collections import Counter
import cg.api as api

ENERGY_NAMES = {0:'Colorless',1:'Grass',2:'Fire',3:'Water',4:'Lightning',
                5:'Psychic',6:'Fighting',7:'Darkness',8:'Metal',9:'Dragon',10:'Rainbow'}

cards = {c.cardId: c for c in api.all_card_data()}
atk_info = {a.attackId: a for a in api.all_attack()}

def etype(e):
    return ENERGY_NAMES.get(e, '?') if e is not None else '-'

def prize_val(c):
    return 3 if c.megaEx else (2 if c.ex else 1)

def card_category(c):
    return {0:'Pokemon',1:'Item',2:'Tool',3:'Supporter',4:'Stadium',5:'Basic Energy',6:'Special Energy'}.get(c.cardType,'?')

deck_path = 'submission/deck.csv'
d = [int(x) for x in open(deck_path) if x.strip()]
cnt = Counter(d)

seen = []
for cid in d:
    if cid not in seen:
        seen.append(cid)

rows = []
for cid in seen:
    c = cards[cid]
    cat = card_category(c)
    is_pkmn = c.cardType == 0

    attack_strs = []
    for aid in (c.attacks or []):
        a = atk_info.get(aid)
        if not a:
            continue
        cost = ''.join(etype(e)[0] for e in a.energies) if a.energies else 'free'
        attack_strs.append(f"{a.name}: {a.damage}dmg [{cost}]  {a.text[:40] if a.text else ''}")

    rows.append({
        'count':        cnt[cid],
        'card_id':      cid,
        'name':         c.name,
        'category':     cat,
        'type':         etype(c.energyType) if is_pkmn else '-',
        'hp':           c.hp if is_pkmn else '-',
        'stage':        'Basic' if c.basic else ('Stage1' if c.stage1 else ('Stage2' if c.stage2 else '-')),
        'is_ex':        'Yes' if c.ex else 'No',
        'is_mega':      'Yes' if c.megaEx else 'No',
        'prize_value':  prize_val(c) if is_pkmn else '-',
        'retreat_cost': c.retreatCost if is_pkmn else '-',
        'weakness':     etype(c.weakness) if is_pkmn else '-',
        'resistance':   etype(c.resistance) if is_pkmn else '-',
        'attacks':      ' // '.join(attack_strs) if attack_strs else '-',
    })

out = 'artifacts/deck_candidates/d3_annotated.csv'
fields = ['count','card_id','name','category','type','hp','stage','is_ex','is_mega',
          'prize_value','retreat_cost','weakness','resistance','attacks']
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"Written: {out}")

# pretty console table
for r in rows:
    atk_short = r['attacks'][:80] if r['attacks'] != '-' else '-'
    print(f"  x{r['count']}  [{r['card_id']:4}]  {r['name']:<35} {r['category']:<14} type={r['type']:<10} hp={r['hp']:<4} prize={r['prize_value']:<2} retreat={r['retreat_cost']:<2} weak={r['weakness']:<10}")
    if r['attacks'] != '-':
        for line in r['attacks'].split(' // '):
            print(f"         -> {line}")
    print()
