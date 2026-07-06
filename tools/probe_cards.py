"""Comprehensive probe of all user-suggested cards + find stadiums, draw supporters, items."""
import sys
sys.path.insert(0,'submission'); sys.path.insert(0,'src')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}
atks  = {a.attackId: a for a in api.all_attack()}

ETYPE = {0:'C',1:'G',2:'R',3:'W',4:'L',5:'P',6:'F',7:'D',8:'M',9:'N',10:'A'}
CTYPE = {0:'Pokemon',1:'Item',2:'Tool',3:'Supporter',4:'Stadium',5:'BasicEnergy',6:'SpecialEnergy'}

def cost_str(e): return ''.join(ETYPE.get(x,'?') for x in e) if e else 'free'
def pv(c): return 3 if c.megaEx else (2 if c.ex else 1)
def stage(c):
    if c.megaEx: return 'Mega-ex'
    if c.ex:     return 'ex'
    if c.stage2: return 'S2'
    if c.stage1: return 'S1'
    if c.basic:  return 'Basic'
    return '?'

def show_card(cid, note=''):
    c = cards.get(cid)
    if c is None:
        print(f"  [{cid:5}] *** NOT FOUND IN ENGINE ***  {note}")
        return
    print(f"  [{cid:5}] {c.name:<38} {CTYPE.get(c.cardType,'?'):<12} {stage(c):<8}", end='')
    if c.cardType == 0:  # Pokemon
        print(f" hp={c.hp:<4} prize={pv(c)} retreat={c.retreatCost} weak={ETYPE.get(c.weakness,'?')} ex={c.ex} mega={c.megaEx}")
        for aid in (c.attacks or []):
            a = atks.get(aid)
            if a:
                print(f"          -> {a.name}: {a.damage}dmg [{cost_str(a.energies)}]  {(a.text or '')[:60]}")
        if c.evolvesFrom:
            pre = next((x for x in cards.values() if x.name == c.evolvesFrom), None)
            print(f"          evolves from: {c.evolvesFrom} (id={pre.cardId if pre else '?'})")
    else:
        print(f" {note}")
    print()

# ── Sheet 2: User-suggested Pokémon ────────────────────────────────────────────
print("="*80)
print("USER-SUGGESTED POKEMON (Sheet 2)")
print("="*80)
suggested = [
    (24,   "TR Kangaskhan ex"),
    (1056, "Mega Zygarde ex"),
    (44,   "Bloodmoon Ursaluna ex"),
    (116,  "Okidogi"),
    (224,  "Annihilape"),
    (437,  "Mankey (pre-evo)"),
    (438,  "Primeape (mid-evo)"),
    (678,  "Mega Lucario ex"),
    (974,  "Riolu (pre-evo)"),
    (232,  "Slaking ex"),
    (998,  "Slakoth (pre-evo)"),
    (999,  "Vigoroth (mid-evo)"),
    (251,  "Regigigas"),
    (337,  "Lugia ex"),
    (527,  "Excadrill ex"),
    (526,  "Drilbur (pre-evo)"),
]
for cid, note in suggested:
    show_card(cid, note)

# ── Draw Supporters ─────────────────────────────────────────────────────────────
print("="*80)
print("DRAW SUPPORTERS (Iono / Prof Research equivalents)")
print("="*80)
draw_kw = ['iono','research','professor','cynthia','draw','oak','lillie','hop','marnie','chip']
for c in sorted(cards.values(), key=lambda x: x.name):
    if c.cardType != 3: continue  # Supporters only
    name_l = c.name.lower()
    for kw in draw_kw:
        if kw in name_l:
            show_card(c.cardId)
            break

# ── Sheet 3 items: verify by name ───────────────────────────────────────────────
print("="*80)
print("ITEMS / TRAINERS from Sheet 3 (find by name)")
print("="*80)
item_names = ['hyper aroma','awakening drum','poke vital','energy search pro','max rod',
              'maximum belt','hero','lucky helmet','air balloon','energy retrieval',
              'energy recycler','premium power','dragon elixir',
              'tarragon','femel','cook','xerosic','machination']
for target in item_names:
    found = [(c.cardId,c.name,c.cardType) for c in cards.values()
             if target in c.name.lower()]
    if found:
        for cid,name,ct in found:
            print(f"  [{cid:5}] {name:<38} {CTYPE.get(ct,'?')}")
    else:
        print(f"  [?????] '{target}' NOT FOUND")

# ── Stadiums ─────────────────────────────────────────────────────────────────────
print()
print("="*80)
print("ALL STADIUMS in pool")
print("="*80)
for c in sorted(cards.values(), key=lambda x: x.name):
    if c.cardType == 4:
        show_card(c.cardId)

# ── All Supporters ─────────────────────────────────────────────────────────────
print("="*80)
print("ALL SUPPORTERS by name")
print("="*80)
for c in sorted(cards.values(), key=lambda x: x.name):
    if c.cardType == 3:
        print(f"  [{c.cardId:5}] {c.name}")

# ── Team-trainer synergy (Ethan, Cynthia, Rocket) ───────────────────────────────
print()
print("="*80)
print("TRAINER-THEMED POKEMON (Ethan, Cynthia, Team Rocket)")
print("="*80)
trainer_kw = ["ethan","cynthia","rocket","team rocket","giovanni","jessie","james"]
for c in sorted(cards.values(), key=lambda x: x.name):
    if c.cardType != 0: continue
    for kw in trainer_kw:
        if kw in c.name.lower():
            show_card(c.cardId)
            break
