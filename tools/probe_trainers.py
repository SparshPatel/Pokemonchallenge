"""Probe trainer/item/stadium effects via the card skills field + card text from the CSV."""
import sys, csv, os
sys.path.insert(0,'submission'); sys.path.insert(0,'src')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}
CTYPE = {0:'Pokemon',1:'Item',2:'Tool',3:'Supporter',4:'Stadium',5:'BasicEnergy',6:'SpecialEnergy'}

# Load text from CSV (has Effect Explanation column)
csv_text = {}
for fname in ['data/raw/EN_Card_Data.csv']:
    if not os.path.exists(fname): continue
    with open(fname, encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row.get('Card ID','').strip())
                text = (row.get('Effect Explanation') or row.get('Move Description') or '').strip()
                if text and text != 'n/a':
                    csv_text[cid] = text
            except: pass

def show(cid):
    c = cards.get(cid)
    if not c:
        print(f"  [{cid}] NOT FOUND")
        return
    skill_text = ''
    if hasattr(c,'skills') and c.skills:
        skill_text = str(c.skills)[:120]
    csv_t = csv_text.get(cid,'')
    print(f"  [{cid:5}] {c.name:<38} {CTYPE.get(c.cardType,'?')}")
    if skill_text: print(f"          skills: {skill_text}")
    if csv_t:      print(f"          text:   {csv_t[:120]}")
    print()

# Key target IDs
TARGET_TRAINERS = {
    "KEY ITEMS/TOOLS": [1158, 1174, 1156, 1159, 1100, 1110, 1118, 1139, 1141, 1082, 1085, 1105],
    "KEY SUPPORTERS": [1227, 1212, 1238, 1197, 1213, 1193, 1208, 1237, 1215, 1234, 1224, 1186, 1196],
    "ALL STADIUMS":   [1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,
                       1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267],
}

for section, ids in TARGET_TRAINERS.items():
    print(f"\n{'='*70}")
    print(f"{section}")
    print('='*70)
    for cid in ids:
        show(cid)

# Also verify Riolu evolution linkage
print('='*70)
print("RIOLU VARIANTS (pre-evo chain for Mega Lucario ex)")
print('='*70)
for cid in [333, 974]:
    show(cid)
c = cards.get(678)
if c: print(f"  Mega Lucario ex evolvesFrom field: '{c.evolvesFrom}'")

# Also check Cynthia's Garchomp ex (mentioned as potential)
print()
print('='*70)
print("CYNTHIA'S CHAIN (Fighting, worth noting)")
print('='*70)
for cid in [379, 380, 381]:
    show(cid)
