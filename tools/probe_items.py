"""Print card skill texts for key items and special supporters."""
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'submission')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}

ITEMS = [1085, 1100, 1110, 1118, 1139, 1141, 1105, 1079, 1086, 1088, 1097, 1121, 1122, 1123, 1130]
SUPPORTERS_EXTRA = [1227, 1182, 1238, 1199, 1221, 1224, 1235, 1208, 1214, 1213]

print("=== ITEMS ===")
for cid in ITEMS:
    c = cards.get(cid)
    if c:
        skills = getattr(c, 'skills', [])
        text = skills[0].text if skills else '(no text)'
        print(f"  [{cid}] {c.name}: {text}")
    else:
        print(f"  [{cid}] NOT FOUND")

print()
print("=== SUPPORTERS (extended) ===")
for cid in SUPPORTERS_EXTRA:
    c = cards.get(cid)
    if c:
        skills = getattr(c, 'skills', [])
        text = skills[0].text if skills else '(no text)'
        print(f"  [{cid}] {c.name}: {text}")
    else:
        print(f"  [{cid}] NOT FOUND")
