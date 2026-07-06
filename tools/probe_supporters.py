"""Print all attributes of key supporter/item cards to find draw effects."""
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'submission')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}

DRAW_CANDS = [1224, 1213, 1221, 1231, 1235, 1222, 1225, 1214, 1185, 1208, 1240,
              1194, 1234, 1190, 1201, 1237, 1199, 1210, 1211, 1212, 1215]
ALL_ITEMS = [cid for cid, c in cards.items() if getattr(c, 'trainerType', -1) == 1]

print("=== KEY SUPPORTERS ===")
for cid in DRAW_CANDS:
    c = cards.get(cid)
    if c is None:
        print(f"  [{cid}] NOT FOUND")
        continue
    attrs = {k: v for k, v in vars(c).items() if v and k not in ('cardId',)}
    print(f"  [{cid}] {c.name}")
    for k, v in attrs.items():
        if k != 'name':
            print(f"       {k}={v!r}")
    print()

print("=== ALL ITEMS ===")
for cid in sorted(ALL_ITEMS):
    c = cards[cid]
    print(f"  [{cid}] {c.name}")
    for k, v in vars(c).items():
        if v and k not in ('cardId', 'name'):
            print(f"       {k}={v!r}")
    print()
