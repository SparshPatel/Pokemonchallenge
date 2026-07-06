"""Print skill texts for stadium and tool cards in current deck."""
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'submission')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}

CHECK = [1156, 1251, 1174, 1123, 1097, 1238, 1122]  # Lucky Helmet, Lively Stadium, Air Balloon, Switch, Night Stretcher, Tarragon, Pokegear

for cid in CHECK:
    c = cards.get(cid)
    if c:
        skills = getattr(c, 'skills', [])
        text = skills[0].text if skills else '(no text)'
        print(f"  [{cid}] {c.name}: {text}")
