import sys
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
from agent.gamedata import GameData
from agent.rules import _card_value
import cg.api as api

gd = GameData.load()
cards = {c.cardId: c for c in api.all_card_data()}

# Simulate Ultra Ball discard: pick 2 lowest value cards to discard
hand_cards = [333, 1182, 6, 1086, 1097, 1227, 979]  # Riolu, Boss, F Energy, Poffin, Stretcher, Lillie, Koraidon
for cid in hand_cards:
    v = _card_value(cid, gd)
    name = cards[cid].name if cid in cards else str(cid)
    print(f'id={cid} {name}: value={v:.1f}')

print('\nDiscard order (lowest first):')
sorted_vals = sorted(hand_cards, key=lambda cid: _card_value(cid, gd))
for cid in sorted_vals:
    name = cards[cid].name if cid in cards else str(cid)
    print(f'  id={cid} {name}: {_card_value(cid, gd):.1f}')

print('\nWe discard (lowest 2):', [cards[c].name for c in sorted_vals[:2]])
