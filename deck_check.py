import sys
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
import cg.game as game
deck = []
with open('submission/deck.csv') as f:
    for line in f:
        s = line.strip()
        if s and s.lstrip('-').isdigit():
            deck.append(int(s))

obs = game.battle_start(deck, deck)
if isinstance(obs, tuple): obs = obs[0]
state = obs.get('current')
if state:
    pl = state.get('players', [])
    for i, p in enumerate(pl):
        if isinstance(p, dict):
            hand_len = len(p.get('hand') or [])
            hand_count = p.get('handCount')
            deck_count = p.get('deckCount')
            print(f'Player {i}: handCount={hand_count}, deckCount={deck_count}, hand_len={hand_len}')
