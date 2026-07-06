import sys
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
import cg.game as game
from submission.main import agent

deck = []
with open('submission/deck.csv') as f:
    for line in f:
        s = line.strip()
        if s and s.lstrip('-').isdigit():
            deck.append(int(s))

obs = game.battle_start(deck, deck)
if isinstance(obs, tuple): obs = obs[0]

appeared_count = 0
for i in range(80):
    state = obs.get('current') if isinstance(obs, dict) else None
    if state and isinstance(state.get('result'), int) and state.get('result') >= 0:
        print(f'Game ended at step {i}')
        break
    
    if isinstance(state, dict):
        pl = state.get('players', [])
        for pi, p in enumerate(pl):
            if not isinstance(p, dict): continue
            active = p.get('active', [])
            if active and isinstance(active[0], dict):
                appeared = active[0].get('appearThisTurn', False)
                turn = state.get('turn', 0)
                if appeared and turn and turn > 2:
                    card_id = active[0].get('id')
                    appeared_count += 1
                    print(f'Step {i}: player{pi} active={card_id} appearThisTurn=True turn={turn}')
    
    action = agent(obs)
    obs2 = game.battle_select(action)
    if isinstance(obs2, tuple): obs2 = obs2[0]
    obs = obs2

print(f'Total appearThisTurn events (turn>2): {appeared_count}')
