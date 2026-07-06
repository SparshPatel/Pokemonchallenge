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

# Play many actions to get a discard pile
for i in range(60):
    state = obs.get('current') if isinstance(obs, dict) else None
    if state and isinstance(state.get('result'), int) and state.get('result') >= 0:
        print(f"Game ended at step {i}")
        break
    action = agent(obs)
    obs2 = game.battle_select(action)
    if isinstance(obs2, tuple): obs2 = obs2[0]
    obs = obs2
    state = obs.get('current') if isinstance(obs, dict) else None
    if state:
        pl = state.get('players', [])
        if pl and isinstance(pl[0], dict):
            discard = pl[0].get('discard', [])
            if len(discard) > 0:
                print(f'Step {i}: discard len={len(discard)}, discard[0]={discard[0]!r}')
                print(f'Turn={state.get("turn")}, energyAttached={state.get("energyAttached")}')
                # Check turn 
                if i > 5:
                    print(f'Full discard sample: {discard[:5]}')
                    break
