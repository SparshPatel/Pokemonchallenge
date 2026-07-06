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

# Play a few actions to get an active pokemon
for i in range(30):
    state = obs.get('current') if isinstance(obs, dict) else None
    if state and isinstance(state.get('result'), int) and state.get('result') >= 0:
        break
    action = agent(obs)
    obs2 = game.battle_select(action)
    if isinstance(obs2, tuple): obs2 = obs2[0]
    obs = obs2
    state = obs.get('current') if isinstance(obs, dict) else None
    if state:
        pl = state.get('players', [])
        if pl and isinstance(pl[0], dict):
            active = pl[0].get('active', [])
            if active and isinstance(active[0], dict) and active[0].get('id'):
                print(f'Step {i}: ACTIVE FIELDS = {list(active[0].keys())}')
                print(f'Active values: {active[0]}')
                bench = pl[0].get('bench', [])
                if bench and isinstance(bench[0], dict):
                    print(f'Bench[0] fields: {list(bench[0].keys())}')
                    print(f'Bench[0] values: {bench[0]}')
                discard = pl[0].get('discard', [])
                print(f'Discard len={len(discard)}, first 3={discard[:3]}')
                print(f'Turn={state.get("turn")}, supporterPlayed={state.get("supporterPlayed")}, energyAttached={state.get("energyAttached")}')
                break
