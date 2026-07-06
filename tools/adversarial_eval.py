"""Evaluate D3 (or any deck) directly against all adversarial archetype decks.

Each archetype is piloted by ALL 5 available agents (random/greedy/tempo/strong/pivot).
That gives the full picture: deck quality AND agent-matchup breakdown.

Usage:
    python tools/adversarial_eval.py [--our-deck PATH] [--games N] [--seed N]
"""
import sys, os, re, subprocess, math, argparse

sys.path.insert(0, 'src')
sys.path.insert(0, 'submission')

parser = argparse.ArgumentParser()
parser.add_argument('--our-deck', default='submission/deck.csv')
parser.add_argument('--games', type=int, default=100)
parser.add_argument('--seed',  type=int, default=42)
args = parser.parse_args()

POOL_DIR = 'artifacts/adversarial_pool'
OUR_DECK = args.our_deck
GAMES    = args.games
SEED     = args.seed
AGENTS   = ['random', 'greedy', 'tempo', 'strong', 'pivot']

# Load archetype deck paths
archetypes = sorted(
    f for f in os.listdir(POOL_DIR) if f.endswith('.csv')
)

print(f"=== ADVERSARIAL EVALUATION ===")
print(f"Our deck : {OUR_DECK}")
print(f"Pool dir : {POOL_DIR}  ({len(archetypes)} archetypes)")
print(f"Agents   : {AGENTS}")
print(f"Games    : {GAMES} per archetype × agent combo  (seed={SEED})")
print(f"Total    : {GAMES * len(archetypes)} games/agent  |  "
      f"{GAMES * len(archetypes) * len(AGENTS)} total")
print()

# Each row: archetype → {agent → (wins, games)}
rows = {}
for arch in archetypes:
    arch_path = f'{POOL_DIR}/{arch}'
    arch_name = arch.replace('.csv','')
    rows[arch_name] = {}
    print(f"  {arch_name} ...", end='', flush=True)

    # Run field eval with this ONE deck repeated GAMES times = pool of GAMES identical decks
    # Use --decks=GAMES, --games=1 and point pool at just this deck
    # Simpler: use run_match_lowlevel via inline python to test our deck vs this arch deck
    # piloted by each agent. That mirrors how we tested pivot_wall.
    out = subprocess.check_output(
        ['python', '-c', f"""
import sys, os
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
from ptcg_agent.harness import _load_agent_module, run_match_lowlevel
from ptcg_agent.evaluate import _our_agent, _read_deck_file, _ModuleOpponent
import random

our_deck = _read_deck_file(r'{OUR_DECK}')
opp_deck = _read_deck_file(r'{arch_path}')
agents = {{'random':'baselines/random_agent','greedy':'baselines/greedy_agent',
           'tempo':'baselines/tempo_agent','strong':'baselines/strong_agent',
           'pivot':'baselines/pivot_wall'}}
results = {{}}
rng = random.Random({SEED})
for name, path in agents.items():
    mod = _load_agent_module(path)
    our = _our_agent(our_deck)
    opp = _ModuleOpponent(mod, opp_deck)
    r   = run_match_lowlevel(our, opp, our_deck, opp_deck, {GAMES}, swap_each=True)
    results[name] = (r.wins_a, r.games)
    print(f"{{name}} {{r.wins_a}}/{{r.games}}")
"""],
        env={**os.environ, 'PYTHONPATH': 'src;submission'},
        text=True, stderr=subprocess.STDOUT
    )
    for line in out.splitlines():
        m = re.match(r'(\w+) (\d+)/(\d+)', line.strip())
        if m:
            rows[arch_name][m.group(1)] = (int(m.group(2)), int(m.group(3)))
    # Print quick summary
    overall_wins  = sum(v[0] for v in rows[arch_name].values())
    overall_games = sum(v[1] for v in rows[arch_name].values())
    wr = overall_wins / overall_games if overall_games else 0
    print(f" overall={wr:.3f}")

# ── Report ──────────────────────────────────────────────────────────────────
print()
print("=" * 115)
header = f"{'Archetype':<32} {'OVERALL':>8} {'StdErr':>7}"
for ag in AGENTS:
    header += f" {ag:>8}"
print(header)
print("=" * 115)

grand_wins = grand_games = 0
arch_totals = []
for arch_name, ag_results in rows.items():
    wins  = sum(v[0] for v in ag_results.values())
    games = sum(v[1] for v in ag_results.values())
    wr    = wins/games if games else 0
    se    = math.sqrt(wr*(1-wr)/games) if games else 0
    grand_wins  += wins
    grand_games += games
    line = f"{arch_name:<32} {wr:>8.3f} {se:>7.4f}"
    for ag in AGENTS:
        w, g = ag_results.get(ag, (0,1))
        line += f" {w/g:>8.3f}"
    arch_totals.append((arch_name, wr, wins, games))
    print(line)

print("=" * 115)
gwr = grand_wins/grand_games if grand_games else 0
gse = math.sqrt(gwr*(1-gwr)/grand_games) if grand_games else 0
print(f"{'GRAND TOTAL':<32} {gwr:>8.3f} {gse:>7.4f}")
print()
print("RANKING (worst matchup first — these are the threats):")
arch_totals.sort(key=lambda x: x[1])
for i,(name,wr,w,g) in enumerate(arch_totals,1):
    bar = '█' * int(wr*30)
    print(f"  #{i} {name:<32} {wr:.3f}  {bar}")
