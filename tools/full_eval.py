"""Large-pool evaluation: 1000 pool decks × 1 game each per opponent.
   At N=1000, stderr ~0.016 — pool-composition variance is eliminated.
   Usage: python tools/full_eval.py [--deck PATH] [--seed N] [--decks N] [--games N]
"""
import sys, subprocess, re, argparse, math

sys.path.insert(0, 'src')
sys.path.insert(0, 'submission')

parser = argparse.ArgumentParser()
parser.add_argument('--deck',  default='submission/deck.csv')
parser.add_argument('--seed',  type=int, default=42)
parser.add_argument('--decks', type=int, default=1000)
parser.add_argument('--games', type=int, default=1)
args = parser.parse_args()

DECK = args.deck
SEEDS = [args.seed]
N_DECKS = args.decks
N_GAMES = args.games
OPPONENTS = ['random', 'greedy', 'tempo', 'strong', 'pivot']

results = {opp: [] for opp in OPPONENTS + ['OVERALL']}

for seed in SEEDS:
    total_games = N_DECKS * N_GAMES
    print(f"Running seed={seed}  pool={N_DECKS} decks × {N_GAMES} games = {total_games} games/opponent ...", flush=True)
    out = subprocess.check_output(
        ['python', '-m', 'ptcg_agent.evaluate',
         '--mode', 'field',
         '--decks', str(N_DECKS),
         '--games', str(N_GAMES),
         '--seed',  str(seed),
         '--our-deck', DECK],
        env={**__import__('os').environ, 'PYTHONPATH': 'src;submission'},
        text=True, stderr=subprocess.STDOUT
    )
    for line in out.splitlines():
        m = re.search(r'vs (\w+)\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
        if m:
            results[m.group(1)].append((int(m.group(2)), int(m.group(3)), float(m.group(4))))
        m2 = re.search(r'OVERALL\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
        if m2:
            results['OVERALL'].append((int(m2.group(1)), int(m2.group(2)), float(m2.group(3))))

print()
print(f"=== FIELD EVALUATION  deck={DECK}  pool={N_DECKS}×{N_GAMES}g  seed={args.seed} ===")
print(f"{'Opponent':<12} {'Wins':>6} {'Games':>6} {'WinRate':>8} {'StdErr':>8}")
print("-" * 50)
for opp in OPPONENTS + ['OVERALL']:
    entries = results[opp]
    if not entries: continue
    wins  = sum(e[0] for e in entries)
    games = sum(e[1] for e in entries)
    wr    = wins / games
    se    = math.sqrt(wr * (1 - wr) / games)
    if opp == 'OVERALL':
        print("-" * 50)
    print(f"{opp:<12} {wins:>6} {games:>6} {wr:>8.3f} {se:>8.4f}")
