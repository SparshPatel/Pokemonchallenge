"""Comprehensive D5 evaluation:
1. Multi-seed 1000-pool: seeds 42,123,456,789,2024 (5000 games/agent = 25000 total)
2. Mirror match: D5 vs D5 piloted by each of 5 agents (500 games each = 2500 total)
3. Trainer decks: vs baseline_default, pivot_wall, sim_sample (200g x 5 agents = 1000 each)

Usage: python tools/comprehensive_eval.py [--deck PATH]
"""
import sys, os, math, argparse, subprocess, re
sys.path.insert(0, 'src')
sys.path.insert(0, 'submission')

from ptcg_agent.harness import _load_agent_module, run_match_lowlevel
from ptcg_agent.evaluate import _our_agent, _read_deck_file, _ModuleOpponent

parser = argparse.ArgumentParser()
parser.add_argument('--deck', default='submission/deck.csv')
parser.add_argument('--out',  default='results/comprehensive_d5.txt')
args = parser.parse_args()

DECK_PATH = args.deck

# Tee all output to file AND terminal
import io
_log = open(args.out, 'w', encoding='utf-8')
class _Tee:
    def __init__(self, *streams): self._s = streams
    def write(self, d):
        for s in self._s: s.write(d)
    def flush(self):
        for s in self._s: s.flush()
sys.stdout = _Tee(sys.__stdout__, _log)
AGENTS = ['random', 'greedy', 'tempo', 'strong', 'pivot']
AGENT_PATHS = {
    'random': 'baselines/random_agent',
    'greedy': 'baselines/greedy_agent',
    'tempo':  'baselines/tempo_agent',
    'strong': 'baselines/strong_agent',
    'pivot':  'baselines/pivot_wall',
}
ENV = {**os.environ, 'PYTHONPATH': 'src;submission'}

def wr_se(wins, games):
    wr = wins / games if games > 0 else 0.0
    se = math.sqrt(wr * (1 - wr) / games) if games > 0 else 0.0
    return wr, se

def pool_eval_subprocess(deck_path, seed, n_decks=1000, n_games=1):
    """Run evaluate_field via subprocess (fast: all 5 agents in one call)."""
    out = subprocess.check_output(
        ['python', '-m', 'ptcg_agent.evaluate',
         '--mode', 'field',
         '--decks', str(n_decks),
         '--games', str(n_games),
         '--seed',  str(seed),
         '--our-deck', deck_path],
        env=ENV, text=True, stderr=subprocess.STDOUT
    )
    res = {}
    for line in out.splitlines():
        m = re.search(r'vs (\w+)\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
        if m:
            res[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        m2 = re.search(r'OVERALL\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
        if m2:
            res['OVERALL'] = (int(m2.group(1)), int(m2.group(2)))
    return res

# ─────────────────────────────────────────────────────────
# 1. MULTI-SEED 1000-POOL  (subprocess for speed)
# ─────────────────────────────────────────────────────────
print("=" * 70)
print("PART 1: MULTI-SEED 1000-POOL  (5 seeds x 1000 pool decks x 5 agents)")
print("=" * 70)
SEEDS = [42, 123, 456, 789, 2024]
pool_totals = {opp: [0, 0] for opp in AGENTS + ['OVERALL']}

for seed in SEEDS:
    print(f"\nSeed {seed}...", flush=True)
    res = pool_eval_subprocess(DECK_PATH, seed)
    for opp, (w, g) in res.items():
        if opp in pool_totals:
            pool_totals[opp][0] += w
            pool_totals[opp][1] += g
        wr, se = wr_se(w, g)
        if opp != 'OVERALL':
            print(f"  vs {opp:<8}  {w}/{g} = {wr:.3f} ±{se:.4f}")
    if 'OVERALL' in res:
        w, g = res['OVERALL']
        wr, se = wr_se(w, g)
        print(f"  OVERALL        {w}/{g} = {wr:.3f} ±{se:.4f}")

print()
print("AGGREGATED ACROSS ALL SEEDS:")
print(f"{'Opponent':<10} {'Wins':>7} {'Games':>7} {'WinRate':>9} {'StdErr':>9}")
print("-" * 50)
for opp in AGENTS:
    w, g = pool_totals[opp]
    wr, se = wr_se(w, g)
    print(f"{opp:<10} {w:>7} {g:>7} {wr:>9.4f} {se:>9.5f}")
print("-" * 50)
w, g = pool_totals['OVERALL']
wr, se = wr_se(w, g)
print(f"{'OVERALL':<10} {w:>7} {g:>7} {wr:>9.4f} {se:>9.5f}")

# Load agents for parts 2 and 3 (direct in-process calls, small game counts)
MODS = {name: _load_agent_module(path) for name, path in AGENT_PATHS.items()}
our_deck = _read_deck_file(DECK_PATH)

def run_vs(opp_deck, opp_mod, n):
    our = _our_agent(our_deck)
    opp = _ModuleOpponent(opp_mod, opp_deck)
    r = run_match_lowlevel(our, opp, our_deck, opp_deck, n, swap_each=True)
    return r

# ─────────────────────────────────────────────────────────
# 2. MIRROR MATCH: D5 vs D5
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2: MIRROR MATCH  D5 vs D5  (500 games x 5 agents = 2500 total)")
print("=" * 70)
MIRROR_N = 500
mirror_total_w, mirror_total_g = 0, 0
print(f"\n{'Agent (opp)':<12} {'Wins':>6} {'Games':>6} {'WinRate':>8} {'StdErr':>8}")
print("-" * 45)
for opp_name in AGENTS:
    mod = MODS[opp_name]
    r = run_vs(our_deck, mod, MIRROR_N)
    wr, se = wr_se(r.wins_a, r.games)
    print(f"{opp_name:<12} {r.wins_a:>6} {r.games:>6} {wr:>8.3f} {se:>8.4f}  ({r.draws}d)")
    mirror_total_w += r.wins_a; mirror_total_g += r.games
print("-" * 45)
wr, se = wr_se(mirror_total_w, mirror_total_g)
print(f"{'OVERALL':<12} {mirror_total_w:>6} {mirror_total_g:>6} {wr:>8.3f} {se:>8.4f}")

# ─────────────────────────────────────────────────────────
# 3. TRAINER DECKS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 3: TRAINER DECKS  (200 games x 5 agents = 1000 per deck)")
print("=" * 70)
TRAINER_DECKS = [
    ('baseline_default',   'baselines/greedy_agent/deck.csv'),
    ('pivot_wall',         'baselines/pivot_wall/deck.csv'),
    ('sim_sample_water',   'data/sim_sample/deck.csv'),
]
TRAINER_N = 200

for deck_name, deck_path in TRAINER_DECKS:
    trainer_deck = _read_deck_file(deck_path)
    print(f"\nvs {deck_name}:")
    print(f"  {'Agent':<10} {'Wins':>5} {'Games':>6} {'WinRate':>8} {'Draws':>6}")
    print("  " + "-" * 40)
    td_w, td_g = 0, 0
    for opp_name in AGENTS:
        mod = MODS[opp_name]
        r = run_vs(trainer_deck, mod, TRAINER_N)
        wr, se = wr_se(r.wins_a, r.games)
        print(f"  {opp_name:<10} {r.wins_a:>5} {r.games:>6} {wr:>8.3f}  {r.draws:>5}d")
        td_w += r.wins_a; td_g += r.games
    wr, se = wr_se(td_w, td_g)
    print("  " + "-" * 40)
    print(f"  {'OVERALL':<10} {td_w:>5} {td_g:>6} {wr:>8.3f} +-{se:.4f}")

print("\nDone.")
