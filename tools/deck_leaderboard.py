"""Run all deck candidates through the 1000-pool evaluator and rank them.
   Usage: python tools/deck_leaderboard.py [--seed N] [--decks N] [--games N]
"""
import sys, subprocess, re, argparse, math, os

sys.path.insert(0, 'src')
sys.path.insert(0, 'submission')

parser = argparse.ArgumentParser()
parser.add_argument('--seed',  type=int, default=42)
parser.add_argument('--decks', type=int, default=1000)
parser.add_argument('--games', type=int, default=1)
args = parser.parse_args()

OPPONENTS = ['random', 'greedy', 'tempo', 'strong', 'pivot']

# All decks to evaluate: label -> path
DECKS = {
    # Main lineage (our best candidates)
    'baseline': 'artifacts/deck_candidates/baseline_ancient_box.csv',
    'D1_TingLu': 'artifacts/deck_candidates/d1_tinglu_wall.csv',
    'D2_TwoWalls': 'artifacts/deck_candidates/d2_two_walls.csv',
    'D3_GreatTusk': 'submission/deck.csv',
    'D4_Throh': 'artifacts/deck_candidates/d4_lowex_throh.csv',
    'v1_Stonjourner': 'artifacts/deck_candidates/v1_stonjourner.csv',
    'v3_Cheren': 'artifacts/deck_candidates/v3_cheren_draw.csv',
    # Type bake-off
    'mono_fighting': 'artifacts/deck_candidates/types/mono_fighting.csv',
    'mono_psychic': 'artifacts/deck_candidates/types/mono_psychic.csv',
    'mono_darkness': 'artifacts/deck_candidates/types/mono_darkness.csv',
    'mono_water': 'artifacts/deck_candidates/types/mono_water.csv',
    'mono_lightning': 'artifacts/deck_candidates/types/mono_lightning.csv',
    'mono_metal': 'artifacts/deck_candidates/types/mono_metal.csv',
    'mono_grass': 'artifacts/deck_candidates/types/mono_grass.csv',
    'mono_fire': 'artifacts/deck_candidates/types/mono_fire.csv',
    'fusion_L_M': 'artifacts/deck_candidates/types/fusion_lightning_metal.csv',
    'fusion_W_L': 'artifacts/deck_candidates/types/fusion_water_lightning.csv',
    'fusion_R_F': 'artifacts/deck_candidates/types/fusion_fire_fighting.csv',
    'fusion_P_D': 'artifacts/deck_candidates/types/fusion_psychic_darkness.csv',
}

# Filter to only existing files
DECKS = {k: v for k, v in DECKS.items() if os.path.exists(v)}

print(f"Evaluating {len(DECKS)} decks  pool={args.decks}×{args.games}g  seed={args.seed}")
print(f"Total games per deck: {args.decks * args.games * len(OPPONENTS)}")
print()

rows = []
for label, path in DECKS.items():
    print(f"  {label:<20} ...", end='', flush=True)
    try:
        out = subprocess.check_output(
            ['python', '-m', 'ptcg_agent.evaluate',
             '--mode', 'field',
             '--decks', str(args.decks),
             '--games', str(args.games),
             '--seed',  str(args.seed),
             '--our-deck', path],
            env={**os.environ, 'PYTHONPATH': 'src;submission'},
            text=True, stderr=subprocess.STDOUT
        )
        opp_rates = {}
        for line in out.splitlines():
            m = re.search(r'vs (\w+)\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
            if m:
                opp_rates[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            m2 = re.search(r'OVERALL\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)', line)
            if m2:
                opp_rates['OVERALL'] = (int(m2.group(1)), int(m2.group(2)), float(m2.group(3)))
        wr = opp_rates.get('OVERALL', (0,1,0))[2]
        wins  = opp_rates.get('OVERALL', (0,1,0))[0]
        games = opp_rates.get('OVERALL', (0,1,0))[1]
        se    = math.sqrt(wr*(1-wr)/games) if games > 0 else 0
        per_opp = {o: opp_rates.get(o, (0,1,0))[2] for o in OPPONENTS}
        rows.append((label, wr, se, wins, games, per_opp))
        print(f" {wr:.3f} ±{se:.3f}")
    except Exception as e:
        print(f" ERROR: {e}")

# Sort by overall win rate descending
rows.sort(key=lambda x: x[1], reverse=True)

print()
print("=" * 110)
print(f"{'RANK':<5} {'Deck':<22} {'WinRate':>8} {'StdErr':>7} {'random':>8} {'greedy':>8} {'tempo':>8} {'strong':>8} {'pivot':>8}")
print("=" * 110)
for i, (label, wr, se, wins, games, per_opp) in enumerate(rows, 1):
    marker = " <<< D3 (current best)" if label == 'D3_GreatTusk' else ""
    print(f"#{i:<4} {label:<22} {wr:>8.3f} {se:>7.4f} "
          f"{per_opp.get('random',0):>8.3f} {per_opp.get('greedy',0):>8.3f} "
          f"{per_opp.get('tempo',0):>8.3f} {per_opp.get('strong',0):>8.3f} "
          f"{per_opp.get('pivot',0):>8.3f}{marker}")
print("=" * 110)
print(f"\nSeed={args.seed}  Pool={args.decks} decks × {args.games} game each  StdErr(overall) ~{rows[0][2]:.4f}")
