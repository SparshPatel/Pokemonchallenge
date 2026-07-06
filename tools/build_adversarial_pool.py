"""Build adversarial archetype deck pool targeting D3's specific weaknesses.

Archetypes built:
1.  grass_kryptonite   - Grass attackers doing ×2 to 5/7 of our Pokémon
2.  psychic_sniper     - Iron Boulder 170dmg [PC] one-shots Koraidon+Great Tusk ×2
3.  evolution_sweeper  - Finizen→Palafin ex (250dmg for 1 Water!), high-HP evolutions
4.  bench_sniper       - Bench damage to kill charging ex before they attack
5.  prize_denial       - All single-prize basics, forces 6 separate KOs from us
6.  hit_and_run        - Aggressive pivot: full attack damage + free/cheap retreat
7.  heavy_tank         - Ultra-high HP Mega ex walls
8.  mirror_fighting    - Exact copy of D3 (our own style vs ourselves)
9.  mirror_ancient_box - D1 variant (Ancient Box with Ting-Lu wall)
"""
import sys, os, csv
sys.path.insert(0,'submission'); sys.path.insert(0,'src')
import cg.api as api

OUT_DIR = 'artifacts/adversarial_pool'
os.makedirs(OUT_DIR, exist_ok=True)

cards = {c.cardId: c for c in api.all_card_data()}
atks  = {a.attackId: a for a in api.all_attack()}

ETYPE = {0:'C',1:'G',2:'R',3:'W',4:'L',5:'P',6:'F',7:'D',8:'M',9:'N',10:'A'}
# Basic Energy IDs: G=1 R=2 W=3 L=4 P=5 F=6 D=7 M=8

# Standard trainer package (24 cards)
TRAINERS = [
    1121,1121,1121,1121,   # Ultra Ball ×4
    1086,1086,1086,1086,   # Buddy-Buddy Poffin ×4
    1122,1122,1122,1122,   # Pokégear ×4
    1123,1123,1123,         # Switch ×3
    1097,1097,1097,         # Night Stretcher ×3
    1182,1182,1182,         # Boss's Orders ×3
    1119,1119,              # Energy Search ×2
    1088,                   # Prime Catcher ×1  (ACE SPEC)
]  # = 24 cards

def write_deck(name, pokemon_ids, energy_id, energy_count=None):
    """Build a legal 60-card deck. energy_count auto-fills if None."""
    trainers = TRAINERS.copy()
    deck = pokemon_ids + trainers
    if energy_count is None:
        energy_count = 60 - len(deck)
    deck += [energy_id] * energy_count
    assert len(deck) == 60, f"{name}: got {len(deck)} cards"
    # Legality checks
    from collections import Counter
    c = Counter(deck)
    ace = sum(v for k,v in c.items() if cards.get(k) and cards[k].aceSpec)
    assert ace <= 1, f"{name}: {ace} ACE specs"
    for k,v in c.items():
        card = cards.get(k)
        if card and not card.cardType == 5:  # not basic energy
            assert v <= 4, f"{name}: {cards[k].name} x{v} > 4 copies"
    path = f'{OUT_DIR}/{name}.csv'
    with open(path,'w') as f:
        for cid in deck:
            f.write(f'{cid}\n')
    # Summary
    species = {k: v for k,v in c.items() if cards.get(k) and cards[k].cardType == 0}
    prize_vals = {1:0, 2:0, 3:0}
    for k,v in species.items():
        card = cards[k]
        pv = 3 if card.megaEx else (2 if card.ex else 1)
        prize_vals[pv] += v
    print(f'  {name:<28} {len(deck)} cards | {sum(species.values())} pkmn | '
          f'ex={prize_vals[2]+prize_vals[3]} (1-prize={prize_vals[1]}) | '
          f'{energy_count} energy({ETYPE[energy_id]})')
    return path

def best_atk(c):
    best = None
    for aid in (c.attacks or []):
        a = atks.get(aid)
        if a and (best is None or a.damage > best.damage):
            best = a
    return best

print("Building adversarial archetype pool...")
print()

# ── 1. GRASS KRYPTONITE ────────────────────────────────────────────────────────
# 5/7 of our Pokémon are Grass-weak. 240 dmg → 480 effective vs our walls.
# Decidueye ex 1022: 240 dmg [GCCC] = 1 Grass + 3 any = very efficient
# Tapu Bulu 920: 220 dmg [GGCC] basic
# Iron Leaves ex 75: 180 dmg [GGC]
# Arboliva ex 404: 160 dmg [CCC] = all colorless = runs on Grass energy
write_deck('1_grass_kryptonite',
    [1022,1022,1022,1022,  # Decidueye ex ×4 (240 dmg GCCC)
      920, 920, 920, 920,  # Tapu Bulu ×4   (220 dmg GGCC)
       75,  75,  75,       # Iron Leaves ex ×3 (180 dmg GGC)
      404, 404, 404, 404,  # Arboliva ex ×4 (160 dmg CCC)
      340, 340,            # Yanmega ex ×2  (210 dmg GGGC, filler)
    ], energy_id=1)  # Grass

# ── 2. PSYCHIC SNIPER ─────────────────────────────────────────────────────────
# Iron Boulder 971: 170 dmg [PC] = 1 Psychic + 1 any
# Koraidon ex (230 HP) → 170×2 = 340 = one-shot KO with weakness!
# Great Tusk (140 HP) → 170×2 = 340 = one-shot KO with weakness!
write_deck('2_psychic_sniper',
    [ 971, 971, 971, 971,  # Iron Boulder ×4 (170 dmg [PC], 140 HP)
      431, 431, 431, 431,  # TR Mewtwo ex ×4 (160 dmg [PPC], 280 HP)
      246, 246, 246, 246,  # Espeon ex ×4    (160 dmg [PCC], 270 HP)
      813, 813, 813,       # Mismagius ex ×3 (150 dmg [PP], 260 HP)
      223, 223,            # PalossandEX ×2  (160 dmg [CCC], 280 HP)
    ], energy_id=5)  # Psychic

# ── 3. EVOLUTION SWEEPER ──────────────────────────────────────────────────────
# Palafin ex 107: Stage-1, 340 HP, 250 dmg for just [W] = 1 Water energy!
# Pre-evo: Finizen 105. Pair with Kyurem ex for Basic bench sniper.
# Also: Azumarill 315 (S1, 230 dmg [PPPP]) from Marill 53 — but different type.
# Stick to Water: Finizen/Palafin + Kyurem ex + some Water basics.
write_deck('3_evolution_sweeper',
    [ 105, 105, 105, 105,  # Finizen ×4        (pre-evo Water basic)
      107, 107, 107, 107,  # Palafin ex ×4     (S1, 340 HP, 250 dmg [W])
      509, 509, 509, 509,  # Kyurem ex ×4      (Ba, W, 230 HP, 130 dmg [WWC] + bench)
      231, 231, 231,       # Tatsugiri ex ×3   (Ba, N/fast, 100 dmg [RW])
      108, 108,            # Wellspring Ogerpon ex ×2 (Water, 100 dmg [WCC])
    ], energy_id=3)  # Water

# ── 4. BENCH SNIPER ───────────────────────────────────────────────────────────
# Strategy: kill our charging ex on the bench before they can attack.
# Flutter Mane 56: Basic P, 90 HP, 90 dmg [CCC] + put 2 damage counters on any bench Pokémon
# Mesprit 216: Basic P, 70 HP, 160 dmg [PP] (bench conditions)
# Landorus 534: Basic F, 130 HP, 110 dmg [FCC] + 10 to each bench Pokémon
# Kyurem ex 509: Ba W, 130 dmg [WWC] + 10 to each opp bench
# Use Psychic (has most bench snipers)
write_deck('4_bench_sniper',
    [  56,  56,  56,  56,  # Flutter Mane ×4 (90 dmg + 2 bench damage)
      216, 216, 216, 216,  # Mesprit ×4      (160 dmg [PP])
      971, 971, 971,       # Iron Boulder ×3 (170 dmg [PC])
      431, 431, 431,       # TR Mewtwo ex ×3 (160 dmg [PPC])
      223, 223, 223,       # PalossandEX ×3  (160 dmg [CCC], bench effects)
    ], energy_id=5)  # Psychic

# ── 5. PRIZE DENIAL (single-prize only) ──────────────────────────────────────
# All non-ex Pokémon: every KO gives us only 1 prize.
# They need 6 KOs from us; we still need 6 KOs from them.
# Best single-prize Fighting basics: Throh 531 (120 [FC]), Sawk 602 (90 [F]),
# Ting-Lu 41 (110 [FFC]), Bloodmoon Ursaluna 135 (100 [FFC])
write_deck('5_prize_denial',
    [ 531, 531, 531, 531,  # Throh ×4          (120 dmg [FC], 130 HP, 1-prize)
      602, 602, 602, 602,  # Sawk ×4           (90 dmg [F],  110 HP, 1-prize)
       41,  41,  41,  41,  # Ting-Lu ×4        (110 dmg [FFC], 140 HP, 1-prize)
      135, 135, 135, 135,  # Bloodmoon Ursaluna ×4 (100 dmg [FFC], 150 HP)
      819, 819,            # Paldean Tauros ×2 (70 dmg [FF], 130 HP)
    ], energy_id=6)  # Fighting

# ── 6. HIT AND RUN (aggressive pivot) ─────────────────────────────────────────
# Unlike our slow build-up, this deck hits hard on 2 energy then pivots out.
# Cornerstone Ogerpon ex 117: 140 dmg [FCC] retreat=1 (our own card — but we know it's good)
# Great Tusk 58: 160 dmg [FFCC], retreat=3 (heavy but hard-hitting)
# Throh 531: 120 dmg [FC] retreat=2 (cheap, agile)
# Sawk 602: 90 dmg [F] retreat=2 (cheapest attack, free pivot)
# Lots of Switch + low retreat to enable the hit-and-run
write_deck('6_hit_and_run',
    [ 117, 117, 117, 117,  # Cornerstone Ogerpon ex ×4 (140 dmg [FCC] retreat=1)
      531, 531, 531, 531,  # Throh ×4                  (120 dmg [FC]  retreat=2)
      602, 602, 602, 602,  # Sawk ×4                   (90 dmg [F]    retreat=2)
       41,  41,  41,       # Ting-Lu ×3                (110 dmg [FFC] retreat=3 — wall)
       58,  58,            # Great Tusk ×2             (160 dmg [FFCC])
    ], energy_id=6)  # Fighting

# ── 7. HEAVY TANK (ultra-high HP Mega ex) ────────────────────────────────────
# Opponent can't KO in one shot → we waste attacks → they build board.
# Mega Kangaskhan ex 756: 300 HP, 200 dmg [CCC], retreat=3 (all-colorless!)
# Mega Zygarde ex 1056: 310 HP, 200 dmg [FFF], retreat=2 (Fighting)
# Mega Hawlucha ex 886: 250 HP, 120 dmg [FFC], retreat=1 (cheap retreat + decent HP)
# Regirock ex 447: 230 HP, accel engine
write_deck('7_heavy_tank',
    [ 756, 756, 756, 756,  # Mega Kangaskhan ex ×4 (300 HP, 200 dmg [CCC])
      886, 886, 886, 886,  # Mega Hawlucha ex ×4   (250 HP, 120 dmg [FFC])
     1056,1056,            # Mega Zygarde ex ×2    (310 HP, 200 dmg [FFF])
      447, 447, 447, 447,  # Regirock ex ×4        (230 HP, accel engine)
    ], energy_id=6)  # Fighting (Kangaskhan uses CCC = any energy)

# ── 8. MIRROR FIGHTING (exact copy of D3 — our deck) ─────────────────────────
import shutil
shutil.copy('submission/deck.csv', f'{OUT_DIR}/8_mirror_d3.csv')
print(f'  {"8_mirror_d3":<28} copied from submission/deck.csv')

# ── 9. MIRROR ANCIENT BOX (D1 variant) ───────────────────────────────────────
shutil.copy('artifacts/deck_candidates/d1_tinglu_wall.csv', f'{OUT_DIR}/9_mirror_d1_ancient_box.csv')
print(f'  {"9_mirror_d1_ancient_box":<28} copied from d1_tinglu_wall.csv')

print()
print(f"Built {len(os.listdir(OUT_DIR))} archetype decks in {OUT_DIR}/")

# ── Validate all built decks ──────────────────────────────────────────────────
from collections import Counter
print()
print("Validation:")
for fname in sorted(os.listdir(OUT_DIR)):
    if not fname.endswith('.csv'): continue
    path = f'{OUT_DIR}/{fname}'
    d = [int(x) for x in open(path) if x.strip()]
    c = Counter(d)
    ace = sum(v for k,v in c.items() if cards.get(k) and cards[k].aceSpec)
    over4 = [(cards[k].name, v) for k,v in c.items()
             if cards.get(k) and not (cards[k].cardType==5 or cards[k].cardType==6) and v>4]
    status = 'OK' if len(d)==60 and ace<=1 and not over4 else 'FAIL'
    print(f'  {fname:<40} {len(d)} cards  ACE={ace}  {status}'
          + (f'  OVER4={over4}' if over4 else ''))
