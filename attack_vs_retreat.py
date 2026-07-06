w_attack_base, w_dmg, w_retreat, w_danger = 380.0, 0.40, 20.0, 300.0
for dmg in [0, 50, 100, 130, 200, 270]:
    a = w_attack_base + dmg * w_dmg
    r = w_retreat + w_danger
    winner = "ATTACK" if a > r else "RETREAT"
    print(f"dmg={dmg:3d}: attack={a:.0f}  retreat_danger={r:.0f}  -> {winner}")
