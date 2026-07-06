"""Generate value-net training data via self-play with the shipped planner.

Plays mirror games (both sides = shipped deck + planner). At every MAIN decision
we record the perspective player's feature vector (agent.value_net.extract_features
computed on the live observed board). After each game, every recorded state is
labeled with that perspective's eventual outcome (win=1 / loss=0) — a Monte-Carlo
value target. Writes an .npz of (X, y) for tools/_train_valuenet.py.

To keep states less correlated we record only the FIRST main decision of each
(player, turn) rather than every micro-decision.

Run: python tools/_gen_valuenet_data.py --games 300 --think 0.05 --out artifacts/valuenet_data.npz
"""
from __future__ import annotations
import argparse, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


def read_deck(path):
    return [int(l) for l in open(path) if l.strip().lstrip("-").isdigit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--think", type=float, default=0.05)
    ap.add_argument("--deck", default=os.path.join(ROOT, "submission", "deck.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "valuenet_data.npz"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import cg.game as game
    import agent.planner as planner_mod
    from agent.adapter import extract_select, current_state
    from agent.gamedata import GameData
    from agent.value_net import extract_features, FEATURE_NAMES
    from ptcg_agent.selfplay import WeightedAgent

    gd = GameData.load()
    deck = read_deck(args.deck)

    X_rows, y_rows = [], []
    t0 = time.perf_counter()
    n_states = 0
    wins = 0

    for g in range(args.games):
        agents = {
            0: WeightedAgent(deck, None, use_planner=True, planner_think_s=args.think),
            1: WeightedAgent(deck, None, use_planner=True, planner_think_s=args.think),
        }
        try:
            obs, _ = game.battle_start(deck, deck)
        except Exception as e:
            print(f"battle_start failed: {e}", file=sys.stderr)
            continue

        # per-game buffer: list of (perspective_index, feature_vector)
        game_rows = []
        seen_turns = set()  # (player, turnCounter) already recorded
        winner = -1
        for _ in range(5000):
            if not isinstance(obs, dict):
                break
            state = current_state(obs)
            if not isinstance(state, dict):
                break
            yi = state.get("yourIndex")
            yi = int(yi) if isinstance(yi, int) else 0
            result = state.get("result")
            if isinstance(result, int) and result in (0, 1):
                winner = result
                break

            sel = extract_select(obs)
            # Record features at the first MAIN decision of each player-turn.
            if sel is not None and sel.select_type == 0:
                turn = state.get("turn") or state.get("turnCount") or len(game_rows)
                key = (yi, turn)
                if key not in seen_turns:
                    seen_turns.add(key)
                    try:
                        feats = extract_features(state, yi, gd, planner_mod)
                        game_rows.append((yi, feats))
                    except Exception:
                        pass

            try:
                choice = agents[yi].agent(obs)
            except Exception:
                choice = [0]
            try:
                obs = game.battle_select(choice)
            except Exception as e:
                print(f"battle_select failed: {e}", file=sys.stderr)
                break

        try:
            game.battle_finish()
        except Exception:
            pass

        if winner in (0, 1):
            if winner == 0:
                wins += 1
            for persp, feats in game_rows:
                X_rows.append(feats)
                y_rows.append(1.0 if persp == winner else 0.0)
            n_states += len(game_rows)

        if (g + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  game {g+1}/{args.games}  states={n_states}  "
                  f"{el/(g+1):.2f}s/game", flush=True)

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"\nDONE: {X.shape[0]} states x {X.shape[1]} feats -> {args.out}")
    print(f"label balance: win={y.mean():.3f}  (p0 game winrate={wins/max(1,args.games):.3f})")


if __name__ == "__main__":
    main()
