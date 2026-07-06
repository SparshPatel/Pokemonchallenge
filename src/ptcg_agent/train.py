"""Offline weight training via self-play cross-entropy / evolutionary RL.

This is a **one-time, offline** activity. The cabt engine is CPU-bound C++ (a
GPU cannot accelerate it), and the runtime agent must be stdlib-only, so we do
*not* train a neural network. Instead we optimize the ~16 decision weights of the
heuristic policy with the **Cross-Entropy Method** — a gradient-free,
noise-robust, RL-family optimizer (cf. OpenAI's "Evolution Strategies as a
Scalable Alternative to Reinforcement Learning"). The learned weights are frozen
into ``submission/agent/trained_weights.json``, which the stdlib runtime reads.

Reward / objective
-------------------
Fitness of a candidate weight vector is its **win rate vs the current champion**
in mirror self-play (same deck both sides) across a sample of generated decks.
A win is +1, a loss is 0 — i.e. losses are directly penalized — and averaging
over many decks/games turns the noisy, unseeded engine into a usable signal.
Training on a *pool of decks* (not just our one list) tunes the policy to be
deck-agnostic and robust.

Scaling
-------
The bottleneck is simulating games, so we parallelize candidate evaluations
across CPU cores with ``multiprocessing``. Run it on a many-core Kaggle/Cloud CPU
instance for the heavy runs.

Usage::

    python -m ptcg_agent.train --gens 20 --pop 16 --decks 8 --games 12 \
        --val-decks 16 --val-games 12 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ptcg_agent import selfplay
from ptcg_agent.deckgen import DeckPool

# Tunable parameters and their (low, high) bounds. ``lethal_base`` is held FIXED
# (not searched) so a lethal KO always dominates every other action — an
# invariant we never want the optimizer to break.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "attack_base": (40.0, 170.0),
    "attack_dmg_scale": (0.0, 0.6),
    "ability": (120.0, 420.0),
    "evolve": (120.0, 380.0),
    "attach_active": (110.0, 340.0),
    "attach_bench": (70.0, 280.0),
    "attach_powered": (50.0, 230.0),
    "play_basic_room": (70.0, 280.0),
    "play_basic_noroom": (0.0, 130.0),
    "play_supporter": (50.0, 260.0),
    "play_item": (50.0, 240.0),
    "play_other": (50.0, 240.0),
    "retreat": (0.0, 130.0),
    "retreat_danger": (0.0, 320.0),
    "attach_completes": (0.0, 130.0),
    "play_basic_empty": (0.0, 40.0),
    "gust_ko": (0.0, 400.0),
    "gust_ex": (0.0, 250.0),
    "discard": (0.0, 70.0),
    "end": (-220.0, -20.0),
    "prefer_first": (0.0, 1.0),
}
PARAMS = list(PARAM_BOUNDS.keys())
LOW = np.array([PARAM_BOUNDS[k][0] for k in PARAMS])
HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAMS])

# Hand-tuned defaults = the current shipped policy = our starting point / champion.
DEFAULTS = {
    "attack_base": 95.0, "attack_dmg_scale": 0.05, "ability": 300.0,
    "evolve": 250.0, "attach_active": 200.0, "attach_bench": 140.0,
    "attach_powered": 130.0, "play_basic_room": 150.0, "play_basic_noroom": 30.0,
    "play_supporter": 120.0, "play_item": 110.0, "play_other": 130.0,
    "retreat": 20.0, "retreat_danger": 130.0, "attach_completes": 40.0,
    "play_basic_empty": 10.0, "gust_ko": 250.0, "gust_ex": 120.0,
    "discard": 10.0, "end": -100.0, "prefer_first": 0.0,
}


def encode(weights: dict[str, float]) -> np.ndarray:
    return np.array([float(weights[k]) for k in PARAMS])


def decode(vec: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(PARAMS, vec)}


# --- worker (top-level so it is picklable for spawn) ----------------------
def _eval_candidate(task):
    """Win rate of ``candidate`` vs ``champion`` over the given decks."""
    candidate, champion, decks, games_per_deck = task
    return selfplay.winrate(candidate, champion, decks, games_per_deck)


def _evaluate_population(pop, champion, decks, games_per_deck, pool):
    tasks = [(decode(ind), champion, decks, games_per_deck) for ind in pop]
    if pool is None:
        return [_eval_candidate(t) for t in tasks]
    return pool.map(_eval_candidate, tasks)


def train(args) -> dict[str, float]:
    rng = np.random.default_rng(args.seed)
    deck_pool = DeckPool()

    champion = dict(DEFAULTS)
    mu = encode(DEFAULTS)
    sigma = (HIGH - LOW) * args.init_sigma

    n_elite = max(2, int(round(args.pop * args.elite_frac)))
    workers = args.workers or os.cpu_count() or 1
    pool = mp.Pool(workers) if workers > 1 else None

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.splitext(out_path)[0] + "_log.jsonl"

    best_val = 0.5  # champion vs DEFAULT baseline (starts equal to itself)
    history = []
    try:
        for gen in range(args.gens):
            t0 = time.time()
            # Fresh decks each generation → guards against deck overfitting.
            decks = deck_pool.sample(args.decks, seed=args.seed + 1000 + gen)

            # Sample, clip, and always include the champion (elitism).
            pop = rng.normal(mu, sigma, size=(args.pop, len(PARAMS)))
            pop = np.clip(pop, LOW, HIGH)
            pop[0] = mu  # keep the incumbent in the race

            fitness = _evaluate_population(pop, champion, decks, args.games, pool)
            order = np.argsort(fitness)[::-1]
            elites = pop[order[:n_elite]]

            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + args.sigma_floor * (HIGH - LOW)

            # Promotion: validate the new mean vs the champion on a larger,
            # independent sample. Only promote on a real (margin) improvement.
            val_decks = deck_pool.sample(args.val_decks, seed=args.seed + 5000 + gen)
            challenger = decode(np.clip(mu, LOW, HIGH))
            vs_champ = selfplay.winrate(challenger, champion, val_decks, args.val_games)
            promoted = False
            if vs_champ >= args.promote_threshold:
                champion = challenger
                promoted = True

            # Absolute yardstick: champion vs the shipped DEFAULT policy.
            vs_default = selfplay.winrate(champion, DEFAULTS, val_decks, args.val_games)
            if vs_default > best_val:
                best_val = vs_default

            dt = time.time() - t0
            rec = {
                "gen": gen, "best_fitness": float(max(fitness)),
                "mean_fitness": float(np.mean(fitness)),
                "challenger_vs_champion": float(vs_champ),
                "champion_vs_default": float(vs_default),
                "promoted": promoted, "seconds": round(dt, 1),
            }
            history.append(rec)
            print(
                f"gen {gen:3d} | bestfit {rec['best_fitness']:.3f} "
                f"| chal-vs-champ {vs_champ:.3f} {'PROMOTED' if promoted else ''} "
                f"| champ-vs-default {vs_default:.3f} | {dt:.1f}s"
            )
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            _save(out_path, champion, rec)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDone. Champion vs DEFAULT (best seen): {best_val:.3f}")
    print(f"Weights written to {out_path}")
    return champion


def _save(path: str, champion: dict[str, float], rec: dict) -> None:
    payload = dict(champion)
    payload["lethal_base"] = 10_000.0  # carried through for completeness
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--decks", type=int, default=8, help="decks sampled per generation")
    ap.add_argument("--games", type=int, default=12, help="games per deck per candidate")
    ap.add_argument("--val-decks", type=int, default=16)
    ap.add_argument("--val-games", type=int, default=12)
    ap.add_argument("--init-sigma", type=float, default=0.22, help="initial std as fraction of range")
    ap.add_argument("--sigma-floor", type=float, default=0.03)
    ap.add_argument("--promote-threshold", type=float, default=0.53)
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "artifacts", "trained_weights.json",
        ),
    )
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
