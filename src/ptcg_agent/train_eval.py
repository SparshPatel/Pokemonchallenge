"""Offline tuner for the planner's leaf-evaluation weights (``planner.EVAL``).

Depth/beam A/B tests showed the planner's ceiling is set by *evaluation quality*,
not search quantity (a wider/deeper beam was neutral). So the highest-leverage
offline knob is the leaf-eval weight vector that ranks the searched lines.

This is the same Cross-Entropy Method optimizer used for the rule weights in
:mod:`ptcg_agent.train`, but the fitness is a **paired planner-vs-planner mirror
match**: a candidate eval vector pilots the planner against the *baseline* eval
vector on the same deck (first turn swapped each game). Deck strength and the
search machinery cancel, so the win rate isolates eval quality — the low-variance
signal we validated with ``tools/probe_h2h.py``.

The tuned vector is written as JSON and baked into the submission by pointing
``PTCG_EVAL_WEIGHTS`` at it (or copying it into ``planner.EVAL``).

Usage::

    python -m ptcg_agent.train_eval --gens 12 --pop 12 --decks 4 --games 8 \
        --val-decks 8 --val-games 10 --think 0.10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# Make the bundled runtime agent package importable (planner.EVAL lives there).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SUBMISSION = os.path.join(_ROOT, "submission")
if _SUBMISSION not in sys.path:
    sys.path.insert(0, _SUBMISSION)

from agent.planner import EVAL as EVAL_DEFAULTS  # noqa: E402  baseline/champion start
from ptcg_agent import selfplay  # noqa: E402
from ptcg_agent.deckgen import DeckPool  # noqa: E402

# Tunable eval terms and their (low, high) bounds. ``win`` (terminal value) is
# held FIXED and excluded — it must dominate every heuristic term so a real
# win/loss is never traded away. Bounds bracket each shipped default generously
# (~0.25x..3x) while keeping ``prize`` the dominant scaled signal.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "prize": (60.0, 240.0),
    "opp_dmg": (20.0, 160.0),
    "setup_ko": (10.0, 130.0),
    "my_hp": (0.0, 70.0),
    "my_ready": (0.0, 60.0),
    "no_active": (80.0, 400.0),
    "bench": (0.0, 40.0),
    "energy": (0.0, 30.0),
    "opp_threat": (10.0, 160.0),
    "hand": (0.0, 12.0),
}
PARAMS = list(PARAM_BOUNDS.keys())
LOW = np.array([PARAM_BOUNDS[k][0] for k in PARAMS])
HIGH = np.array([PARAM_BOUNDS[k][1] for k in PARAMS])

# Fixed terms carried through into every candidate (not searched).
FIXED = {"win": float(EVAL_DEFAULTS["win"])}

DEFAULTS = {k: float(EVAL_DEFAULTS[k]) for k in PARAMS}


def encode(weights: dict[str, float]) -> np.ndarray:
    return np.array([float(weights[k]) for k in PARAMS])


def decode(vec: np.ndarray) -> dict[str, float]:
    d = {k: float(v) for k, v in zip(PARAMS, vec)}
    d.update(FIXED)
    return d


# --- worker (top-level so it is picklable for spawn) ----------------------
def _eval_candidate(task):
    candidate, champion, decks, games_per_deck, think = task
    return selfplay.winrate_planner_eval(candidate, champion, decks, games_per_deck, think)


def _evaluate_population(pop, champion, decks, games_per_deck, think, pool):
    tasks = [(decode(ind), champion, decks, games_per_deck, think) for ind in pop]
    if pool is None:
        return [_eval_candidate(t) for t in tasks]
    return pool.map(_eval_candidate, tasks)


def train(args) -> dict[str, float]:
    rng = np.random.default_rng(args.seed)
    deck_pool = DeckPool()

    champion = decode(encode(DEFAULTS))
    mu = encode(DEFAULTS)
    sigma = (HIGH - LOW) * args.init_sigma

    n_elite = max(2, int(round(args.pop * args.elite_frac)))
    workers = args.workers or os.cpu_count() or 1
    pool = mp.Pool(workers) if workers > 1 else None

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.splitext(out_path)[0] + "_log.jsonl"

    best_val = 0.5
    try:
        for gen in range(args.gens):
            t0 = time.time()
            decks = deck_pool.sample(args.decks, seed=args.seed + 1000 + gen)

            pop = rng.normal(mu, sigma, size=(args.pop, len(PARAMS)))
            pop = np.clip(pop, LOW, HIGH)
            pop[0] = mu  # elitism: keep the incumbent in the race

            fitness = _evaluate_population(
                pop, champion, decks, args.games, args.think, pool
            )
            order = np.argsort(fitness)[::-1]
            elites = pop[order[:n_elite]]

            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + args.sigma_floor * (HIGH - LOW)

            val_decks = deck_pool.sample(args.val_decks, seed=args.seed + 5000 + gen)
            challenger = decode(np.clip(mu, LOW, HIGH))
            vs_champ = selfplay.winrate_planner_eval(
                challenger, champion, val_decks, args.val_games, args.think
            )
            promoted = False
            if vs_champ >= args.promote_threshold:
                champion = challenger
                promoted = True

            vs_default = selfplay.winrate_planner_eval(
                champion, decode(encode(DEFAULTS)), val_decks, args.val_games, args.think
            )
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
    print(f"Eval weights written to {out_path}")
    return champion


def _save(path: str, champion: dict[str, float], rec: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(champion, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite-frac", type=float, default=0.34)
    ap.add_argument("--decks", type=int, default=4, help="decks sampled per generation")
    ap.add_argument("--games", type=int, default=8, help="games per deck per candidate")
    ap.add_argument("--val-decks", type=int, default=8)
    ap.add_argument("--val-games", type=int, default=10)
    ap.add_argument("--think", type=float, default=0.10, help="planner think seconds")
    ap.add_argument("--init-sigma", type=float, default=0.25)
    ap.add_argument("--sigma-floor", type=float, default=0.04)
    ap.add_argument("--promote-threshold", type=float, default=0.54)
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "artifacts", "eval_weights.json",
        ),
    )
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
