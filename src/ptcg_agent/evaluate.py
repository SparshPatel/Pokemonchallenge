"""Cross-deck, multi-opponent evaluation for the submission agent.

The local *mirror* harness (:mod:`ptcg_agent.selfplay`) deliberately pits two
agents on the **same** deck to isolate play skill — which means it is blind to
two things that actually matter for the competition score:

* **Deck quality** — invisible in a mirror (both sides share it), yet it powers
  the Model score against the *external* opponent pool and is 20% of the
  Strategy score.
* **Generalization** — does our policy hold up piloting decks other than the one
  curated Ancient Box list, and against opponents of *different* play styles?

This module measures both by running our agent across a **diverse pool of decks**
(from :mod:`ptcg_agent.deckgen`) against a **panel of opponent archetypes**
(``baselines/``) using independently-assigned decks. Because the engine takes
each player's deck directly in ``battle_start`` and every agent reads the board
to play, we can deal any deck to any agent.

Modes
-----
* ``skill``  — mirror per deck (both sides pilot the *same* pool deck). Aggregated
  across the pool and panel, this is a deck-agnostic measure of play skill /
  generalization.
* ``field``  — our agent pilots a fixed deck (``submission/deck.csv`` by default,
  or ``--our-deck``) while each opponent pilots pool decks. This measures
  real-world strength against a varied field and is the right signal for
  comparing candidate decks.

Usage::

    $env:PYTHONPATH="src"
    python -m ptcg_agent.evaluate --mode skill  --decks 12 --games 8
    python -m ptcg_agent.evaluate --mode field  --decks 12 --games 8
    python -m ptcg_agent.evaluate --mode field  --our-deck path/to/candidate.csv
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SUBMISSION = os.path.join(_ROOT, "submission")
if _SUBMISSION not in sys.path:
    sys.path.insert(0, _SUBMISSION)

from ptcg_agent.deckgen import DeckPool  # noqa: E402
from ptcg_agent.harness import _load_agent_module, run_match_lowlevel  # noqa: E402
from ptcg_agent.selfplay import WeightedAgent  # noqa: E402


def _read_deck_file(path: str) -> list[int]:
    """Read a deck.csv (one Card ID per line) into a list of ints."""
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and s.lstrip("-").isdigit():
                ids.append(int(s))
    return ids

# Opponent panel: directory name under ``baselines/`` -> display label.
OPPONENTS: list[tuple[str, str]] = [
    ("random_agent", "random"),
    ("greedy_agent", "greedy"),
    ("tempo_agent", "tempo"),
    ("strong_agent", "strong"),
    ("pivot_wall", "pivot"),
]


def _opponent_dir(name: str) -> str:
    return os.path.join(_ROOT, "baselines", name)


def _load_opponents(names: list[str] | None = None):
    """Load opponent agent modules. Missing baselines are skipped, not fatal."""
    panel = []
    for dirname, label in OPPONENTS:
        if names is not None and dirname not in names and label not in names:
            continue
        path = _opponent_dir(dirname)
        if not os.path.exists(os.path.join(path, "main.py")):
            continue
        try:
            mod = _load_agent_module(path)
            panel.append((label, mod))
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip {label}: {exc})", file=sys.stderr)
    return panel


class _ModuleOpponent:
    """Adapt a baseline ``main.py`` module to pilot an *assigned* deck.

    In the low-level driver the deck passed to ``battle_start`` is authoritative,
    so the baseline never sees a deck-selection phase; it simply reads the board
    and plays. We only keep ``deck`` so the harness can hand the engine a deck.
    """

    def __init__(self, module, deck: list[int]):
        self._module = module
        self.deck = list(deck)

    def agent(self, obs):
        return self._module.agent(obs)


def _load_our_rules(name: str | None):
    """Import an alternate rules module from the bundled ``agent`` package.

    ``name`` is the module's basename (e.g. ``rules_seq`` -> ``agent.rules_seq``).
    ``None`` uses the shipped ``agent.rules``.
    """
    import importlib

    mod_name = f"agent.{name}" if name else "agent.rules"
    return importlib.import_module(mod_name)


def _our_agent(deck: list[int], rules_module=None) -> WeightedAgent:
    """Our shipped rules agent (default weights) piloting ``deck``."""
    return WeightedAgent(deck, None, rules_module=rules_module)


def evaluate_skill(decks, panel, games_per_deck, rules_module=None):
    """Mirror per deck: our agent vs each opponent, both on the same deck."""
    results = {}
    for label, mod in panel:
        wins = total = errs = 0
        for deck in decks:
            ours = _our_agent(deck, rules_module)
            opp = _ModuleOpponent(mod, deck)
            res = run_match_lowlevel(ours, opp, deck, deck, games_per_deck, swap_each=True)
            wins += res.wins_a
            total += res.games
            errs += res.errors_a + res.errors_b
        results[label] = (wins, total, errs)
    return results


def evaluate_field(our_deck, decks, panel, games_per_deck, rules_module=None):
    """Our agent on a fixed deck vs each opponent piloting the deck pool."""
    results = {}
    for label, mod in panel:
        wins = total = errs = 0
        for opp_deck in decks:
            ours = _our_agent(our_deck, rules_module)
            opp = _ModuleOpponent(mod, opp_deck)
            res = run_match_lowlevel(ours, opp, our_deck, opp_deck, games_per_deck, swap_each=True)
            wins += res.wins_a
            total += res.games
            errs += res.errors_a + res.errors_b
        results[label] = (wins, total, errs)
    return results


def _print_results(title, results):
    print(f"\n=== {title} ===")
    g_wins = g_total = g_errs = 0
    for label, (wins, total, errs) in results.items():
        wr = wins / total if total else 0.0
        print(f"  vs {label:8s}: {wins:4d}/{total:<4d} = {wr:.3f}  (errors {errs})")
        g_wins += wins
        g_total += total
        g_errs += errs
    overall = g_wins / g_total if g_total else 0.0
    print(f"  {'OVERALL':11s}: {g_wins:4d}/{g_total:<4d} = {overall:.3f}  (errors {g_errs})")
    return overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["skill", "field"], default="skill")
    ap.add_argument("--decks", type=int, default=12, help="size of the deck pool")
    ap.add_argument("--games", type=int, default=8, help="games per (deck, opponent)")
    ap.add_argument("--seed", type=int, default=7, help="deck-pool seed")
    ap.add_argument("--our-deck", default=None,
                    help="path to our deck.csv for --mode field (default submission/deck.csv)")
    ap.add_argument("--opponents", nargs="*", default=None,
                    help="restrict the panel (e.g. strong greedy)")
    ap.add_argument("--our-rules", default=None,
                    help="alternate rules module under agent/ (e.g. rules_seq)")
    args = ap.parse_args()

    pool = DeckPool()
    decks = pool.sample(args.decks, seed=args.seed)
    panel = _load_opponents(args.opponents)
    if not panel:
        print("No opponents available under baselines/.", file=sys.stderr)
        return

    rules_module = _load_our_rules(args.our_rules)
    labels = ", ".join(lbl for lbl, _ in panel)
    rules_note = f" | our-rules: {args.our_rules}" if args.our_rules else ""
    print(f"pool: {len(decks)} decks (seed {args.seed}) | panel: {labels} | "
          f"{args.games} games each{rules_note}")

    if args.mode == "skill":
        results = evaluate_skill(decks, panel, args.games, rules_module)
        _print_results("SKILL (mirror across deck pool)", results)
    else:
        our_path = args.our_deck or os.path.join(_SUBMISSION, "deck.csv")
        our_deck = _read_deck_file(our_path)
        print(f"our deck: {our_path} ({len(our_deck)} cards)")
        results = evaluate_field(our_deck, decks, panel, args.games, rules_module)
        _print_results("FIELD (our deck vs pool-piloting opponents)", results)


if __name__ == "__main__":
    main()
