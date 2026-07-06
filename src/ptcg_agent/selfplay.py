"""Self-play evaluation for offline weight training.

Two weight vectors pilot the **same** deck against each other (a *mirror*
match), so deck strength cancels out and the win rate isolates *play skill* —
exactly the signal we want when tuning a deck-agnostic policy. The reward is the
game outcome (win / loss); aggregated over many decks and games it becomes a
low-variance fitness for the cross-entropy / evolutionary optimizer in
:mod:`ptcg_agent.train`.

The runtime agent code in ``submission/agent`` is reused verbatim — we only
inject weights via :func:`agent.rules.set_weights` immediately before each
decision, which is safe because the engine queries one player at a time.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SUBMISSION = os.path.join(_ROOT, "submission")
# Make the bundled runtime agent package *and* the cabt ``cg`` engine importable
# in this process (and in spawned worker processes, which re-import this module).
if _SUBMISSION not in sys.path:
    sys.path.insert(0, _SUBMISSION)

from agent import rules  # noqa: E402
from agent.adapter import extract_select, is_deck_phase  # noqa: E402
from agent.gamedata import GameData  # noqa: E402
from ptcg_agent.harness import run_match_lowlevel  # noqa: E402


class WeightedAgent:
    """Runtime heuristic agent piloted with a specific weight vector.

    When ``use_planner`` is set (or ``PTCG_ENABLE_PLANNER=1``), MAIN turn
    decisions are first routed through the engine-simulated :class:`TurnPlanner`
    (falling back to the weighted rule policy on any miss), so the offline
    gauntlet can A/B the planner against pure rules on identical decks/opponents.
    """

    def __init__(
        self,
        deck: list[int],
        weights: dict[str, float] | None,
        rules_module=None,
        use_planner: bool | None = None,
        planner_think_s: float = 0.15,
        eval_weights: dict[str, float] | None = None,
        use_value_net: bool | None = None,
        value_net_path: str | None = None,
        max_nodes: int = 300,
    ):
        self.deck = list(deck)
        self.weights = dict(weights) if weights else {}
        self.gd = GameData.load()
        # Allow piloting with an alternate rules implementation (e.g. an
        # experimental ``agent.rules_seq``) without touching the shipped agent.
        self.rules = rules_module or rules
        if use_planner is None:
            use_planner = os.environ.get("PTCG_ENABLE_PLANNER", "0") == "1"
        self.use_planner = use_planner
        self.planner_think_s = float(
            os.environ.get("PTCG_PLANNER_THINK_S", planner_think_s)
        )
        self.eval_weights = dict(eval_weights) if eval_weights else None
        self.use_value_net = use_value_net
        self.value_net_path = value_net_path
        self.max_nodes = int(max_nodes)
        self._planner = None
        self._planner_tried = False

    def _maybe_planner(self):
        if not self.use_planner:
            return None
        if self._planner_tried:
            return self._planner
        self._planner_tried = True
        try:
            from agent.planner import TurnPlanner

            planner = TurnPlanner(
                None,
                gamedata=self.gd,
                your_deck_ids=self.deck,
                max_think_s=self.planner_think_s,
                eval_weights=self.eval_weights,
                use_value_net=self.use_value_net,
                value_net_path=self.value_net_path,
                max_nodes=self.max_nodes,
            )
            if planner.available():
                self._planner = planner
        except Exception:
            self._planner = None
        return self._planner

    def agent(self, obs) -> list[int]:
        try:
            if is_deck_phase(obs):
                return list(self.deck)
            sel = extract_select(obs)
            if sel is None:
                return list(self.deck)
            self.rules.set_weights(self.weights)  # sequential => safe in self-play
            planner = self._maybe_planner()
            if planner is not None:
                try:
                    import time

                    choice = planner.choose(obs, sel, time.monotonic() + 8.0)
                    if choice and self._valid(choice, sel):
                        return choice
                except Exception:
                    pass  # fall through to the weighted rule policy
            choice = self.rules.choose(obs, sel, self.gd)
            if self._valid(choice, sel):
                return choice
            return self._fallback(sel)
        except Exception:
            return [0]

    @staticmethod
    def _valid(choice, sel) -> bool:
        if not isinstance(choice, list) or len(choice) != len(set(choice)):
            return False
        n = len(sel.options)
        if not (sel.min_count <= len(choice) <= sel.max_count):
            return False
        return all(isinstance(i, int) and 0 <= i < n for i in choice)

    @staticmethod
    def _fallback(sel) -> list[int]:
        n = len(sel.options)
        k = max(0, min(sel.min_count, n))
        return list(range(k))


def play_mirror(
    weights_a: dict[str, float] | None,
    weights_b: dict[str, float] | None,
    decks: list[list[int]],
    games_per_deck: int,
) -> tuple[int, int, int]:
    """Play ``weights_a`` vs ``weights_b`` on each deck (mirror).

    Returns ``(wins_a, games, errors)``. ``swap_each`` alternates who takes the
    first turn so the going-first advantage is balanced across the sample.
    """
    wins_a = 0
    games = 0
    errors = 0
    for deck in decks:
        a = WeightedAgent(deck, weights_a)
        b = WeightedAgent(deck, weights_b)
        res = run_match_lowlevel(a, b, deck, deck, games_per_deck, swap_each=True)
        wins_a += res.wins_a
        games += res.games
        errors += res.errors_a + res.errors_b
    return wins_a, games, errors


def winrate(
    weights_a: dict[str, float] | None,
    weights_b: dict[str, float] | None,
    decks: list[list[int]],
    games_per_deck: int,
) -> float:
    wins_a, games, _ = play_mirror(weights_a, weights_b, decks, games_per_deck)
    return wins_a / games if games else 0.0


def play_mirror_planner_eval(
    eval_a: dict[str, float] | None,
    eval_b: dict[str, float] | None,
    decks: list[list[int]],
    games_per_deck: int,
    think_s: float = 0.10,
) -> tuple[int, int, int]:
    """Planner(eval_a) vs planner(eval_b) on each deck (mirror, paired).

    Both sides run the engine-simulated planner and pilot the *same* deck, so the
    win rate isolates the quality of the leaf-eval weight vector — the low-
    variance signal for tuning ``planner.EVAL``. First turn is swapped each game.
    """
    wins_a = 0
    games = 0
    errors = 0
    for deck in decks:
        a = WeightedAgent(
            deck, None, use_planner=True, planner_think_s=think_s, eval_weights=eval_a
        )
        b = WeightedAgent(
            deck, None, use_planner=True, planner_think_s=think_s, eval_weights=eval_b
        )
        res = run_match_lowlevel(a, b, deck, deck, games_per_deck, swap_each=True)
        wins_a += res.wins_a
        games += res.games
        errors += res.errors_a + res.errors_b
    return wins_a, games, errors


def winrate_planner_eval(
    eval_a: dict[str, float] | None,
    eval_b: dict[str, float] | None,
    decks: list[list[int]],
    games_per_deck: int,
    think_s: float = 0.10,
) -> float:
    wins_a, games, _ = play_mirror_planner_eval(
        eval_a, eval_b, decks, games_per_deck, think_s
    )
    return wins_a / games if games else 0.0
