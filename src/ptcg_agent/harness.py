"""Local battle harness for evaluating agents against each other.

Requires the bundled cabt ``cg`` package (downloaded from the Simulation
competition's sample submission and placed on ``PYTHONPATH``). Without it, the
harness reports that the engine is missing and exits cleanly — all engine calls
are guarded so importing this module never fails.

Two drivers are provided:

* ``run_match_kaggle`` — uses ``kaggle_environments.make("cabt", ...)`` and
  ``env.run([agent, agent])`` (the documented test path).
* ``run_match_lowlevel`` — drives ``cg.game.battle_start`` / ``battle_select``
  directly, reading ``obs["current"]["yourIndex"]`` to route decisions and
  ``obs["current"]["result"]`` to detect the end.

Usage::

    python -m ptcg_agent.harness --games 50 --a submission --b submission
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field


def engine_available() -> bool:
    try:
        import cg  # noqa: F401

        return True
    except Exception:
        return False


def kaggle_env_available() -> bool:
    try:
        import kaggle_environments  # noqa: F401

        return True
    except Exception:
        return False


def _load_agent_module(agent_dir: str):
    """Import ``main.py`` from an agent directory under a unique module name."""
    main_path = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(main_path)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    name = f"agent_main_{abs(hash(agent_dir))}"
    spec = importlib.util.spec_from_file_location(name, main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_deck(agent_dir: str) -> list[int]:
    path = os.path.join(agent_dir, "deck.csv")
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or not s.lstrip("-").isdigit():
                continue
            ids.append(int(s))
    return ids


@dataclass
class MatchResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    errors_a: int = 0
    errors_b: int = 0
    turns: list[int] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def winrate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    def __str__(self) -> str:
        avg_t = sum(self.turns) / len(self.turns) if self.turns else 0
        return (
            f"A wins {self.wins_a} | B wins {self.wins_b} | draws {self.draws} "
            f"| A winrate {self.winrate_a:.3f} | avg turns {avg_t:.1f} "
            f"| errors A/B {self.errors_a}/{self.errors_b}"
        )


# --- kaggle_environments driver ------------------------------------------
def run_match_kaggle(
    agent_a, agent_b, deck_a, deck_b, n_games: int, swap_each: bool = True
) -> MatchResult:
    from kaggle_environments import make

    res = MatchResult()
    for g in range(n_games):
        a_first = (g % 2 == 0) if swap_each else True
        fa, fb = (agent_a, agent_b) if a_first else (agent_b, agent_a)
        da, db = (deck_a, deck_b) if a_first else (deck_b, deck_a)

        env = make("cabt", configuration={"decks": [da, db]})
        try:
            env.run([fa.agent, fb.agent])
        except Exception as e:  # noqa: BLE001
            print(f"env.run failed: {e}", file=sys.stderr)
            res.draws += 1
            continue

        rewards = [s.get("reward") for s in env.state] if env.state else []
        outcome = _outcome_from_rewards(rewards)
        _tally(res, outcome, a_first, _count_turns(env))
    return res


def _outcome_from_rewards(rewards) -> int:
    """0 = first player wins, 1 = second wins, -1 = draw/unknown."""
    if len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
        return -1
    if rewards[0] > rewards[1]:
        return 0
    if rewards[1] > rewards[0]:
        return 1
    return -1


def _count_turns(env) -> int:
    try:
        return len(env.steps)
    except Exception:
        return 0


# --- low-level cg.game driver --------------------------------------------
def run_match_lowlevel(
    agent_a, agent_b, deck_a, deck_b, n_games: int, swap_each: bool = True
) -> MatchResult:
    import cg.game as game

    res = MatchResult()

    for g in range(n_games):
        a_first = (g % 2 == 0) if swap_each else True
        fa, fb = (agent_a, agent_b) if a_first else (agent_b, agent_a)
        da, db = (deck_a, deck_b) if a_first else (deck_b, deck_a)
        agents = {0: fa, 1: fb}

        try:
            obs, _start = game.battle_start(da, db)
        except Exception as e:  # noqa: BLE001
            print(f"battle_start failed: {e}", file=sys.stderr)
            res.draws += 1
            continue

        outcome, turns = _play_loop(game, agents, obs)
        try:
            game.battle_finish()
        except Exception:
            pass
        _tally(res, outcome, a_first, turns)

    return res


def _play_loop(game, agents, obs):
    """Drive one battle. Returns (outcome, turns).

    ``obs`` is the engine observation dict. ``current.yourIndex`` selects which
    agent decides; ``current.result`` (0/1/2) signals the end.
    """
    turns = 0
    for _ in range(5000):
        if not isinstance(obs, dict):
            return -1, turns
        state = obs.get("current")
        # Determine the acting player.
        yi = 0
        if isinstance(state, dict) and isinstance(state.get("yourIndex"), int):
            yi = state["yourIndex"]
            result = state.get("result")
            if isinstance(result, int) and result >= 0:
                return (result if result in (0, 1) else -1), turns

        try:
            choice = agents[yi].agent(obs)
        except Exception:
            choice = [0]

        try:
            obs = game.battle_select(choice)
        except Exception as e:  # noqa: BLE001
            print(f"battle_select failed: {e}", file=sys.stderr)
            return -1, turns
        turns += 1

    return -1, turns


# --- shared ---------------------------------------------------------------
def _tally(res: MatchResult, outcome: int, a_first: bool, turns: int) -> None:
    if outcome == 0:
        winner_is_a = a_first
    elif outcome == 1:
        winner_is_a = not a_first
    else:
        res.draws += 1
        res.turns.append(turns)
        return
    if winner_is_a:
        res.wins_a += 1
    else:
        res.wins_b += 1
    res.turns.append(turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--a", default="submission")
    ap.add_argument("--b", default="submission")
    ap.add_argument("--driver", choices=["auto", "kaggle", "lowlevel"], default="auto")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a_dir = os.path.join(root, args.a)
    b_dir = os.path.join(root, args.b)
    a = _load_agent_module(a_dir)
    b = _load_agent_module(b_dir)
    deck_a = _read_deck(a_dir)
    deck_b = _read_deck(b_dir)

    driver = args.driver
    if driver == "auto":
        driver = "kaggle" if kaggle_env_available() else "lowlevel"

    if driver == "kaggle":
        if not kaggle_env_available():
            print("kaggle_environments not installed. pip install kaggle-environments")
            return
        res = run_match_kaggle(a, b, deck_a, deck_b, args.games)
    else:
        if not engine_available():
            print(
                "cabt engine ('cg') not found on PYTHONPATH.\n"
                "Download the Simulation sample submission and place its 'cg/' "
                "package where Python can import it, then re-run."
            )
            return
        res = run_match_lowlevel(a, b, deck_a, deck_b, args.games)
    print(res)


if __name__ == "__main__":
    main()
