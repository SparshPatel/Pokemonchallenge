"""Competition entrypoint for the Pokémon TCG AI Battle (Simulation category).
The cabt engine calls ``agent(obs_dict) -> list[int]`` once per decision point:
* **Deck selection** (first call): ``obs.select`` is ``None``. We must return the
  full 60-card deck as a ``list[int]`` of Card IDs.
* **Action selection** (every later call): we must return a list of option
  *indices* into the presented option list, with length between ``minCount`` and
  ``maxCount`` and no duplicates.
Design priority: **never raise**. In this competition an exception is scored as a
forfeit loss, so every code path is wrapped and falls back to a guaranteed-valid
selection (the lowest ``minCount`` indices).
"""
from __future__ import annotations
import os
import sys
import traceback

def _resolve_here() -> str:
    """Locate the directory that bundles this agent.
    Kaggle loads ``main.py`` via ``exec()``, so ``__file__`` is *not* defined in
    the runtime. We therefore try, in order: ``__file__`` (local dev / tests),
    the Kaggle agent mount point, then the current working directory.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    kaggle_dir = "/kaggle_simulations/agent"
    if os.path.isdir(kaggle_dir):
        return kaggle_dir
    return os.getcwd()

# Make the bundled ``agent`` package importable regardless of the working
# directory the engine launches us from.
_HERE = _resolve_here()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from agent.adapter import extract_select, is_deck_phase  # noqa: E402
from agent.deck import load_deck  # noqa: E402
from agent.policy import Policy  # noqa: E402

_DECK: list[int] | None = None
_POLICY: Policy | None = None

def _get_deck() -> list[int]:
    global _DECK
    if _DECK is None:
        _DECK = load_deck(os.path.join(_HERE, "deck.csv"))
    return _DECK

def _get_policy() -> Policy:
    global _POLICY
    if _POLICY is None:
        _POLICY = Policy(
            card_stats_path=os.path.join(_HERE, "cards.json"),
            deck_ids=_get_deck(),
        )
    return _POLICY

def _safe_fallback(obs_dict) -> list[int]:
    """Return a guaranteed-valid selection without touching policy code.
    Mirrors the reference random agent: select ``minCount`` of the lowest option
    indices. When ``minCount`` is 0 the empty selection is valid (the engine
    treats it as declining an optional choice).
    """
    try:
        sel = extract_select(obs_dict)
        if sel is None:
            return _get_deck()
        n_options = len(sel.options)
        k = min(sel.min_count, n_options) if n_options else sel.min_count
        return list(range(k))
    except Exception:
        # Absolute last resort: pick the first option.
        return [0]
    
def _valid(choice, sel) -> bool:
    if not isinstance(choice, list):
        return False
    n = len(sel.options)
    if len(choice) != len(set(choice)):
        return False
    if not (sel.min_count <= len(choice) <= sel.max_count):
        return False
    return all(isinstance(i, int) and 0 <= i < n for i in choice)

# NOTE: ``agent`` MUST be the last callable defined in this module. Kaggle's
# ``kaggle_environments.agent.get_last_callable`` selects the *last* callable in
# the module namespace (``[v for v in env.values() if callable(v)][-1]``), not
# a function named ``agent``. Any callable defined after this point would be
# invoked instead, causing an invalid action (and a "deck does not have 60
# cards" rejection during the deck phase). Do not add functions below ``agent``.
def agent(obs_dict) -> list[int]:
    try:
        if is_deck_phase(obs_dict):
            return _get_deck()
        sel = extract_select(obs_dict)
        if sel is None:
            return _get_deck()
        choice = _get_policy().choose(obs_dict, sel)
        if _valid(choice, sel):
            return choice
        return _safe_fallback(obs_dict)
    except Exception:
        # Log to stderr for local debugging; the engine ignores stderr.
        if os.getenv("DEBUG_AGENT"):
            traceback.print_exc(file=sys.stderr)
        return _safe_fallback(obs_dict)