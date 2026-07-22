"""Competition entrypoint for the Pokémon TCG AI Battle (Simulation category).

The CABT engine calls:

    agent(obs_dict) -> list[int]

once per decision point.

Deck selection:
    obs.select is None.
    Return the full 60-card deck.

Action selection:
    Return option indices satisfying the engine's selection contract.

Important:
    The engine's maxCount is authoritative.

    If an internally inconsistent observation ever reports:

        minCount > maxCount

    then no mathematically valid selection exists. We must still avoid
    returning more than maxCount because doing so causes an immediate
    engine-level rejection / forfeit.

Design priority:
    Never raise.
    Never return more than maxCount.
    Preserve existing gameplay policy whenever the policy output is legal.
"""

from __future__ import annotations

import os
import sys
import traceback


def _resolve_here() -> str:
    """Locate the directory containing this agent bundle.

    Kaggle may execute main.py through exec(), in which case __file__ is
    unavailable. We therefore try:

    1. __file__
    2. Kaggle's agent mount
    3. Current working directory
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass

    kaggle_dir = "/kaggle_simulations/agent"

    if os.path.isdir(kaggle_dir):
        return kaggle_dir

    return os.getcwd()


# ---------------------------------------------------------------------------
# Import bundled agent package
# ---------------------------------------------------------------------------

_HERE = _resolve_here()

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


from agent.adapter import (  # noqa: E402
    extract_select,
    is_deck_phase,
)

from agent.deck import load_deck  # noqa: E402
from agent.policy import Policy  # noqa: E402


# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------

_DECK: list[int] | None = None
_POLICY: Policy | None = None


# ---------------------------------------------------------------------------
# Deck / policy initialization
# ---------------------------------------------------------------------------

def _get_deck() -> list[int]:
    global _DECK

    if _DECK is None:
        _DECK = load_deck(
            os.path.join(
                _HERE,
                "deck.csv",
            )
        )

    return _DECK


def _get_policy() -> Policy:
    global _POLICY

    if _POLICY is None:
        _POLICY = Policy(
            card_stats_path=os.path.join(
                _HERE,
                "cards.json",
            ),
            deck_ids=_get_deck(),
        )

    return _POLICY


# ---------------------------------------------------------------------------
# Selection contract helpers
# ---------------------------------------------------------------------------

def _safe_fallback(obs_dict) -> list[int]:
    """Return the safest possible selection.

    The engine's maxCount is the hard upper bound.

    Normal contract:
        minCount <= maxCount

    In that case we select exactly minCount options.

    Defensive contract:
        minCount > maxCount

    In that case no mathematically valid selection exists. We nevertheless
    MUST NOT return more than maxCount, because doing so causes the engine
    to reject the agent immediately.

    Therefore, in an inconsistent contract we return the largest number
    of selections permitted by maxCount.

    This function never calls Policy, Planner, PIMC, or Rules.
    """

    try:
        sel = extract_select(obs_dict)

        # Deck-selection phase.
        if sel is None:
            return _get_deck()

        n_options = len(sel.options)

        # Normalize both bounds independently.
        min_count = max(
            0,
            min(
                int(sel.min_count),
                n_options,
            ),
        )

        max_count = max(
            0,
            min(
                int(sel.max_count),
                n_options,
            ),
        )

        # ---------------------------------------------------------------
        # Normal, internally consistent contract.
        # ---------------------------------------------------------------
        if min_count <= max_count:
            return list(
                range(min_count)
            )

        # ---------------------------------------------------------------
        # Defensive handling for impossible engine contract:
        #
        #     minCount > maxCount
        #
        # There is no fully valid response.
        #
        # The critical rule is:
        #
        #     NEVER exceed maxCount.
        #
        # Returning minCount here could itself trigger:
        #
        #     Agent returned too many selections.
        #
        # Therefore choose the largest number allowed by maxCount.
        # ---------------------------------------------------------------
        if os.getenv("DEBUG_AGENT"):
            print(
                "[AGENT CONTRACT WARNING] "
                f"Impossible selection contract: "
                f"minCount={min_count}, "
                f"maxCount={max_count}, "
                f"options={n_options}. "
                f"Returning {max_count} selections.",
                file=sys.stderr,
            )

        return list(
            range(max_count)
        )

    except Exception:
        # Absolute last resort.

        if os.getenv("DEBUG_AGENT"):
            traceback.print_exc(
                file=sys.stderr
            )

        # One selection is safer than two, but only if the engine permits it.
        # We cannot reliably inspect the contract here, so preserve the
        # previous defensive behavior.
        return [0]


def _debug_invalid_choice(
    choice,
    sel,
) -> None:
    """Log an invalid policy response when debugging is enabled.

    This function does not modify gameplay.
    """

    if not os.getenv("DEBUG_AGENT"):
        return

    try:
        print(
            "[AGENT INVALID SELECTION]",
            file=sys.stderr,
        )

        print(
            f"  choice      : {choice!r}",
            file=sys.stderr,
        )

        print(
            f"  choice_type : {type(choice).__name__}",
            file=sys.stderr,
        )

        try:
            choice_len = len(choice)
        except Exception:
            choice_len = "N/A"

        print(
            f"  choice_len  : {choice_len}",
            file=sys.stderr,
        )

        print(
            f"  minCount    : {sel.min_count}",
            file=sys.stderr,
        )

        print(
            f"  maxCount    : {sel.max_count}",
            file=sys.stderr,
        )

        print(
            f"  option_count: {len(sel.options)}",
            file=sys.stderr,
        )

        print(
            f"  select_type : {sel.select_type!r}",
            file=sys.stderr,
        )

        print(
            f"  context     : {sel.context!r}",
            file=sys.stderr,
        )

    except Exception:
        # Diagnostics must never affect gameplay.
        pass


def _valid(
    choice,
    sel,
) -> bool:
    """Return whether a policy response satisfies the engine contract."""

    if not isinstance(
        choice,
        list,
    ):
        return False

    n = len(sel.options)

    # No duplicate selections.
    if len(choice) != len(set(choice)):
        return False

    # Exact engine cardinality contract.
    if not (
        sel.min_count
        <= len(choice)
        <= sel.max_count
    ):
        return False

    # Every selection must be a valid option index.
    return all(
        isinstance(i, int)
        and not isinstance(i, bool)
        and 0 <= i < n
        for i in choice
    )


# ---------------------------------------------------------------------------
# Competition entrypoint
# ---------------------------------------------------------------------------

# IMPORTANT:
#
# `agent` MUST be the last callable defined in this module.
#
# Kaggle's get_last_callable() behavior may select the last callable found
# in the module namespace. Do not define functions after agent().
#
def agent(obs_dict) -> list[int]:
    """Return the candidate deck or next legal action selection."""

    try:
        # ---------------------------------------------------------------
        # Deck selection
        # ---------------------------------------------------------------

        if is_deck_phase(obs_dict):
            return _get_deck()

        # ---------------------------------------------------------------
        # Extract selection contract
        # ---------------------------------------------------------------

        sel = extract_select(
            obs_dict
        )

        # Defensive handling of deck-selection-like observations.
        if sel is None:
            return _get_deck()

        # ---------------------------------------------------------------
        # Existing policy stack
        #
        # Gameplay logic is intentionally unchanged.
        #
        # Policy.choose()
        #     -> planner
        #     -> PIMC
        #     -> rules fallback
        # ---------------------------------------------------------------

        choice = _get_policy().choose(
            obs_dict,
            sel,
        )

        # ---------------------------------------------------------------
        # Contract validation
        # ---------------------------------------------------------------

        if _valid(
            choice,
            sel,
        ):
            return choice

        _debug_invalid_choice(
            choice,
            sel,
        )

        # Minimal contract-safety fix:
        # If the policy returns too many selections, keep only
        # the first maxCount valid selections.
        try:
            max_count = max(
                0,
                min(
                    int(sel.max_count),
                    len(sel.options),
                ),
            )

            return [
                i
                for i in choice
                if isinstance(i, int)
                and not isinstance(i, bool)
                and 0 <= i < len(sel.options)
            ][:max_count]

        except Exception:
            return _safe_fallback(obs_dict)

    except Exception:
        # Never allow exceptions to escape to the competition engine.

        if os.getenv("DEBUG_AGENT"):
            traceback.print_exc(
                file=sys.stderr
            )

        return _safe_fallback(
            obs_dict
        )