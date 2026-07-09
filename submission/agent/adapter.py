"""Adapter between the raw engine ``obs_dict`` and a normalized view.
Grounded in the cabt API reference. The observation is a plain ``dict`` with::
    obs = {
        "select": SelectData | None,   # None during initial deck selection
        "logs":   [Log, ...],
        "current": State | None,       # None during initial deck selection
        "search_begin_input": str | None,
    }
``SelectData`` (the ``select`` block)::
    {
        "type": int (SelectType),
        "context": int (SelectContext),
        "minCount": int,               # may be 0
        "maxCount": int,
        "remainDamageCounter": int,
        "remainEnergyCost": int,
        "option": [Option, ...],       # note: singular key "option"
        "deck": [Card] | None,
        "contextCard": Card | None,
        "effect": Card | None,
    }
Each ``Option`` carries ``type`` (int OptionType) plus context-dependent fields
(``area``, ``index``, ``playerIndex``, ``attackId``, ``cardId`` ...). The chosen
*action* is the list of option **indices** within ``option``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

def _get(
    d: Any,
    key: str,
    default=None,
):
    if not isinstance(d, dict):
        return default
    value = d.get(key, default)
    return default if value is None else value

@dataclass
class Option:
    """A single selectable option (one entry of ``select.option``)."""
    index: int                # position within select.option (this is the action)
    type: int | None          # OptionType
    raw: dict = field(default_factory=dict)

    # Convenience accessors for commonly-used Option fields.
    @property
    def area(self) -> int | None:
        return self.raw.get("area")

    @property
    def card_id(self) -> int | None:
        return self.raw.get("cardId")

    @property
    def attack_id(self) -> int | None:
        return self.raw.get("attackId")

    @property
    def hand_index(self) -> int | None:
        return self.raw.get("index")

    @property
    def in_play_area(self) -> int | None:
        return self.raw.get("inPlayArea")

    @property
    def in_play_index(self) -> int | None:
        return self.raw.get("inPlayIndex")

    @property
    def player_index(self) -> int | None:
        return self.raw.get("playerIndex")

@dataclass
class Select:
    """Normalized description of the current choice the agent must make."""
    options: list[Option]
    min_count: int
    max_count: int
    select_type: int | None
    context: int | None
    remain_energy_cost: int
    remain_damage_counter: int
    deck: list | None
    raw: dict = field(default=None, repr=False)

def is_deck_phase(obs_dict) -> bool:
    """True when the engine is asking for the 60-card deck.
    Per the API, ``select`` is ``None`` during the initial deck-selection phase.
    """
    if not isinstance(obs_dict, dict):
        return False
    return obs_dict.get("select") is None

def extract_select(obs_dict) -> Select | None:
    """Normalize the ``select`` block, or ``None`` during the deck phase."""
    if not isinstance(obs_dict, dict):
        return None
    sel = obs_dict.get("select")
    if not isinstance(sel, dict):
        return None
    raw_options = sel.get("option") or []
    options: list[Option] = []
    for i, opt in enumerate(raw_options):
        opt_dict = opt if isinstance(opt, dict) else {}
        options.append(Option(index=i, type=opt_dict.get("type"), raw=opt_dict))
    n = len(options)
    min_count = sel.get("minCount")
    max_count = sel.get("maxCount")
    min_count = 0 if min_count is None else int(min_count)
    max_count = n if max_count is None else int(max_count)
    # Clamp to valid bounds against the actual option count.
    if n:
        min_count = max(0, min(min_count, n))
        max_count = max(min_count, min(max_count, n))
    else:
        max_count = max(min_count, max_count)
    return Select(
        options=options,
        min_count=min_count,
        max_count=max_count,
        select_type=sel.get("type"),
        context=sel.get("context"),
        remain_energy_cost=int(sel.get("remainEnergyCost") or 0),
        remain_damage_counter=int(sel.get("remainDamageCounter") or 0),
        deck=sel.get("deck"),
        raw=sel,
    )

def current_state(obs_dict) -> dict | None:
    """Return the ``current`` State dict, or ``None`` during deck selection."""
    return _get(obs_dict, "current")

def your_index(obs_dict) -> int:
    state = current_state(obs_dict)
    yi = _get(state, "yourIndex")
    return int(yi) if isinstance(yi, int) else 0

# ============================================================================
# Unified State Access Helpers
# ============================================================================
def player(state: dict | None, index: int) -> dict:
    """Safely return a player dictionary."""
    if not isinstance(state, dict):
        return {}
    players = state.get("players") or []
    if (
        isinstance(players, list)
        and 0 <= index < len(players)
        and isinstance(players[index], dict)
    ):
        return players[index]
    return {}

def opponent_index(index: int) -> int:
    """Return opponent index."""
    return 1 - index

def active(state: dict | None, index: int) ->dict | None:
    """Return active Pokémon dictionary."""
    p = player(state, index)
    active_list = p.get("active") or []
    if (
        isinstance(active_list, list)
        and active_list
        and isinstance(active_list[0], dict)
    ):
        return active_list[0]
    return None

def bench(state: dict | None, index: int) -> list:
    """Return bench list."""
    p = player(state, index)
    b = p.get("bench")
    return b if isinstance(b, list) else []

def hand(state: dict | None, index: int) -> list:
    """Return hand list."""
    p = player(state, index)
    h = p.get("hand")
    return h if isinstance(h, list) else []

def discard(state: dict | None, index: int) -> list:
    """Return discard pile."""
    p = player(state, index)
    d = p.get("discard")
    return d if isinstance(d, list) else []

def prizes(state: dict | None, index: int) -> list:
    """Return prize cards."""
    p = player(state, index)
    pr = p.get("prize")
    return pr if isinstance(pr, list) else []

def card_id(pokemon: dict | None) -> int | None:
    """Return Pokémon card id."""
    if not isinstance(pokemon, dict):
        return None
    cid = pokemon.get("id")
    return cid if isinstance(cid, int) else None

def hp(pokemon: dict | None) -> int:
    """Return current HP."""
    if not isinstance(pokemon, dict):
        return 0
    value = pokemon.get("hp")
    return int(value) if isinstance(value, (int, float)) else 0

def max_hp(pokemon: dict | None) -> int:
    """Return maximum HP."""
    if not isinstance(pokemon, dict):
        return 0
    value = pokemon.get("maxHp")
    return int(value) if isinstance(value, (int, float)) else 0

def energies(pokemon: dict | None) -> list:
    """Return attached energies."""
    if not isinstance(pokemon, dict):
        return []
    e = pokemon.get("energies")
    return e if isinstance(e, list) else []

def active_card_id(state: dict | None, index: int) -> int | None:
    """Return active Pokémon card id."""
    return card_id(active(state, index))

def active_hp(state: dict | None, index: int) -> int:
    """Return active Pokémon HP."""
    return hp(active(state, index))

def bench_slots(state: dict | None, index: int) -> int:
    """Return remaining bench capacity."""
    p = player(state, index)
    bench_max = p.get("benchMax")
    if not isinstance(bench_max, int):
        bench_max = 5
    return max(0, bench_max - len(bench(state, index)))