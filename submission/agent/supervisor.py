"""Hardcoded high-impact supervisor — the deterministic top of the stack.

The autonomous layers (planner beam-search, then the heuristic rule policy) are
good on average but statistical: a leaf-eval quirk or an exhausted search budget
can, rarely, make them walk past a *game-winning* KO or pass the turn with lethal
on the board. Those are exactly the mistakes that swing a match.

This supervisor encodes the handful of situations where the correct move is not a
matter of judgement but of the rules of the game, and forces it regardless of
what search decided:

* **Winning KO (force)** — if an affordable attack Knocks Out the opponent's
  Active *and* that KO takes our last needed Prize (or leaves them with no
  Pokémon in play), play it now. This is an unconditional win; never search past
  it.
* **Never pass up a KO to end the turn (guard)** — if the autonomous stack chose
  to END the turn while an affordable KO on the Active is available, take the KO
  instead. Ending on an un-taken KO is never correct for an aggro deck.

Everything is opt-in-safe: each method returns ``None`` (defer to the autonomous
choice) on anything it is not certain about, and is wrapped so an exception can
never change legality — the caller keeps its original, already-legal selection.
"""
from __future__ import annotations

from .adapter import Select
from .enums import OptionType, SelectType
from .gamedata import GameData

# Bloodmoon Ursaluna "Mad Bite" (attackId 175): 100 + 30 per damage counter on
# the defender. Kept in sync with the same special-case in rules._score_main.
_MAD_BITE_ID = 175


def _player(state, pi) -> dict:
    players = state.get("players") or [] if isinstance(state, dict) else []
    if 0 <= pi < len(players) and isinstance(players[pi], dict):
        return players[pi]
    return {}


def _active(player: dict) -> dict | None:
    arr = player.get("active") or []
    if arr and isinstance(arr[0], dict):
        return arr[0]
    return None


def _lethal_options(obs_dict, select: Select, gd: GameData):
    """Yield ``(index, effective_damage, opp_prize_value)`` for each affordable
    ATTACK option that Knocks Out the opponent's Active."""
    state = obs_dict.get("current") if isinstance(obs_dict, dict) else None
    if not isinstance(state, dict):
        return []
    yi = state.get("yourIndex", 0)
    if not isinstance(yi, int):
        yi = 0
    me = _player(state, yi)
    opp = _player(state, 1 - yi)
    my_active = _active(me)
    opp_active = _active(opp)
    if not isinstance(opp_active, dict):
        return []
    opp_hp = opp_active.get("hp")
    if not isinstance(opp_hp, (int, float)) or opp_hp <= 0:
        return []
    my_id = my_active.get("id") if isinstance(my_active, dict) else None
    opp_id = opp_active.get("id")
    opp_maxhp = opp_active.get("maxHp") or 0

    out = []
    for opt in select.options:
        if opt.type != OptionType.ATTACK:
            continue
        # Check if the attack is actually affordable (enough Energy to pay).
        cost = gd.attack_cost(opt.attack_id)
        energies = my_active.get("energies") or [] if isinstance(my_active, dict) else []
        if not gd.can_pay(cost, energies):
            continue  # Skip attacks we can't afford
        dmg = gd.attack_damage(opt.attack_id)
        if opt.attack_id == _MAD_BITE_ID:
            counters = max(0, int((opp_maxhp - opp_hp) / 10)) if opp_maxhp else 0
            if counters > 0:
                dmg = 100 + counters * 30
        eff = gd.effective_damage(my_id, dmg, opp_id)
        if eff >= opp_hp:
            out.append((opt.index, eff, gd.prize_value(opp_id)))
    return out


def _prizes_remaining(state, yi) -> int:
    prize = _player(state, yi).get("prize")
    return len(prize) if isinstance(prize, list) else 6


def _opp_has_backup(state, yi) -> bool:
    """True if the opponent has any benched Pokémon to promote after a KO."""
    opp = _player(state, 1 - yi)
    for b in opp.get("bench") or []:
        if isinstance(b, dict) and (b.get("hp") or 0) > 0:
            return True
    return False


def forced_main(obs_dict, select: Select, gd: GameData) -> list[int] | None:
    """Force a game-winning KO when one is available; else defer (``None``)."""
    try:
        if select.select_type != SelectType.MAIN:
            return None
        lethals = _lethal_options(obs_dict, select, gd)
        if not lethals:
            return None
        state = obs_dict.get("current")
        yi = state.get("yourIndex", 0) if isinstance(state, dict) else 0
        if not isinstance(yi, int):
            yi = 0
        my_prizes = _prizes_remaining(state, yi)
        opp_has_backup = _opp_has_backup(state, yi)

        winning = []
        for idx, eff, pv in lethals:
            takes_last_prize = my_prizes <= pv
            wins_by_wipe = not opp_has_backup  # KO the Active, no Pokémon left
            if takes_last_prize or wins_by_wipe:
                winning.append((pv, eff, idx))
        if not winning:
            return None
        # Prefer the KO that banks the most prizes, then the biggest hit.
        winning.sort(reverse=True)
        return [winning[0][2]]
    except Exception:
        return None


def guard_main(obs_dict, select: Select, gd: GameData, choice: list[int]) -> list[int]:
    """Veto ending the turn on an un-taken KO; otherwise return ``choice``."""
    try:
        if select.select_type != SelectType.MAIN or not choice:
            return choice
        chosen = choice[0]
        chosen_type = None
        for opt in select.options:
            if opt.index == chosen:
                chosen_type = opt.type
                break
        if chosen_type != OptionType.END:
            return choice
        lethals = _lethal_options(obs_dict, select, gd)
        if not lethals:
            return choice
        # Take the highest-prize, then highest-damage available KO.
        lethals.sort(key=lambda t: (t[2], t[1]), reverse=True)
        return [lethals[0][0]]
    except Exception:
        return choice
