"""Unit tests for the hardcoded supervisor and archetype sub-agents.

Engine-independent: a minimal ``GameData`` is populated by hand (a couple of
attacks and ex/mega flags), and observations are built as plain ``obs_dict``
payloads shaped like the cabt schema, mirroring ``tests/test_agent.py``.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(os.path.dirname(HERE), "submission")
sys.path.insert(0, SUBMISSION)

from agent import supervisor  # noqa: E402
from agent.adapter import extract_select  # noqa: E402
from agent.archetype import AGGRO, UNKNOWN, WALL, ArchetypeDetector  # noqa: E402
from agent.enums import OptionType, SelectType  # noqa: E402
from agent.gamedata import GameData, _Attack  # noqa: E402


def _gd():
    """A tiny hand-built GameData: attack 10 hits 120, attack 11 hits 30."""
    gd = GameData()
    gd.attacks = {
        10: _Attack(10, "big", 120, [6]),
        11: _Attack(11, "small", 30, [6]),
    }
    gd.is_ex_id = {979, 117}   # some ex ids
    gd.is_mega_id = {678}      # a mega id
    return gd


def _mon(cid, hp, maxhp, serial=1):
    return {"id": cid, "serial": serial, "hp": hp, "maxHp": maxhp, "energies": [6]}


def _obs(options, *, my_prize=1, opp_active_hp=100, opp_bench=None, turn=5, yi=0):
    me = {"active": [_mon(979, 200, 230)], "bench": [], "prize": [0] * my_prize}
    opp = {
        "active": [_mon(500, opp_active_hp, 210, serial=9)],
        "bench": opp_bench if opp_bench is not None else [],
        "prize": [0] * 3,
    }
    players = [me, opp] if yi == 0 else [opp, me]
    return {
        "current": {"turn": turn, "yourIndex": yi, "players": players},
        "logs": [],
        "select": {
            "type": SelectType.MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
            "deck": None,
        },
    }


# --- supervisor: winning-KO force ----------------------------------------
def test_forces_winning_ko_by_prize():
    # We have 1 prize left; a 120-damage attack KOs the 100-HP Active -> win.
    obs = _obs(
        [{"type": OptionType.END}, {"type": OptionType.ATTACK, "attackId": 10}],
        my_prize=1, opp_active_hp=100,
    )
    sel = extract_select(obs)
    assert supervisor.forced_main(obs, sel, _gd()) == [1]


def test_forces_winning_ko_by_wipe():
    # Opponent has no bench; KO'ing the Active leaves them with no Pokémon.
    obs = _obs(
        [{"type": OptionType.ATTACK, "attackId": 10}, {"type": OptionType.END}],
        my_prize=3, opp_active_hp=90, opp_bench=[],
    )
    sel = extract_select(obs)
    assert supervisor.forced_main(obs, sel, _gd()) == [0]


def test_no_force_when_ko_does_not_win():
    # KO available but we still need 3 prizes and opponent has a backup -> defer.
    obs = _obs(
        [{"type": OptionType.END}, {"type": OptionType.ATTACK, "attackId": 10}],
        my_prize=3, opp_active_hp=100, opp_bench=[_mon(333, 70, 70, serial=2)],
    )
    sel = extract_select(obs)
    assert supervisor.forced_main(obs, sel, _gd()) is None


def test_no_force_when_not_lethal():
    # 30-damage attack cannot KO a 100-HP Active.
    obs = _obs(
        [{"type": OptionType.ATTACK, "attackId": 11}], my_prize=1, opp_active_hp=100,
    )
    sel = extract_select(obs)
    assert supervisor.forced_main(obs, sel, _gd()) is None


# --- supervisor: never pass up a KO to end the turn ----------------------
def test_guard_overrides_end_with_lethal():
    obs = _obs(
        [{"type": OptionType.END}, {"type": OptionType.ATTACK, "attackId": 10}],
        my_prize=3, opp_active_hp=100, opp_bench=[_mon(333, 70, 70, serial=2)],
    )
    sel = extract_select(obs)
    # Autonomous stack chose END (index 0); guard swaps to the lethal attack.
    assert supervisor.guard_main(obs, sel, _gd(), [0]) == [1]


def test_guard_keeps_non_end_choice():
    obs = _obs(
        [{"type": OptionType.END}, {"type": OptionType.ATTACK, "attackId": 10}],
        opp_active_hp=100,
    )
    sel = extract_select(obs)
    # Already attacking -> unchanged.
    assert supervisor.guard_main(obs, sel, _gd(), [1]) == [1]


# --- archetype detection --------------------------------------------------
def test_detects_wall():
    gd = _gd()
    det = ArchetypeDetector(gd)
    # Opponent: two high-HP non-ex bodies, tanky, slow prize race, we lost none.
    for turn in range(1, 8):
        opp_bench = [_mon(41, 140, 140, serial=2), _mon(135, 150, 150, serial=3)]
        obs = _obs([{"type": OptionType.END}], my_prize=6, turn=turn,
                   opp_active_hp=140, opp_bench=opp_bench)
        # active is a 210-HP tank (serial 9) -> reads as a wall body via maxHp
        det.update(obs)
    assert det.archetype() == WALL


def test_detects_aggro():
    gd = _gd()
    det = ArchetypeDetector(gd)
    # Opponent: multiple ex in play and they have taken 3 prizes off us fast.
    for turn in range(1, 6):
        opp_bench = [_mon(979, 200, 230, serial=2), _mon(117, 210, 210, serial=3)]
        obs = _obs([{"type": OptionType.END}], my_prize=3, turn=turn,
                   opp_active_hp=200, opp_bench=opp_bench)
        det.update(obs)
    assert det.archetype() == AGGRO


def test_unknown_early_and_delta_empty():
    det = ArchetypeDetector(_gd())
    obs = _obs([{"type": OptionType.END}], turn=1)
    det.update(obs)
    assert det.archetype() == UNKNOWN
    assert det.rules_delta() == {}
    assert det.eval_delta() == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
