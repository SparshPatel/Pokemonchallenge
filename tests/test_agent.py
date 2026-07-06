"""Smoke + validity tests for the runtime agent using synthetic observations.

These do not require the cabt engine. They feed hand-built ``obs_dict`` payloads
shaped like the documented cabt schema (``select.option`` with integer
``OptionType`` values) and assert the agent always returns a legal selection and
never raises.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(os.path.dirname(HERE), "submission")
sys.path.insert(0, SUBMISSION)

import main  # noqa: E402
from agent.enums import OptionType, SelectType  # noqa: E402


def _select(options, min_count=1, max_count=1, select_type=SelectType.MAIN, context=0):
    return {
        "current": {"yourIndex": 0, "turn": 1},
        "logs": [],
        "select": {
            "type": select_type,
            "context": context,
            "minCount": min_count,
            "maxCount": max_count,
            "remainEnergyCost": 0,
            "remainDamageCounter": 0,
            "option": options,
            "deck": None,
        },
    }


def test_deck_phase_returns_60_ids():
    obs = {"current": None, "select": None, "logs": []}
    out = main.agent(obs)
    assert isinstance(out, list)
    assert len(out) == 60
    assert all(isinstance(i, int) for i in out)


def test_single_pick_action():
    # During the main phase the agent sets up before attacking: attaching Energy
    # ranks above an attack (which would end the turn) and above passing.
    obs = _select(
        [
            {"type": OptionType.END},
            {"type": OptionType.ATTACK, "attackId": 1},
            {"type": OptionType.ATTACH},
        ]
    )
    out = main.agent(obs)
    assert out == [2], f"expected the setup (attach) option, got {out}"


def test_multi_pick_respects_min_count():
    # Unknown/non-acquire context: take exactly the minimum required count.
    obs = _select(
        [{"type": OptionType.DISCARD} for _ in range(5)],
        min_count=2,
        max_count=3,
        select_type=SelectType.CARD,
    )
    out = main.agent(obs)
    assert len(out) == 2
    assert len(set(out)) == 2
    assert all(0 <= i < 5 for i in out)


def test_fixed_count_select():
    obs = _select(
        [{"type": OptionType.CARD} for _ in range(4)],
        min_count=2,
        max_count=2,
        select_type=SelectType.CARD,
    )
    out = main.agent(obs)
    assert len(out) == 2
    assert len(set(out)) == 2


def test_empty_options_does_not_crash():
    obs = _select([], min_count=0, max_count=0, select_type=SelectType.CARD)
    out = main.agent(obs)
    assert isinstance(out, list)


def test_unknown_schema_falls_back():
    obs = {"totally": "unexpected", "select": {"weird": True}}
    out = main.agent(obs)
    assert isinstance(out, list)


def test_garbage_input_never_raises():
    for bad in [None, {}, [], 42, "x", {"select": 123}]:
        out = main.agent(bad)
        assert isinstance(out, list)


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
