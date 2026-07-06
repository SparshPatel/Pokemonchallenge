"""Freeze offline-trained weights into the stdlib runtime agent.

Training (:mod:`ptcg_agent.train`) writes the champion weights to
``artifacts/trained_weights.json``. This tool copies a validated weight file to
``submission/agent/trained_weights.json``, which ``agent.rules`` loads at import
time. With no such file present the agent uses its hand-tuned defaults, so
baking is the *only* step that changes shipped behavior — and it is reversible
with ``--remove``.

Only bake weights you have validated (e.g. via the gauntlet vs the strong
baseline); a regression here ships a weaker agent.

Usage::

    python tools/bake_weights.py                 # copy artifacts -> submission
    python tools/bake_weights.py --dry-run       # show, don't write
    python tools/bake_weights.py --remove        # revert to defaults
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, "artifacts", "trained_weights.json")
DEST = os.path.join(ROOT, "submission", "agent", "trained_weights.json")

# Keys the runtime understands (mirror of rules.DEFAULT_WEIGHTS).
KNOWN_KEYS = {
    "lethal_base", "attack_base", "attack_dmg_scale", "ability", "evolve",
    "attach_active", "attach_bench", "attach_powered", "play_basic_room",
    "play_basic_noroom", "play_supporter", "play_item", "play_other",
    "retreat", "retreat_danger", "attach_completes", "play_basic_empty",
    "gust_ko", "gust_ex", "discard", "end", "prefer_first",
}


def _validate(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("weights file is not a JSON object")
    clean = {}
    for key, val in data.items():
        if key not in KNOWN_KEYS:
            print(f"  (ignoring unknown key: {key})")
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"value for {key!r} is not numeric: {val!r}")
        clean[key] = float(val)
    if not clean:
        raise ValueError("no usable weights found")
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true", help="delete shipped weights (revert to defaults)")
    args = ap.parse_args()

    if args.remove:
        if os.path.exists(DEST):
            if args.dry_run:
                print(f"[dry-run] would remove {DEST}")
            else:
                os.remove(DEST)
                print(f"Removed {DEST} (agent now uses hand-tuned defaults)")
        else:
            print("No baked weights present; agent already on defaults.")
        return

    with open(args.src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    clean = _validate(data)
    print(f"Validated {len(clean)} weights from {args.src}:")
    for k in sorted(clean):
        print(f"  {k:18s} = {clean[k]}")

    if args.dry_run:
        print(f"[dry-run] would write {DEST}")
        return
    shutil.copyfile(args.src, DEST)
    print(f"\nBaked -> {DEST}")
    print("Validate with the gauntlet before committing, e.g.:")
    print('  $env:PYTHONPATH="src"; python -m ptcg_agent.harness --games 200 '
          "--a submission --b baselines/strong_agent --driver lowlevel")


if __name__ == "__main__":
    main()
