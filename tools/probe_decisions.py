"""Probe real decision points: play a self-play game and log SelectData/Option detail.

Run: $env:PYTHONPATH="src"; python tools/probe_decisions.py
"""
import sys, os, json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cg.game as game  # noqa: E402
import cg.api as api  # noqa: E402
from agent.deck import load_deck  # noqa: E402

deck = load_deck(os.path.join(ROOT, "submission", "deck.csv"))

# Build a quick cardId -> name map from the engine.
cards = {c.cardId: c for c in api.all_card_data()}

sel_type_names = {int(v): v.name for v in api.SelectType}
opt_type_names = {int(v): v.name for v in api.OptionType}
ctx_names = {int(v): v.name for v in api.SelectContext}

obs, _ = game.battle_start(deck, deck)
import random
random.seed(0)

sel_type_counter = Counter()
opt_type_counter = Counter()
samples = []
n = 0
while n < 400:
    sel = obs.get("select")
    if sel is None:
        # deck phase
        obs = game.battle_select(deck)
        continue
    st = sel.get("type")
    ctx = sel.get("context")
    opts = sel.get("option") or []
    sel_type_counter[sel_type_names.get(st, st)] += 1
    for o in opts:
        opt_type_counter[opt_type_names.get(o.get("type"), o.get("type"))] += 1

    # Capture a few interesting MAIN decisions (multiple option types).
    types_here = {o.get("type") for o in opts}
    if len(samples) < 12 and (st == int(api.SelectType.MAIN) or len(types_here) > 1):
        samples.append({
            "selectType": sel_type_names.get(st, st),
            "context": ctx_names.get(ctx, ctx),
            "min": sel.get("minCount"), "max": sel.get("maxCount"),
            "options": [
                {
                    "type": opt_type_names.get(o.get("type"), o.get("type")),
                    **{k: v for k, v in o.items() if k != "type" and v is not None},
                    "name": cards[o["cardId"]].name if o.get("cardId") in cards else None,
                }
                for o in opts
            ],
        })

    # random legal pick to keep the game moving
    k = sel.get("maxCount") or 1
    choice = random.sample(range(len(opts)), min(k, len(opts))) if opts else []
    try:
        obs = game.battle_select(choice)
    except Exception as e:
        print("select failed:", e); break
    cur = obs.get("current")
    if cur and isinstance(cur.get("result"), int) and cur["result"] >= 0:
        print("game ended, result =", cur["result"]); break
    n += 1

print("\n=== SelectType frequency ===")
for k, v in sel_type_counter.most_common():
    print(f"  {k}: {v}")
print("\n=== OptionType frequency ===")
for k, v in opt_type_counter.most_common():
    print(f"  {k}: {v}")
print("\n=== Sample decisions ===")
for s in samples:
    print(json.dumps(s, ensure_ascii=False))
game.battle_finish()
