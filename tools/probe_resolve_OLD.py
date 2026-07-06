"""Verify resolving option index -> hand card, and attackId -> damage.

Run: $env:PYTHONPATH="src"; python tools/probe_resolve.py
"""
import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cg.game as game  # noqa: E402
import cg.api as api  # noqa: E402
from agent.deck import load_deck  # noqa: E402

deck = load_deck(os.path.join(ROOT, "submission", "deck.csv"))
cards = {c.cardId: c for c in api.all_card_data()}
attacks = {a.attackId: a for a in api.all_attack()}

import random
random.seed(2)
obs, _ = game.battle_start(deck, deck)

shown = 0
n = 0
while n < 400 and shown < 6:
    sel = obs.get("select")
    if sel is None:
        obs = game.battle_select(deck); continue
    cur = obs.get("current")
    opts = sel.get("option") or []
    yi = cur["yourIndex"] if cur else 0
    hand = (cur["players"][yi].get("hand") if cur else None) or []

    types_here = {o.get("type") for o in opts}
    if cur and sel.get("type") == int(api.SelectType.MAIN) and (
        int(api.OptionType.ATTACK) in types_here or int(api.OptionType.PLAY) in types_here
    ):
        print(f"\n--- MAIN decision (turn {cur.get('turn')}) hand={[cards[h['id']].name for h in hand]} ---")
        for i, o in enumerate(opts):
            t = api.OptionType(o["type"]).name
            extra = ""
            if o["type"] == int(api.OptionType.PLAY):
                hidx = o.get("index")
                cd = cards.get(hand[hidx]["id"]) if hidx is not None and hidx < len(hand) else None
                extra = f"hand[{hidx}]={cd.name if cd else '?'} ({api.CardType(cd.cardType).name if cd else '?'})"
            elif o["type"] == int(api.OptionType.ATTACH):
                hidx = o.get("index")
                cd = cards.get(hand[hidx]["id"]) if hidx is not None and hidx < len(hand) else None
                extra = f"energy hand[{hidx}]={cd.name if cd else '?'} -> area{o.get('inPlayArea')}#{o.get('inPlayIndex')}"
            elif o["type"] == int(api.OptionType.ATTACK):
                a = attacks.get(o.get("attackId"))
                extra = f"{a.name if a else '?'} dmg={a.damage if a else '?'} cost={[api.EnergyType(e).name for e in a.energies] if a else '?'}"
            print(f"  [{i}] {t} {extra}")
        shown += 1

    k = sel.get("maxCount") or 1
    opts = sel.get("option") or []
    choice = random.sample(range(len(opts)), min(k, len(opts))) if opts else []
    try:
        obs = game.battle_select(choice)
    except Exception as e:
        print("select failed:", e); break
    cur = obs.get("current")
    if cur and isinstance(cur.get("result"), int) and cur["result"] >= 0:
        break
    n += 1
game.battle_finish()
