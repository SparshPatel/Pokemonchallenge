"""Tempo-aggro opponent archetype for benchmarking.

A third distinct play style for the evaluation panel, sitting between
``greedy_agent`` (attacks ASAP but is *blind* to Energy costs) and
``strong_agent`` (sets up patiently, energy-aware). Tempo is **energy-aware but
impatient**:

* **Lethal first** — always take a KO in range.
* **Attack the moment it deals meaningful damage** — attacking out-ranks most
  setup, so it applies pressure early and races on Prizes rather than developing
  a perfect board.
* **Minimal development** — it still benches a Basic when the Active is alone and
  attaches Energy to whatever can attack soonest, but it will not pass up damage
  to dig or build the Bench.

Modelling a fast, pressure-first metagame deck gives the gauntlet an opponent
that punishes slow setups — a different failure mode from the patient
``strong_agent``. Stdlib + ``cg`` only; crash-safe.
"""
from __future__ import annotations

import os

# OptionType / SelectType / AreaType / CardType / EnergyType ints.
O_YES, O_NO = 1, 2
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK, O_END = 7, 8, 9, 10, 11, 12, 13, 14
S_MAIN, S_ENERGY, S_YES_NO = 0, 4, 9
A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE = 2, 3, 4, 5, 6
CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM = 0, 1, 2, 3, 4
CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = 5, 6
E_COLORLESS = 0

ACQUIRE_CTX = {1, 2, 3, 4, 5, 6, 7, 18, 19, 21, 22}


class _Data:
    _inst = None

    def __init__(self):
        self.ctype = {}
        self.basic = set()
        self.ex = set()
        self.attacks = {}        # attackId -> (damage, energies)
        self.card_attacks = {}   # cardId -> [attackId]
        self.ok = False

    @classmethod
    def get(cls):
        if cls._inst is None:
            d = cls()
            d._load()
            cls._inst = d
        return cls._inst

    def _load(self):
        try:
            from cg import api
        except Exception:
            return
        try:
            for a in api.all_attack():
                self.attacks[a.attackId] = (int(a.damage or 0),
                                            [int(e) for e in (a.energies or [])])
            for c in api.all_card_data():
                self.ctype[c.cardId] = int(c.cardType)
                if getattr(c, "basic", False) and int(c.cardType) == CT_POKEMON:
                    self.basic.add(c.cardId)
                if getattr(c, "ex", False) or getattr(c, "megaEx", False):
                    self.ex.add(c.cardId)
                self.card_attacks[c.cardId] = list(c.attacks or [])
            self.ok = True
        except Exception:
            pass

    def is_pokemon(self, cid): return self.ctype.get(cid) == CT_POKEMON
    def is_basic(self, cid): return cid in self.basic
    def is_supporter(self, cid): return self.ctype.get(cid) == CT_SUPPORTER
    def is_item(self, cid): return self.ctype.get(cid) == CT_ITEM
    def is_energy(self, cid): return self.ctype.get(cid) in (CT_BASIC_ENERGY, CT_SPECIAL_ENERGY)
    def attack(self, aid): return self.attacks.get(aid, (0, []))


_DECK = None


def _deck():
    global _DECK
    if _DECK is not None:
        return _DECK
    ids = []
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.lstrip("-").isdigit():
                    ids.append(int(s))
    except Exception:
        pass
    _DECK = ids
    return ids


def agent(obs):
    if not isinstance(obs, dict):
        return [0]
    sel = obs.get("select")
    if sel is None:
        return _deck()
    try:
        return _decide(obs, sel)
    except Exception:
        return _fallback(sel)


def _decide(obs, sel):
    options = sel.get("option") or []
    n = len(options)
    if n == 0:
        return list(range(max(0, int(sel.get("minCount") or 0))))

    stype = sel.get("type")
    d = _Data.get()
    state = obs.get("current") if isinstance(obs, dict) else None
    yi = state.get("yourIndex", 0) if isinstance(state, dict) else 0

    if stype == S_MAIN:
        return _main(options, state, yi, d)
    if stype == S_YES_NO:
        for i, o in enumerate(options):
            if o.get("type") == O_YES:
                return [i]
        return [0]
    if stype == S_ENERGY:
        return _lowest(sel, n)
    return _cards(options, sel, state, yi, d)


def _main(options, state, yi, d):
    hand = _hand(state, yi)
    me = _player(state, yi)
    opp = _player(state, 1 - yi)
    opp_hp = _active_hp(opp)
    bench_room = _bench_room(me)
    has_active = _active(me) is not None

    best_i, best_s = 0, float("-inf")
    for i, o in enumerate(options):
        s = _score(o, hand, me, opp_hp, bench_room, has_active, d)
        if s > best_s:
            best_s, best_i = s, i
    return [best_i]


def _score(o, hand, me, opp_hp, bench_room, has_active, d):
    t = o.get("type")

    if t == O_ATTACK:
        dmg, _ = d.attack(o.get("attackId"))
        if opp_hp is not None and dmg >= opp_hp > 0:
            return 100000 + dmg                 # lethal — always
        if dmg > 0:
            # Tempo: a damaging attack out-ranks setup (ability/evolve/attach).
            return 900 + dmg * 0.5
        return 50                               # 0-damage attack: low value

    if t == O_ABILITY:
        return 700                              # free, but below attacking

    if t == O_ATTACH:
        target = _attach_target(o, me)
        if target is not None and not _needs_energy(target, d):
            return 60                           # already powered
        return 650 if o.get("inPlayArea") == A_ACTIVE else 300

    if t == O_EVOLVE:
        return 500

    if t == O_PLAY:
        cid = _card_at(hand, o.get("index"))
        if d.is_basic(cid):
            if not has_active:
                return 950                      # must have an Active to attack
            return 250 if bench_room else 40    # minimal bench development
        if d.is_supporter(cid):
            return 200
        if d.is_item(cid):
            return 180
        return 220

    if t == O_RETREAT:
        return 25                               # rarely retreat — keep pressure
    if t == O_DISCARD:
        return 20
    if t == O_END:
        return -1000
    return 5


def _cards(options, sel, state, yi, d):
    ctx = sel.get("context")
    acquire = ctx in ACQUIRE_CTX
    deck = sel.get("deck")
    scored = []
    for i, o in enumerate(options):
        cid = _resolve_id(o, deck, state, yi)
        scored.append((_value(cid, d), i))
    scored.sort(reverse=acquire)
    n = len(options)
    mn = int(sel.get("minCount") or 0)
    mx = sel.get("maxCount")
    mx = n if mx is None else int(mx)
    k = (mx or mn) if acquire else mn
    k = max(mn, min(k, n))
    return sorted(i for _, i in scored[:k])


def _value(cid, d):
    if cid is None:
        return 1.0
    if d.is_basic(cid):
        dmg = max((d.attack(a)[0] for a in d.card_attacks.get(cid, [])), default=0)
        return 100.0 + dmg * 0.15 + (30.0 if cid in d.ex else 0.0)
    if d.is_pokemon(cid):
        return 70.0
    if d.is_energy(cid):
        return 45.0                             # tempo values Energy highly
    if d.is_supporter(cid):
        return 40.0
    if d.is_item(cid):
        return 30.0
    return 10.0


def _needs_energy(mon, d):
    if not isinstance(mon, dict):
        return True
    attached = [int(e) for e in (mon.get("energies") or [])]
    costs = [d.attack(a)[1] for a in d.card_attacks.get(mon.get("id"), [])]
    if not costs:
        return False
    cheapest = min(costs, key=len)              # tempo: power the *cheapest* attack
    return not _can_pay(cheapest, attached)


def _can_pay(cost, attached):
    if len(attached) < len(cost):
        return False
    pool = list(attached)
    for need in cost:
        if need == E_COLORLESS:
            continue
        if need in pool:
            pool.remove(need)
        else:
            return False
    colorless = sum(1 for c in cost if c == E_COLORLESS)
    return len(pool) >= colorless


def _player(state, idx):
    pl = state.get("players") if isinstance(state, dict) else None
    if isinstance(pl, list) and 0 <= idx < len(pl) and isinstance(pl[idx], dict):
        return pl[idx]
    return {}


def _hand(state, yi):
    return _player(state, yi).get("hand") or []


def _active(me):
    a = me.get("active") or []
    return a[0] if a and isinstance(a[0], dict) else None


def _active_hp(p):
    a = p.get("active") or []
    if a and isinstance(a[0], dict):
        return a[0].get("hp")
    return None


def _bench_room(me):
    bench = me.get("bench") or []
    return len(bench) < int(me.get("benchMax") or 5)


def _card_at(hand, idx):
    if idx is None or not isinstance(hand, list) or not (0 <= idx < len(hand)):
        return None
    e = hand[idx]
    return e.get("id") if isinstance(e, dict) else None


def _attach_target(o, me):
    area, idx = o.get("inPlayArea"), o.get("inPlayIndex")
    if idx is None:
        return None
    arr = (me.get("active") if area == A_ACTIVE else me.get("bench")) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx]
    return None


_AREA_FIELD = {A_HAND: "hand", A_BENCH: "bench", A_DISCARD: "discard",
               A_ACTIVE: "active", A_PRIZE: "prize"}


def _resolve_id(o, deck, state, yi):
    if o.get("cardId") is not None:
        return o.get("cardId")
    idx = o.get("index")
    if deck and idx is not None and 0 <= idx < len(deck):
        e = deck[idx]
        if isinstance(e, dict):
            return e.get("id")
    field = _AREA_FIELD.get(o.get("area"))
    if field is None or idx is None or not isinstance(state, dict):
        return None
    pi = o.get("playerIndex")
    pi = yi if pi is None else pi
    arr = _player(state, pi).get(field) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx].get("id")
    return None


def _lowest(sel, n):
    mn = int(sel.get("minCount") or 0)
    mx = sel.get("maxCount")
    mx = n if mx is None else int(mx)
    k = max(mn, 1) if mx >= 1 else mn
    return list(range(min(k, n)))


def _fallback(sel):
    options = sel.get("option") or []
    n = len(options)
    mn = max(0, min(int(sel.get("minCount") or 0), n))
    if mn == 0 and sel.get("type") == S_MAIN and n >= 1:
        mn = 1
    return list(range(mn))
