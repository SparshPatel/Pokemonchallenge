"""Game-state-aware heuristic policy — the baseline and the safety fallback.
Unlike a pure type-priority scorer, this policy resolves each option to the real
card/attack it represents (using the engine-backed :mod:`agent.gamedata`) and the
current board state, so it can:
* take a **lethal attack** when the best affordable attack KOs the opponent's
  Active Pokémon,
* otherwise **set up first** — use free abilities, attach Energy to the Active
  attacker, develop the Bench, dig with draw/search trainers — and only then
  attack,
* make sensible **setup / search / discard** picks (best attacker to the Active
  Spot; discard the least valuable cards),
* never pass the turn while a productive action remains.
Every accessor is defensive; missing fields degrade to a safe, legal choice.
"""
from __future__ import annotations
import json
import os
from .adapter import (
    Option,
    Select,
    player,
    opponent_index,
    active,
    bench,
    hand,
    card_id,
    hp,
    active_card_id,
    active_hp,
    bench_slots,
    energies,
)
from .enums import AreaType, OptionType, SelectContext, SelectType
from .gamedata import GameData
_HERE = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------------------
# Tunable decision weights
# ---------------------------------------------------------------------------
# Every magic number in the MAIN scorer lives here so it can be optimized
# offline (see ``ptcg_agent.train``: self-play cross-entropy / evolutionary RL
# with a +1 win / -1 loss reward) and then *frozen* into ``trained_weights.json``
# for the stdlib runtime to read. The defaults below are the hand-tuned values
# validated in the local gauntlet; with no JSON present the agent behaves exactly
# as before, so training can only ever *add* value (we ship trained weights only
# after they beat these defaults).
DEFAULT_WEIGHTS: dict[str, float] = {
    "lethal_base": 10_000.0,    # take the KO — must dominate everything
    "attack_base": 380.0,       # non-lethal attack (ends turn) baseline — attack when ready
    "attack_dmg_scale": 0.40,   # strong preference for higher-damage attacks
    "ability": 1_000.0,         # free ability value — always do first
    "evolve": 800.0,            # board upgrade / unlocks attacks
    "attach_active": 700.0,     # attach Energy to the (needy) Active attacker
    "attach_bench": 500.0,      # attach Energy to a (needy) Bench Pokémon
    "attach_powered": 30.0,     # attach to an already-powered target — never waste before attacking
    "play_basic_room": 600.0,   # develop the Bench when there is room
    "play_basic_noroom": 80.0,  # Bench full — low priority
    "play_supporter": 520.0,    # draw / search engine
    "play_item": 480.0,         # ball / draw / recovery
    "play_other": 540.0,        # tool / stadium / other
    "retreat": 20.0,
    "retreat_danger": 300.0,    # extra to retreat a doomed Active when a fresh
                                # Bench attacker is ready to take its place
    "retreat_promote": 750.0,   # retreat a *support* Active (0-damage, e.g.
                                # Fezandipiti ex) to promote a benched attacker —
                                # ranks just above attaching so we swap the
                                # dead-weight out, then power the real attacker.
                                # Set 0 to disable (legacy behaviour).
    "attach_completes": 40.0,   # extra to attach Energy that *finishes* powering
                                # the Active's attack (ready to swing this turn)
    "attach_dmg_scale": 0.30,   # per damage-point bonus for attaching to a high-damage attacker
    "attach_dmg_cap": 120.0,    # cap on the damage-scaled attach bonus. Bigger
                                # than before so a real attacker (Mega Lucario ex
                                # 270, Koraidon ex 200) strongly out-attracts
                                # Energy over a low-damage body. Set 60 for legacy.
    "attach_concentrate": 25.0, # per-Energy bonus for feeding a target that ALREADY
                                # has Energy invested — commit to charging ONE
                                # attacker instead of spreading 1 Energy/turn
                                # across many Pokémon (the core loss pattern).
                                # Set 0 to disable.
    "attacker_dmg_floor": 1.0,  # a target whose best attack deals < this is a
                                # pure support Pokémon (Fezandipiti ex = 0 dmg):
                                # never pour Energy into it. Set 0 to disable so
                                # Riolu (10 dmg, evolves to Mega Lucario) is safe.
    "play_basic_empty": 10.0,   # per empty Bench slot: bias early development
    "gust_ko": 300.0,           # gust (Boss's Orders) a target we can KO now
    "gust_ex": 120.0,           # prefer gusting an ex (2 Prizes)
    "gust_mega_ex": 180.0,      # prefer gusting a Mega ex (3 Prizes) — extra bonus
    "attack_prize_bonus": 30.0,  # bonus per extra prize value of opp active (above 1)
    "attack_twoshot_bonus": 20.0, # bonus per prize value when can 2-shot (dmg ≥ hp/2)
    "attack_urgency": 200.0,    # extra aggression when either side has ≤2 prizes
    "discard": 10.0,
    "end": -100.0,              # only when nothing productive remains
    "prefer_first": 0.0,        # >0.5 => choose to go FIRST on the coin
}
def _load_weights() -> dict[str, float]:
    """Load frozen, offline-trained weights if present; else the defaults.
    Crash-safe and validated per-key: an unknown key, wrong type, or missing /
    corrupt file degrades to the hand-tuned default for that entry, so a bad
    artifact can never break the runtime.
    """
    weights = dict(DEFAULT_WEIGHTS)
    path = os.path.join(_HERE, "trained_weights.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for key, val in data.items():
                if key in weights and isinstance(val, (int, float)) and not isinstance(val, bool):
                    weights[key] = float(val)
    except Exception:
        pass
    return weights

WEIGHTS: dict[str, float] = _load_weights()
# Going first vs second. Our Ancient Box list is an aggressive setup deck and the
# gauntlet shows it performs better taking the SECOND turn (attacking on its first
# turn rather than ceding tempo): over 200-game runs vs the strong baseline,
# going second ~0.53 vs going first ~0.45. Driven by the ``prefer_first`` weight
# so the offline trainer can revisit the choice. Default (0.0) => go second.
PREFER_FIRST: bool = WEIGHTS["prefer_first"] > 0.5
def set_weights(weights: dict[str, float] | None) -> None:
    """Override the decision weights at runtime (used by the offline trainer).
    Sequential by construction: the engine asks one player to choose at a time,
    and each agent calls this immediately before :func:`choose`, so self-play
    with two different weight vectors in one process is safe.
    """
    global WEIGHTS, PREFER_FIRST
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        merged.update({k: float(v) for k, v in weights.items() if k in DEFAULT_WEIGHTS})
    WEIGHTS = merged
    PREFER_FIRST = WEIGHTS["prefer_first"] > 0.5
# Contexts where we are *acquiring / placing* cards (pick the most valuable).
_ACQUIRE_CONTEXTS = {
    SelectContext.SETUP_ACTIVE_POKEMON,
    SelectContext.SETUP_BENCH_POKEMON,
    SelectContext.SWITCH,
    SelectContext.TO_ACTIVE,
    SelectContext.TO_BENCH,
    SelectContext.TO_FIELD,
    SelectContext.TO_HAND,
    SelectContext.EVOLVES_FROM,
    SelectContext.EVOLVES_TO,
    SelectContext.ATTACH_FROM,
    SelectContext.ATTACH_TO,
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def choose(obs_dict, select: Select, gd: GameData) -> list[int]:
    n = len(select.options)
    if n == 0:
        return list(range(max(0, select.min_count)))
    st = select.select_type
    try:
        if st == SelectType.MAIN:
            return [_choose_main(obs_dict, select, gd)]
        if st == SelectType.YES_NO:
            return [_choose_yes_no(select)]
        if st in (SelectType.ENERGY, SelectType.COUNT):
            return _choose_lowest(select)
        # CARD / ATTACHED_CARD / EVOLVE / etc. — value-based pick.
        return _choose_cards(obs_dict, select, gd)
    except Exception:
        # Any resolution failure: return a guaranteed-legal selection.
        return _fallback(select)

# ---------------------------------------------------------------------------
# MAIN turn decisions
# ---------------------------------------------------------------------------
def _choose_main(obs_dict, select: Select, gd: GameData) -> int:
    state = obs_dict.get("current") if isinstance(obs_dict, dict) else None
    yi = state.get("yourIndex", 0) if isinstance(state, dict) else 0
    me = _player(state, yi)
    hand = me.get("hand") or []
    opp_hp = _opponent_active_hp(state, yi)
    bench_room = _bench_has_room(state, yi)
    # Board context for the threat-aware features (computed once per decision).
    opp_active = _active_pokemon(state, 1 - yi)
    opp_player = _player(state, 1 - yi)
    ctx = {
        "danger": _in_danger(state, yi, gd),
        "bench_attacker": _has_bench_attacker(state, yi, gd),
        "empty_slots": _bench_empty_slots(state, yi),
        "my_active_id": _active_card_id(state, yi),
        "opp_active_id": _active_card_id(state, 1 - yi),
        "my_prizes": len(me.get("prize") or []),
        "opp_prizes": len(opp_player.get("prize") or []),
        # Opponent's damage counters (for Mad Bite scaling: +30/counter)
        "opp_counters": max(0, int(((opp_active.get("maxHp") or 0) - (opp_hp or 0)) / 10))
                        if isinstance(opp_active, dict) else 0,
        # For Boss's Orders / Prime Catcher KO-value scoring in _score_main.
        "opp_bench": opp_player.get("bench") or [],
        "my_active": _active_pokemon(state, yi),
        # Active is a pure support Pokémon (best attack deals no damage, e.g.
        # Fezandipiti ex): it can never trade prizes, so promote a real attacker.
        "active_is_support": gd.best_damage(_active_card_id(state, yi)) <= 0,
    }
    best_idx, best_score = 0, float("-inf")
    for opt in select.options:
        s = _score_main(
            opt,
            hand,
            me,
            state,
            yi,
            opp_hp,
            bench_room,
            ctx,
            gd,
        )
        if s > best_score:
            best_score, best_idx = s, opt.index
    return best_idx

def _score_main(
    opt,
    hand,
    me,
    state,
    yi,
    opp_hp,
    bench_room,
    ctx,
    gd,
):
    t = opt.type
    w = WEIGHTS
    if t == OptionType.ATTACK:
        dmg = gd.attack_damage(opt.attack_id)
        # Mad Bite (Bloodmoon Ursaluna, attack 175): 100 + 30 per damage counter
        # on the opponent's Active. Estimate counters from maxHp - current_hp.
        if opt.attack_id == 175:
            counters = ctx.get("opp_counters", 0)
            if counters > 0:
                dmg = 100 + counters * 30
        # Apply Weakness (×2) / Resistance so lethal detection catches KOs that
        # only happen *after* the type multiplier, and we prefer the attack that
        # actually does the most damage to this defender.
        dmg = gd.effective_damage(ctx["my_active_id"], dmg, ctx["opp_active_id"])
        if opp_hp is not None and dmg >= opp_hp > 0:
            return w["lethal_base"] + dmg  # lethal: take the KO
        # Attack ends the turn, so it ranks below every productive setup play
        # but above passing / retreating.
        # Include expected residual damage from Special Conditions (burn/poison).
        eff_dmg = dmg + gd.attack_effect_bonus(opt.attack_id)
        score = w["attack_base"] + eff_dmg * w["attack_dmg_scale"]
        score += _future_attack_value(
            opt.attack_id,
            ctx["my_active"],
            _active_pokemon(state, 1 - yi),
            gd,
        )
        # Prize-value bonus: targeting a high-prize active has intrinsic value
        # even without a KO (builds toward 2HKO on a Mega ex / ex).
        opp_pv = gd.prize_value(ctx["opp_active_id"])
        score += (opp_pv - 1) * w["attack_prize_bonus"]
        # Two-shot value: dealing ≥50% HP now means we KO next turn.
        if opp_hp is not None and opp_hp > 0 and dmg * 2 >= opp_hp:
            score += opp_pv * w["attack_twoshot_bonus"]
        # Late-game urgency: when either side is within 2 prizes, bypass
        # low-value setup and attack — the game is almost over.
        # Guard 0: missing state returns 0 prizes — treat as unknown, no urgency.
        my_prizes = ctx.get("my_prizes", 6)
        opp_prizes = ctx.get("opp_prizes", 6)
        if opp_prizes > 0 and (0 < my_prizes <= 2 or opp_prizes <= 2):
            score += w["attack_urgency"]
        return score
    if t == OptionType.ABILITY:
        return w["ability"]               # free value
    if t == OptionType.EVOLVE:
        # Board upgrade: unlocks higher attacks and more HP. Prefer evolving a
        # target that will be attack-ready (or 1 energy away) after evolution,
        # so the evolved Pokémon can threaten immediately.
        target = _attach_target(opt, me)  # reuse: resolves active/bench by index
        if target is not None:
            evo_cid = _card_at(hand, opt.hand_index)
            energies = target.get("energies") or []
            # Check if the *evolved* card can already attack with current energy.
            if evo_cid is not None:
                for aid in gd.card_attacks.get(evo_cid, []):
                    cost = gd.attack_cost(aid)
                    if cost and gd.can_pay(cost, energies):
                        return w["evolve"] + 80.0  # attack-ready post-evolution
                # One energy away from attacking: still a strong evolve.
                for aid in gd.card_attacks.get(evo_cid, []):
                    cost = gd.attack_cost(aid)
                    if cost and len(energies) + 1 >= len(cost):
                        return w["evolve"] + 30.0
        return w["evolve"]
    if t == OptionType.ATTACH:
        # Energy-need awareness: feed a Pokémon that still needs Energy to power
        # its hardest attack; an already-powered target is the lowest-priority
        # attach (but still above attacking, so we never attack prematurely just
        # to avoid a harmless extra attachment). Prefer the Active attacker.
        target = _attach_target(opt, me)
        if target is not None and not gd.needs_energy(
            target.get("id"), target.get("energies") or []
        ):
            return w["attach_powered"]
        # Never pour Energy into a pure *support* Pokémon (best attack deals no
        # damage, e.g. Fezandipiti ex). Attaching to it is wasted tempo — treat
        # it as the lowest-priority attach so the Energy is saved for a real
        # attacker. Guarded by ``attacker_dmg_floor`` (default 1) so Riolu
        # (10 dmg, evolves into Mega Lucario ex) is never starved. Only applies
        # when the target actually resolves — an unknown target is not assumed
        # to be support.
        tgt_best = gd.best_damage(target.get("id")) if target is not None else None
        if tgt_best is not None and tgt_best < w["attacker_dmg_floor"]:
            return w["attach_powered"]
        to_active = opt.in_play_area == AreaType.ACTIVE
        score = w["attach_active"] if to_active else w["attach_bench"]
        # Finishing the Active's attack this turn is worth a little extra.
        if to_active and target is not None and _completes_attack(target, gd):
            score += w["attach_completes"]
        # Prefer attaching to higher-damage attackers (Mega Lucario ex > Riolu).
        if target is not None:
            score += min((tgt_best or 0) * w["attach_dmg_scale"], w["attach_dmg_cap"])
            # Concentration: reward feeding a target that already has Energy
            # invested, so we finish charging ONE attacker rather than spreading
            # a single Energy/turn across several Pokémon (the observed loss
            # pattern where no attacker ever reached its attack cost).
            invested = len(target.get("energies") or [])
            score += invested * w["attach_concentrate"]
        score += _future_value(
            opt,
            hand,
            me,
            state,
            yi,
            gd,
        )
        return score
    if t == OptionType.PLAY:
        cid = _card_at(hand, opt.hand_index)
        if gd.is_basic_pokemon(cid):
            if not bench_room:
                return w["play_basic_noroom"]
            return w["play_basic_room"] + ctx["empty_slots"] * w["play_basic_empty"]
        if gd.is_supporter(cid):
            # Supporters are more valuable when hand is small — draw urgency.
            bonus = max(0.0, (4 - len(hand)) * 10.0)  # +10 per card below 4
            return w["play_supporter"] + bonus        # draw / search engine
        if gd.is_item(cid):
            # Dig items (balls, Poffin, Gear) should be played before the once-
            # per-turn Supporter when hand is small: search for Pokémon/Energy
            # first, then let the Supporter draw into a larger pool of options.
            if gd.is_dig_item(cid) and len(hand) <= 3:
                return w["play_supporter"] + 5.0
            # Recovery items (Night Stretcher / Tarragon): very high priority
            # when discard pile likely has Energy or key Pokémon (after mid-game).
            # Proxy: if hand is small (≤4), recovery items are worth playing.
            if len(hand) <= 4:
                return w["play_item"] + 20.0
            return w["play_item"]                 # ball / draw / recovery
        return w["play_other"]                    # tool / stadium / other
    if t == OptionType.RETREAT:
        # Retreat a doomed Active (the opponent can KO it next turn) only when a
        # fresh Bench attacker is ready to take over.
        if ctx["danger"] and ctx["bench_attacker"]:
            return w["retreat"] + w["retreat_danger"]
        # Promote out a pure *support* Active (0-damage, e.g. Fezandipiti ex) as
        # soon as a benched attacker is available: leaving it Active wastes the
        # turn since it cannot trade prizes. Ranks above attaching so we swap
        # first, then charge the real attacker that comes in.
        if ctx.get("active_is_support") and ctx["bench_attacker"]:
            return w["retreat"] + w["retreat_promote"]
        return w["retreat"]
    if t == OptionType.DISCARD:
        return w["discard"]
    if t == OptionType.END:
        return w["end"]                   # only when nothing productive remains
    return 1

def _future_value(
    opt: Option,
    hand: list,
    me: dict,
    state,
    yi: int,
    gd: GameData,
) -> float:
    """
    One-ply board evaluation.
    Simulate this action, then score the resulting board.
    """
    sim = _simulate_action(
        opt,
        hand,
        me,
        gd,
    )
    return _evaluate_board(
        sim,
        gd,
    )

def _future_attack_value(
    attack_id: int,
    my_active: dict | None,
    opp_active: dict |None,
    gd: GameData,
) -> float:
    """
    Strategic value beyond immediate damage.
    Rewards attacks that improve future board state, not just HP removal.
    """
    atk = gd.attack(attack_id)
    if atk is None:
        return 0.0
    value = 0.0
    # ------------------------
    # Future KO pressure
    # ------------------------
    if my_active is not None and opp_active is not None:
        hp = opp_active.get("hp") or 0
        dmg = gd.effective_damage(
            attacker_id=my_active.get("id"),
            attack_id=attack_id,
            base_dmg=atk.damage,
            defender_id=opp_active.get("id"),
        )
        if hp > 0:
            if dmg >= hp:
                value += 90.0
            elif dmg * 2 >= hp:
                value += 38.0
            elif dmg * 3 >= hp:
                value += 14.0
    # ------------------------
    # Draw
    # ------------------------
    value += gd.attack_draw(attack_id) * 7.0
    # ------------------------
    # Healing
    # ------------------------
    value += gd.attack_heal(attack_id) * 0.45
    # ------------------------
    # Bench pressure
    # ------------------------
    bench = gd.attack_bench_damage(attack_id)
    value += bench * 0.30
    if gd.attack_spread(attack_id):
        value += 22.0
    # ------------------------
    # Gust
    # ------------------------
    if gd.attack_gust(attack_id):
        value += 55.0
    # ------------------------
    # Pivot attacks
    # ------------------------
    if gd.attack_switch(attack_id):
        value += 25.0
    # ------------------------
    # Energy acceleration
    # ------------------------
    if gd.attack_energy_acceleration(attack_id):
        value += 50.0
    # ------------------------
    # Energy denial
    # ------------------------
    if gd.attack_discards_opponent_energy(attack_id):
        value += 35.0
    if gd.attack_discards_self_energy(attack_id):
        value -= 15.0
    # ------------------------
    # Status
    # ------------------------
    value += gd.attack_status_score(attack_id)
    return value

# ---------------------------------------------------------------------------
# Card selections (setup / search / discard)
# # ---------------------------------------------------------------------------
def _choose_cards(obs_dict, select: Select, gd: GameData) -> list[int]:
    state = obs_dict.get("current") if isinstance(obs_dict, dict) else None
    yi = state.get("yourIndex", 0) if isinstance(state, dict) else 0
    # EVOLVES_FROM: which pre-evolution to evolve from. Prefer the copy with
    # the most energy already attached — it carries over to the evolved form
    # and saves setup turns. Same card ID for all options, so _card_value alone
    # can't differentiate: need to look at the actual board Pokémon.
    if select.context == SelectContext.EVOLVES_FROM:
        scored = []
        for opt in select.options:
            pkmn = _option_pokemon(opt, state, yi)
            attached = len(pkmn.get("energies") or []) if isinstance(pkmn, dict) else 0
            scored.append((attached, opt.index))
        scored.sort(reverse=True)
        k = max(select.min_count, 1)
        k = min(k, select.max_count or k, len(select.options))
        return sorted(idx for _, idx in scored[:k])
    # Gust selection (e.g. Boss's Orders): every option targets an opponent
    # Pokémon. Pick the one we can KO now (prize), preferring an ex (2 Prizes).
    opp = 1 - yi
    if select.options and all(
        o.player_index == opp for o in select.options
    ):
        return _choose_gust_target(select, state, yi, gd)
    # Bench-snipe target selection (e.g. Cruel Arrow / spread damage attacks):
    # options target opponent's bench specifically. Score KO-able targets highest,
    # then prefer ex/Mega-ex for prize value, then lowest HP.
    if select.options and all(
        o.player_index == opp and o.in_play_area == AreaType.BENCH
        for o in select.options
        if o.player_index is not None
    ) and any(o.player_index == opp for o in select.options):
        my_active = _active_pokemon(state, yi)
        scored = []
        for opt in select.options:
            target = _option_pokemon(opt, state, opp)
            if not isinstance(target, dict):
                scored.append((0.0, opt.index))
                continue
            hp = target.get("hp") or 0
            pv = gd.prize_value(target.get("id"))
            # Fixed 100 bench damage (Cruel Arrow / similar), adjusted for dmg.
            bench_dmg = 100
            v = pv * 20.0 - hp * 0.05
            if 0 < hp <= bench_dmg:
                v += 200.0 + pv * 50.0  # KO on the bench — massive value
            scored.append((v, opt.index))
        scored.sort(reverse=True)
        k = max(select.min_count, 1)
        k = min(k, select.max_count or k, len(select.options))
        return sorted(idx for _, idx in scored[:k])
    # Switch-in selection: prefer already-powered attackers over cold ones.
    # This fires when we choose which Bench Pokémon to bring Active (after a KO
    # or a manual retreat), and gets the highest-value ready attacker in.
    if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        scored = []
        for opt in select.options:
            pkmn = _option_pokemon(opt, state, yi)
            if pkmn is not None:
                val = _powered_switch_value(pkmn, gd)
            else:
                # Fallback for hand cards (e.g. trainer effect TO_ACTIVE).
                cid = _resolve_card_id(opt, obs_dict, select, state, yi)
                val = _card_value(cid, gd)
            scored.append((val, opt.index))
        scored.sort(reverse=True)
        k = max(select.min_count, 1)
        k = min(k, select.max_count or k, len(select.options))
        return sorted(idx for _, idx in scored[:k])
    acquire = select.context in _ACQUIRE_CONTEXTS
    scored = []
    for opt in select.options:
        cid = _resolve_card_id(opt, obs_dict, select, state, yi)
        scored.append((_card_value(cid, gd), opt.index))
    # Acquire: take the most valuable; otherwise take the least valuable.
    scored.sort(reverse=acquire)
    n = len(select.options)
    # Only grab the extra (optional) cards when the context is clearly
    # beneficial; for discards and unknown contexts take the minimum required.
    k = (select.max_count or select.min_count) if acquire else select.min_count
    k = max(select.min_count, min(k, n))
    return sorted(idx for _, idx in scored[:k])

def _card_value(
    cid,
    gd: GameData,
) -> float:
    if cid is None:
        return 1.0
    if gd.is_basic_pokemon(cid):
        value = 100.0
        value += gd.best_damage(cid) * 0.12
        if gd.is_ex(cid):
            value += 35.0
        if gd.is_mega(cid):
            value += 30.0
        return value
    if gd.is_pokemon(cid):
        value = 75.0
        value += gd.best_damage(cid) * 0.14
        if gd.is_ex(cid):
            value += 40.0
        if gd.is_mega(cid):
            value += 40.0
        return value
    if gd.is_supporter(cid):
        return 42.0
    if gd.is_item(cid):
        if gd.is_dig_item(cid):
            return 38.0
        return 30.0
    if gd.is_energy(cid):
        return 16.0
    return 10.0

# ---------------------------------------------------------------------------
# Simple selections
# ---------------------------------------------------------------------------
def _choose_yes_no(select: Select) -> int:
    """Yes/No decisions.
    Default to YES — the engine asks Yes/No to *activate* beneficial effects. The
    one deliberate exception is the go-first/second coin choice (IS_FIRST), which
    we resolve via :data:`PREFER_FIRST`.
    """
    if select.context == SelectContext.IS_FIRST:
        want = OptionType.YES if PREFER_FIRST else OptionType.NO
        for opt in select.options:
            if opt.type == want:
                return opt.index
    for opt in select.options:
        if opt.type == OptionType.YES:
            return opt.index
    return select.options[0].index

def _choose_lowest(select: Select) -> list[int]:
    k = max(select.min_count, 1)
    k = min(k, select.max_count or k, len(select.options))
    return list(range(k))

def _fallback(select: Select) -> list[int]:
    """Return a contract-safe fallback selection.
    maxCount is always treated as the hard upper bound.
    """
    n = len(select.options)
    min_count = max(
        0,
        min(
            int(select.min_count),
            n,
        ),
    )
    max_count = max(
        0,
        min(
            int(select.max_count),
            n,
        ),
    )
    # Normal contract.
    if min_count <= max_count:
        # MAIN decisions generally require one action when optional
        # selection is otherwise allowed.
        if (
            min_count == 0
            and max_count >= 1
            and select.select_type == SelectType.MAIN
        ):
            return [0]
        return list(
            range(min_count)
        )
    # Impossible contract: never exceed maxCount.
    return list(
        range(max_count)
    )

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _player(state, yi):
    return player(state, yi)

def _attach_target(opt: Option, me: dict) -> dict | None:
    """Resolve the Pokémon an ATTACH option targets (Active or a Bench slot)."""
    idx = opt.in_play_index
    if idx is None:
        return None
    field = "active" if opt.in_play_area == AreaType.ACTIVE else "bench"
    arr = me.get(field) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx]
    return None

def _clone_player(me: dict) -> dict:
    """Cheap deep copy of only the player structure we mutate."""
    import copy
    return copy.deepcopy(me)

def _simulate_attach(
    opt: Option,
    hand: list,
    me: dict,
    gd: GameData,
) -> dict:
    sim = _clone_player(me)
    target = _attach_target(opt, sim)
    if target is None:
        return sim
    cid = _card_at(hand, opt.hand_index)
    energies = list(target.get("energies") or [])
    provides = gd.energy_provides(cid)
    if not provides:
        provides = ["C"]
    energies.append(
        {
            "id": cid,
            "provides": provides,
        }
    )
    target["energies"] = energies
    return sim

def _simulate_play_basic(
    opt: Option,
    hand: list,
    me: dict,
) -> dict:
    sim = _clone_player(me)
    cid = _card_at(hand, opt.hand_index)
    if cid is None:
        return sim
    bench = sim.setdefault("bench", [])
    if len(bench) >= 5:
        return sim
    bench.append(
        {
            "id": cid,
            "energies": [],
        }
    )
    return sim

def _simulate_evolve(
    opt: Option,
    hand: list,
    me: dict,
) -> dict:
    sim = _clone_player(me)
    evo = _card_at(hand, opt.hand_index)
    if evo is None:
        return sim
    target = _attach_target(opt, sim)
    if target is None:
        return sim
    target["id"] = evo
    return sim

def _simulate_action(
    opt: Option,
    hand: list,
    me: dict,
    gd: GameData,
) -> dict:
    if opt.type == OptionType.ATTACH:
        return _simulate_attach(opt, hand, me, gd)
    if opt.type == OptionType.PLAY:
        cid = _card_at(hand, opt.hand_index)
        if gd.is_basic_pokemon(cid):
            return _simulate_play_basic(opt, hand, me)
        return me
    if opt.type == OptionType.EVOLVE:
        return _simulate_evolve(opt, hand, me)
    return me

def _evaluate_board(
    me: dict,
    gd: GameData,
) -> float:
    """
    Evaluate a simulated board after one action.
    Larger is better.
    """
    score = 0.0
    # ---------- Active ----------
    active = me.get("active") or []
    if active:
        p = active[0]
        cid = p.get("id")
        energies = p.get("energies") or []
        best = gd.best_damage(cid)
        score += best * 0.55
        if not gd.needs_energy(cid, energies):
            score += 220.0
        score += len(energies) * 14.0
        if gd.is_ex(cid):
            score += 20.0
    # ---------- Bench ----------
    for p in me.get("bench") or []:
        cid = p.get("id")
        energies = p.get("energies") or []
        best = gd.best_damage(cid)
        score += best * 0.22
        if not gd.needs_energy(cid, energies):
            score += 90.0
        score += len(energies) * 5.0
        if gd.is_ex(cid):
            score += 8.0
    # Slight reward for wider board.
    score += len(me.get("bench") or []) * 12.0
    return score

# ---------------------------------------------------------------------------
# Threat model / board reading (for the danger-aware features)
# ---------------------------------------------------------------------------
def _active_pokemon(state, pi):
    return active(state, pi)

def _active_card_id(state, pi):
    return active_card_id(state, pi)

def _best_affordable_damage(
    pkmn: dict | None,
    gd: GameData,
    defender: dict | None = None,
) -> int:
    """
    Highest effective damage currently available.
    Includes attack-effect bonus so tactical attacks compete fairly with
    raw-damage attacks.
    """
    if not isinstance(pkmn, dict):
        return 0
    cid = pkmn.get("id")
    if cid is None:
        return 0
    attached = pkmn.get("energies") or []
    defender_id = (
        defender.get("id")
        if isinstance(defender, dict)
        else None
    )
    best = 0
    for attack_id in gd.card_attacks.get(cid, ()):
        cost = gd.attack_cost(attack_id)
        if not gd.can_pay(cost, attached):
            continue
        dmg = gd.attack_damage(attack_id)
        if defender_id is not None:
            dmg = gd.effective_damage(
                cid,
                dmg,
                defender_id,
            )
        dmg += gd.attack_effect_bonus(attack_id)
        if dmg > best:
            best = dmg
    return best

def _in_danger(state, yi, gd: GameData) -> bool:
    """True if the opponent's Active can KO our Active on its next turn."""
    mine = _active_pokemon(state, yi)
    opp = _active_pokemon(state, 1 - yi)
    if mine is None or opp is None:
        return False
    my_hp = mine.get("hp")
    if not isinstance(my_hp, (int, float)) or my_hp <= 0:
        return False
    # Weakness cuts both ways: account for *our* Active's Weakness to the
    # opponent's type when judging whether we are about to be KO'd.
    return _best_affordable_damage(opp, gd, defender=mine) >= my_hp

def _has_bench_attacker(state, yi, gd: GameData) -> bool:
    """True if a healthy Benched Pokémon could take over as attacker."""
    for p in _player(state, yi).get("bench") or []:
        if isinstance(p, dict) and (p.get("hp") or 0) > 0 and gd.best_damage(p.get("id")) > 0:
            return True
    return False

def _bench_empty_slots(state, yi):
    return bench_slots(state, yi)

def _completes_attack(pkmn: dict, gd: GameData) -> bool:
    """True if one more Energy likely finishes the Pokémon's cheapest attack."""
    cid = pkmn.get("id")
    attached = pkmn.get("energies") or []
    costs = [gd.attack_cost(a) for a in gd.card_attacks.get(cid, [])]
    costs = [c for c in costs if c]
    if not costs:
        return False
    cheapest = min(costs, key=len)
    if gd.can_pay(cheapest, attached):
        return False  # already able to attack
    return len(attached) + 1 >= len(cheapest)

def _option_pokemon(opt: Option, state, default_pi: int) -> dict | None:
    """Resolve the board Pokémon an option references (active or bench)."""
    pi = opt.player_index if opt.player_index is not None else default_pi
    area = opt.in_play_area if opt.in_play_area is not None else opt.area
    idx = opt.in_play_index if opt.in_play_index is not None else opt.hand_index
    field = _AREA_FIELD.get(area)
    if field not in ("active", "bench") or not isinstance(idx, int):
        return None
    arr = _player(state, pi).get(field) or []
    if 0 <= idx < len(arr) and isinstance(arr[idx], dict):
        return arr[idx]
    return None

def _choose_gust_target(select: Select, state, yi, gd: GameData) -> list[int]:
    """Pick the best opponent Pokémon to drag up (Boss's Orders / gust)."""
    opp = 1 - yi
    my_active = _active_pokemon(state, yi)
    scored = []
    for opt in select.options:
        target = _option_pokemon(opt, state, opp)
        scored.append((_gust_value(target, my_active, gd), opt.index))
    scored.sort(reverse=True)
    n = len(select.options)
    k = max(select.min_count, 1)
    k = min(k, select.max_count or k, n)
    return sorted(idx for _, idx in scored[:k])

def _gust_value(
    target: dict | None,
    my_active: dict | None,
    gd: GameData,
) -> float:
    """
    Score an opponent Pokémon for gust effects.
    Priorities:
        1. Immediate KO
        2. Prize value
        3. Damage output
        4. Lowest remaining HP
    """
    if not isinstance(target, dict):
        return 0.0
    hp = target.get("hp") or 0
    cid = target.get("id")
    my_dmg = _best_affordable_damage(
        my_active,
        gd,
        defender=target,
    )
    value = 0.0
    prize = gd.prize_value(cid)
    if hp > 0 and my_dmg >= hp:
        value += WEIGHTS["gust_ko"]
        value += prize * 80.0
    else:
        value += prize * 35.0
    value += gd.best_damage(cid) * 0.18
    value -= hp * 0.05
    if gd.is_mega(cid):
        value += WEIGHTS["gust_mega_ex"]
    elif gd.is_ex(cid):
        value += WEIGHTS["gust_ex"]
    return value

def _hand(state, yi):
    return hand(state, yi)

def _card_at(hand, idx):
    if idx is None or not isinstance(hand, list) or not (0 <= idx < len(hand)):
        return None
    entry = hand[idx]
    return entry.get("id") if isinstance(entry, dict) else None

def _opponent_active_hp(state, yi):
    return active_hp(state, opponent_index(yi))

def _bench_has_room(state, yi):
    return bench_slots(state, yi) > 0

_AREA_FIELD = {
    AreaType.HAND: "hand",
    AreaType.BENCH: "bench",
    AreaType.DISCARD: "discard",
    AreaType.ACTIVE: "active",
    AreaType.PRIZE: "prize",
}
def _resolve_card_id(opt: Option, obs_dict, select: Select, state, yi):
    """Best-effort mapping of a CARD option to its underlying card id."""
    if opt.card_id is not None:
        return opt.card_id
    idx = opt.hand_index
    # Cards drawn from the deck being searched are listed in select.deck.
    if select.deck and idx is not None and 0 <= idx < len(select.deck):
        entry = select.deck[idx]
        if isinstance(entry, dict):
            return entry.get("id")
    # Otherwise resolve via (area, index) on the owning player.
    area = opt.area
    field = _AREA_FIELD.get(area)
    if field is None or idx is None or not isinstance(state, dict):
        return None
    players = state.get("players") or []
    pi = opt.player_index if opt.player_index is not None else yi
    if pi >= len(players) or not isinstance(players[pi], dict):
        return None
    arr = players[pi].get(field) or []
    if not (0 <= idx < len(arr)) or not isinstance(arr[idx], dict):
        return None
    return arr[idx].get("id")

def _can_attack(pkmn: dict, gd: GameData) -> bool:
    cid = pkmn.get("id")
    if cid is None:
        return False
    attached = pkmn.get("energies") or []
    for aid in gd.card_attacks.get(cid, []):
        if gd.can_pay(gd.attack_cost(aid), attached):
            return True
    return False

def _powered_switch_value(
    pkmn: dict,
    gd: GameData,
) -> float:
    """
    Score a Pokémon as a switch target.
    Priorities:
      1. Already able to attack
      2. High damage
      3. Healthy
      4. Already has energy
    """
    hp = pkmn.get("hp") or 0
    cid = pkmn.get("id")
    energies = pkmn.get("energies") or []
    damage = gd.best_damage(cid)
    value = damage * 0.30
    if _can_attack(pkmn, gd):
        value += 150.0
    value += len(energies) * 15.0
    value += min(hp, 220) * 0.08
    if gd.is_ex(cid):
        value += 15.0
    if hp <= 40:
        value -= 45.0
    return value