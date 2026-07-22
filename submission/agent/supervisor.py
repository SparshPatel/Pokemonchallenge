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

class Supervisor:
    """
    Tournament-level supervisor.
    Responsibilities
    ----------------
    1. Force tactical wins.
    2. Prevent obvious blunders.
    3. Calibrate the ValueNet online.
    """
    def __init__(self, value_net=None):
        self.value_net = value_net
        self.games = 0
        self.correct = 0
        self.predictions = []
        
    def register_prediction(
        self,
        probability,
    ):
        self.predictions.append(
            probability
        )
        
    def finish_game(
        self,
        won,
    ):
        """
        Called once at game end.
        """
        if not self.predictions:
            return
        outcome = 1.0 if won else 0.0
        if self.value_net is not None:
            for p in self.predictions:
                self.value_net.calibrate(
                    p,
                    outcome,
                )
        self.games += 1
        self.predictions.clear()

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
    
def forced_retreat_survival(
    obs_dict,
    select: Select,
    gd: GameData,
) -> list[int] | None:
    """
    If our Active is guaranteed to be KO'd next turn and we have a legal
    retreat into a healthier Pokémon, force the retreat.
    This only triggers when:
      • retreat is legal
      • opponent has lethal
      • bench contains a better survivor
    """
    try:
        if select.select_type != SelectType.MAIN:
            return None
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return None
        yi = state.get("yourIndex", 0)
        me = _player(state, yi)
        opp = _player(state, 1 - yi)
        active = _active(me)
        opp_active = _active(opp)
        if active is None or opp_active is None:
            return None
        active_hp = (
            active.get("hp", 0)
            - active.get("damage", 0)
        )
        incoming = 0
        for aid in gd.card_attacks.get(
            opp_active.get("id"),
            [],
        ):
            if not gd.can_pay(
                gd.attack_cost(aid),
                opp_active.get("energies") or [],
            ):
                continue
            dmg = gd.effective_damage(
                opp_active.get("id"),
                gd.attack_damage(aid),
                active.get("id"),
            )
            incoming = max(incoming, dmg)
        if incoming < active_hp:
            return None
        bench = me.get("bench") or []
        if not bench:
            return None
        healthiest = max(
            bench,
            key=lambda b:
                b.get("hp", 0)
                - b.get("damage", 0),
        )
        healthy_hp = (
            healthiest.get("hp", 0)
            - healthiest.get("damage", 0)
        )
        if healthy_hp <= active_hp:
            return None
        for option in select.options:
            if option.type == OptionType.RETREAT:
                return [option.index]
        return None
    except Exception:
        return None
    
def forced_prize_trade(
    obs_dict,
    select: Select,
    gd: GameData,
) -> list[int] | None:
    """
    If multiple attacks KO this turn, prefer the one that gives
    the best prize trade.
    Priority:
        1. More prizes taken
        2. Less self-risk
        3. Higher damage
    """
    try:
        if select.select_type != SelectType.MAIN:
            return None
        lethals = _lethal_options(
            obs_dict,
            select,
            gd,
        )
        if len(lethals) < 2:
            return None
        state = obs_dict.get("current")
        yi = state.get("yourIndex", 0)
        me = _player(state, yi)
        active = _active(me)
        my_prize_value = gd.prize_value(
            active.get("id"),
        )
        scored = []
        for idx, dmg, opp_prize in lethals:
            trade = opp_prize - my_prize_value
            scored.append(
                (
                    trade,
                    opp_prize,
                    dmg,
                    idx,
                )
            )
        scored.sort(reverse=True)
        return [scored[0][3]]
    except Exception:
        return None
    
def forced_ability_order(
    obs_dict,
    select: Select,
) -> list[int] | None:
    """
    Always use free abilities before Items/Supporters/Attack
    when they are available.
    This prevents the planner from attacking or ending the turn
    while leaving free value on the table.
    """
    try:
        if select.select_type != SelectType.MAIN:
            return None
        abilities = []
        for option in select.options:
            if option.type != OptionType.ABILITY:
                continue
            abilities.append(option.index)
        if not abilities:
            return None
        if len(select.options) == len(abilities):
            return [abilities[0]]
        return None
    except Exception:
        return None
    
def forced_energy_conservation(
    obs_dict,
    select: Select,
    gd: GameData,
) -> list[int] | None:
    """
    Avoid discarding valuable Energy when another legal discard
    exists.
    Priority:
        Basic Energy
            >
        Double Turbo
            >
        Special Energy
    """
    try:
        if select.select_type != SelectType.CHOOSE_DISCARD:
            return None
        safe = []
        risky = []
        for option in select.options:
            card = getattr(
                option,
                "card",
                None,
            )
            if not isinstance(card, dict):
                continue
            cid = card.get("id")
            if cid is None:
                continue
            if gd.is_basic_energy(cid):
                safe.append(option.index)
                continue
            if gd.is_special_energy(cid):
                risky.append(option.index)
                continue
            safe.append(option.index)
        if safe:
            return [safe[0]]
        return None
    except Exception:
        return None
    
def forced_bench_preservation(
    obs_dict,
    select: Select,
    gd: GameData,
) -> list[int] | None:
    """
    Avoid attaching or evolving onto a Pokémon that is almost
    certainly getting KO'd next turn if a safer target exists.
    """
    try:
        if select.select_type != SelectType.MAIN:
            return None
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return None
        yi = state.get("yourIndex", 0)
        me = _player(state, yi)
        opp = _player(state, 1 - yi)
        opp_active = _active(opp)
        if opp_active is None:
            return None
        threatened = []
        for mon in me.get("bench") or []:
            hp = (
                mon.get("hp", 0)
                - mon.get("damage", 0)
            )
            incoming = 0
            for aid in gd.card_attacks.get(
                opp_active.get("id"),
                [],
            ):
                if not gd.can_pay(
                    gd.attack_cost(aid),
                    opp_active.get("energies") or [],
                ):
                    continue
                dmg = gd.effective_damage(
                    opp_active.get("id"),
                    gd.attack_damage(aid),
                    mon.get("id"),
                )
                incoming = max(incoming, dmg)
            if incoming >= hp:
                threatened.append(mon.get("id"))
        if not threatened:
            return None
        for option in select.options:
            if option.type not in (
                OptionType.ATTACH,
                OptionType.EVOLVE,
            ):
                continue
            target = getattr(
                option,
                "target",
                None,
            )
            if (
                isinstance(target, dict)
                and target.get("id") in threatened
            ):
                continue
            return [option.index]
        return None
    except Exception:
        return None
    
def forced_boss_finish(
    obs_dict,
    select: Select,
    gd: GameData,
) -> list[int] | None:
    """
    If a gust effect (Boss's Orders / Counter Catcher /
    Prime Catcher / etc.) creates an immediate KO on a
    benched Pokémon, always force it.
    """
    try:
        if select.select_type != SelectType.MAIN:
            return None
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return None
        yi = state.get("yourIndex", 0)
        me = _player(state, yi)
        opp = _player(state, 1 - yi)
        my_active = _active(me)
        if my_active is None:
            return None
        attached = my_active.get("energies") or []
        best = None
        for option in select.options:
            if option.type != OptionType.PLAY:
                continue
            text = str(
                getattr(option, "text", "")
            ).lower()
            if not any(
                k in text
                for k in (
                    "boss",
                    "catcher",
                    "switch your opponent",
                    "gust",
                )
            ):
                continue
            for bench in opp.get("bench") or []:
                hp = (
                    bench.get("hp", 0)
                    - bench.get("damage", 0)
                )
                best_damage = 0
                for aid in gd.card_attacks.get(
                    my_active.get("id"),
                    [],
                ):
                    if not gd.can_pay(
                        gd.attack_cost(aid),
                        attached,
                    ):
                        continue
                    dmg = gd.effective_damage(
                        my_active.get("id"),
                        gd.attack_damage(aid),
                        bench.get("id"),
                    )
                    best_damage = max(
                        best_damage,
                        dmg,
                    )
                if best_damage >= hp:
                    prize = gd.prize_value(
                        bench.get("id"),
                    )
                    score = (
                        prize * 1000
                        + hp
                    )
                    if (
                        best is None
                        or score > best[0]
                    ):
                        best = (
                            score,
                            option.index,
                        )
        if best is None:
            return None
        return [best[1]]
    except Exception:
        return None
    
def strategic_override(
    obs_dict,
    select: Select,
    gd: GameData,
    choice: list[int],
) -> list[int] | None:
    try:
        boss = forced_boss_finish(
            obs_dict,
            select,
            gd,
        )
        if boss is not None:
            return boss
        survive = forced_retreat_survival(
            obs_dict,
            select,
            gd,
        )
        if survive is not None:
            return survive
        trade = forced_prize_trade(
            obs_dict,
            select,
            gd,
        )
        if trade is not None:
            return trade
        bench = forced_bench_preservation(
            obs_dict,
            select,
            gd,
        )
        if bench is not None:
            return bench
        ability = forced_ability_order(
            obs_dict,
            select,
        )
        if ability is not None:
            return ability
        energy = forced_energy_conservation(
            obs_dict,
            select,
            gd,
        )
        if energy is not None:
            return energy
        return None
    except Exception:
        return None

def guard_main(obs_dict, select: Select, gd: GameData, choice: list[int]) -> list[int]:
    """
    Highest-priority tactical override.
    If an affordable attack immediately Knocks Out the opponent's Active,
    always take it.
    This overrides:
      • END TURN
      • RETREAT
      • ITEM
      • SUPPORTER
      • STADIUM
      • TOOL
      • EVOLUTION
      • ENERGY
      • Ability-first sequencing
    There is almost never a stronger play than removing the opponent's
    Active immediately.
    """
    try:
        if select.select_type != SelectType.MAIN:
            return choice
        lethals = _lethal_options(
            obs_dict,
            select,
            gd,
        )
        if not lethals:
            return choice
        lethals.sort(
            key=lambda x: (
                x[2],
                x[1],
                x[0],
            ),
            reverse=True,
        )
        best_attack = lethals[0][0]
        if choice and choice[0] == best_attack:
            return choice
        return [best_attack]
    except Exception:
        return choice