"""
brain_state.py
Shared board representation used by every Brain module.
Planner constructs ONE BrainState every turn.
Every specialist (Tactical, Strategic, Predictive, Adaptive,
Supervisor) receives the same immutable object.
This file contains NO search.
NO evaluation.
NO engine calls.
It only converts observation -> structured state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
# ---------------------------------------------------------
# Pokemon representation
# ---------------------------------------------------------
@dataclass(slots=True)
class PokemonState:
    id: int | None
    hp: int
    damage: int
    energies: int
    can_attack: bool
    evolved: bool
    tools: tuple = ()
    special_conditions: tuple = ()
    prize_value: int = 1

# ---------------------------------------------------------
# Player representation
# ---------------------------------------------------------
@dataclass(slots=True)
class PlayerState:
    active: PokemonState | None
    bench: list[PokemonState]
    prizes_remaining: int
    deck_count: int
    hand_count: int
    discard_count: int
    energy_in_play: int
    supporter_played: bool

# ---------------------------------------------------------
# Brain state
# ---------------------------------------------------------
@dataclass(slots=True)
class BrainState:
    me: PlayerState
    opponent: PlayerState
    turn: int
    turn_player: int
    phase: str
    tempo: float
    winning: bool
    losing: bool
    stadium: int | None
    game_result: int
    metadata: dict = field(default_factory=dict)

# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------
class BrainBuilder:
    def __init__(
        self,
        gamedata,
    ):
        self.gamedata = gamedata
    # -----------------------------------------------------

    def build(
        self,
        state,
        me,
    ) -> BrainState:
        players = state["players"]
        my_player = players[me]
        opp_player = players[1 - me]
        me_state = self._player(my_player)
        opp_state = self._player(opp_player)
        tempo = (
            me_state.prizes_remaining
            - opp_state.prizes_remaining
        )
        winning = tempo < 0
        losing = tempo > 0
        return BrainState(
            me=me_state,
            opponent=opp_state,
            turn=state.get("turn", 0),
            turn_player=state.get("turnPlayer", 0),
            phase=self._phase(
                me_state,
                opp_state,
            ),
            tempo=float(tempo),
            winning=winning,
            losing=losing,
            stadium=state.get("stadium"),
            game_result=state.get(
                "result",
                -1,
            ),
        )

    # -----------------------------------------------------
    def _player(
        self,
        player,
    ):
        active = self._pokemon(
            self._active(player)
        )
        bench = [
            self._pokemon(mon)
            for mon in player.get("bench") or []
            if isinstance(mon, dict)
        ]
        energy = 0
        if active:
            energy += active.energies
        for mon in bench:
            energy += mon.energies
        return PlayerState(
            active=active,
            bench=bench,
            prizes_remaining=len(
                player.get("prize") or []
            ),
            deck_count=player.get(
                "deckCount",
                0,
            ),
            hand_count=player.get(
                "handCount",
                0,
            ),
            discard_count=len(
                player.get("discard") or []
            ),
            energy_in_play=energy,
            supporter_played=player.get(
                "supporterPlayed",
                False,
            ),
        )

    # -----------------------------------------------------
    def _pokemon(
        self,
        mon,
    ):
        if not isinstance(mon, dict):
            return None
        cid = mon.get("id")
        energies = len(
            mon.get("energies") or []
        )
        return PokemonState(
            id=cid,
            hp=mon.get("hp", 0),
            damage=mon.get("damage", 0),
            energies=energies,
            can_attack=self._can_attack(
                cid,
                energies,
            ),
            evolved=bool(
                mon.get("preEvolution")
            ),
            tools=tuple(
                mon.get("tools") or []
            ),
            special_conditions=tuple(
                mon.get(
                    "specialConditions"
                )
                or []
            ),
        )

    # -----------------------------------------------------
    def _phase(
        self,
        me,
        opp,
    ):
        total = (
            me.prizes_remaining
            + opp.prizes_remaining
        )
        if total >= 10:
            return "opening"
        if total >= 5:
            return "midgame"
        return "endgame"

    # -----------------------------------------------------
    def _can_attack(
        self,
        card_id,
        energy_count,
    ):
        if card_id is None:
            return False
        attacks = self.gamedata.card_attacks.get(
            card_id,
            [],
        )
        for attack in attacks:
            cost = self.gamedata.attack_cost(
                attack,
            )
            if len(cost) <= energy_count:
                return True
        return False

    # -----------------------------------------------------
    @staticmethod
    def _active(
        player,
    ):
        arr = player.get("active") or []
        if arr and isinstance(arr[0], dict):
            return arr[0]
        return None