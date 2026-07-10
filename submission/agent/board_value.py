from __future__ import annotations
from dataclasses import dataclass
TERMINAL_WIN = 100000.0
@dataclass(slots=True)
class BoardWeights:
    prize_weight: float = 400.0
    hp_weight: float = 0.40
    energy_weight: float = 70.0
    retreat_weight: float = 60.0
    lost_energy_weight: float = 80.0
    active_bonus: float = 35.0
    attack_ready_bonus: float = 140.0
    tempo_bonus: float = 120.0

@dataclass(slots=True)
class BoardEvaluator:
    gamedata: object
    weights: BoardWeights = BoardWeights()
    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def evaluate(
        self,
        state: dict,
        me: int,
    ) -> float:
        players = state["players"]
        mine = players[me]
        opp = players[1 - me]
        score = 0.0
        score += self.prize_value(
            mine,
            opp,
        )
        score += self.board_value(
            mine,
            opp,
        )
        score -= self.board_value(
            opp,
            mine,
        )
        score += self.tempo_value(
            mine,
            opp,
        )
        return score
    
    # ---------------------------------------------------------
    # Prize race
    # ---------------------------------------------------------
    def prize_value(
        self,
        mine,
        opp,
    ):
        my_left = len(
            mine.get("prize") or []
        )
        opp_left = len(
            opp.get("prize") or []
        )
        return (
            my_left - opp_left
        ) * self.weights.prize_weight
    
    # ---------------------------------------------------------
    # Board value
    # ---------------------------------------------------------
    def board_value(
        self,
        player,
        opponent,
    ):
        value = 0.0
        opp_active = self.active(
            opponent,
        )
        active = self.active(
            player,
        )
        if active is not None:
            value += self.pokemon_value(
                active,
                opponent=opp_active,
                active=True,
            )
        for mon in player.get(
            "bench"
        ) or []:
            value += self.pokemon_value(
                mon,
                opponent=opp_active,
                active=False,
            )
        return value

    # ---------------------------------------------------------
    # Individual Pokémon value
    # ---------------------------------------------------------
    def pokemon_value(
        self,
        pokemon,
        opponent=None,
        active=False,
    ):
        if pokemon is None:
            return 0.0
        hp = pokemon.get(
            "hp",
            0,
        )
        damage_taken = pokemon.get(
            "damage",
            0,
        )
        remaining_hp = max(
            0,
            hp - damage_taken,
        )
        energies = len(
            pokemon.get("energies") or []
        )
        retreat = pokemon.get(
            "retreatCost",
            0,
        )
        score = 0.0
        score += (
            remaining_hp
            * self.weights.hp_weight
        )
        score += self.energy_value(
            pokemon,
        )
        score -= self.retreat_cost(
            retreat,
            energies,
        )
        if active:
            score += self.weights.active_bonus
        damage = self.best_attack_damage(
            pokemon,
        )
        if damage > 0:
            score += self.weights.attack_ready_bonus
            score += damage * 0.45
            if remaining_hp > hp * 0.65:
                score += damage * 0.20
            # NEW:
            # fully powered attackers become
            # valuable long-term resources
            cid = pokemon.get("id")
            attacks = self.gamedata.card_attacks.get(
                cid,
                [],
            )
            if attacks:
                hardest_cost = max(
                    (
                        len(
                            self.gamedata.attack_cost(
                                attack,
                            )
                        )
                        for attack in attacks
                    ),
                    default=0,
                )
                if hardest_cost > 0:
                    ratio = min(
                        energies / hardest_cost,
                        1.5,
                    )
                    score += (
                        ratio
                        * 180.0
                    )
        if opponent is not None:
            incoming = self._best_attack_damage(
                opponent,
                pokemon,
            )
            if incoming >= remaining_hp:
                score *= 0.40
            elif incoming >= remaining_hp * 0.75:
                score *= 0.70
        return score

    # ---------------------------------------------------------
    # Energy value
    # --------------------------------------------------------
    def energy_value(
        self,
        pokemon,
    ):
        if pokemon is None:
            return 0.0
        cid = pokemon.get("id")
        if cid is None:
            return 0.0
        attached = pokemon.get(
            "energies",
            [],
        )
        attached_count = len(attached)
        attacks = self.gamedata.card_attacks.get(
            cid,
            [],
        )
        if not attacks:
            return (
                attached_count
                * self.weights.energy_weight
                * 0.50
            )
        best_score = 0.0
        for attack in attacks:
            cost = len(
                self.gamedata.attack_cost(
                    attack,
                )
            )
            if cost == 0:
                continue
            useful = min(
                attached_count,
                cost,
            )
            surplus = max(
                0,
                attached_count - cost,
            )
            score = (
                useful
                * self.weights.energy_weight
            )
            score += (
                surplus
                * self.weights.energy_weight
                * 0.25
            )
            if score > best_score:
                best_score = score
        return best_score

    # ---------------------------------------------------------
    # Retreat penalty
    # ---------------------------------------------------------
    def retreat_cost(
        self,
        retreat_cost,
        attached,
    ):
        if retreat_cost <= 0:
            return 0.0
        lost = min(
            retreat_cost,
            attached,
        )
        penalty = (
            retreat_cost
            * self.weights.retreat_weight
        )
        penalty += (
            lost
            * self.weights.lost_energy_weight
        )
        if attached >= retreat_cost:
            penalty *= 1.25
        if attached >= retreat_cost + 2:
            penalty *= 1.35
        return penalty
        
    # ---------------------------------------------------------
    # Tempo
    # ---------------------------------------------------------
    def tempo_value(
        self,
        mine,
        opp,
    ):
        score = 0.0
        my_active = self.active(
            mine,
        )
        opp_active = self.active(
            opp,
        )
        if self.can_attack(
            my_active,
        ):
            score += self.weights.tempo_bonus
        if self.can_attack(
            opp_active,
        ):
            score -= self.weights.tempo_bonus
        if (
            my_active is not None
            and opp_active is not None
        ):
            my_damage = self._best_attack_damage(
                my_active,
                opp_active,
            )
            opp_damage = self._best_attack_damage(
                opp_active,
                my_active,
            )
            my_hp = (
                my_active.get("hp", 0)
                - my_active.get("damage", 0)
            )
            opp_hp = (
                opp_active.get("hp", 0)
                - opp_active.get("damage", 0)
            )
            if my_damage >= opp_hp:
                score += self._pokemon_prize_value(
                    opp_active,
                )
            if opp_damage >= my_hp:
                score -= self._pokemon_prize_value(
                    my_active,
                )
        return score

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def active(
        self,
        player,
    ):
        arr = player.get("active") or []
        if arr:
            return arr[0]
        return None
    
    def can_attack(
        self,
        pokemon,
    ):
        return self.best_attack_damage(
            pokemon,
        ) > 0
        
    def _best_attack_damage(
        self,
        attacker,
        defender,
    ):
        if (
            attacker is None
            or defender is None
        ):
            return 0
        attacker_id = attacker.get("id")
        defender_id = defender.get("id")
        if (
            attacker_id is None
            or defender_id is None
        ):
            return 0
        attached = attacker.get(
            "energies",
            [],
        )
        attacks = self.gamedata.card_attacks.get(
            attacker_id,
            [],
        )
        defender_hp = max(
            0,
            defender.get("hp", 0)
            - defender.get("damage", 0),
        )
        best_score = 0.0
        for attack in attacks:
            if not self.gamedata.can_pay(
                self.gamedata.attack_cost(
                    attack,
                ),
                attached,
            ):
                continue
            damage = self.gamedata.attack_damage(
                attack,
            )
            damage = self.gamedata.effective_damage(
                attacker_id,
                damage,
                defender_id,
            )
            damage += self.gamedata.attack_effect_bonus(
                attack,
            )
            score = float(damage)
            if damage >= defender_hp:
                score += (
                    self.gamedata.prize_value(
                        defender_id,
                    )
                    * self.weights.prize_weight
                )
            if score > best_score:
                best_score = score
        return best_score
    
    def _pokemon_prize_value(
        self,
        pokemon,
    ):
        if pokemon is None:
            return 0.0
        rule = pokemon.get(
            "ruleBox",
            "",
        )
        if isinstance(rule, str):
            text = rule.lower()
            if "vmax" in text:
                return 3.0 * self.weights.prize_weight
            if (
                "vstar" in text
                or "ex" in text
                or "gx" in text
            ):
                return 2.0 * self.weights.prize_weight
        return 1.0 * self.weights.prize_weight
    
    def _best_attack_damage(
        self,
        attacker,
        defender,
    ):
        if (
            attacker is None
            or defender is None
        ):
            return 0
        cid = attacker.get("id")
        if cid is None:
            return 0
        attached = attacker.get(
            "energies",
            [],
        )
        defender_id = defender.get(
            "id",
        )
        attacks = self.gamedata.card_attacks.get(
            cid,
            [],
        )
        best = 0
        for attack in attacks:
            if not self.gamedata.can_pay(
                self.gamedata.attack_cost(
                    attack,
                ),
                attached,
            ):
                continue
            damage = self.gamedata.attack_damage(
                attack,
            )
            if defender_id is not None:
                damage = self.gamedata.effective_damage(
                    cid,
                    damage,
                    defender_id,
                )
            if damage > best:
                best = damage
        return best