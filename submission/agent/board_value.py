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

        my_prizes = len(
            mine.get("prize") or []
        )

        opp_prizes = len(
            opp.get("prize") or []
        )

        if opp_prizes == 0:
            return TERMINAL_WIN

        if my_prizes == 0:
            return -TERMINAL_WIN

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
        for mon in player.get("bench") or []:
            pv = self.pokemon_value(
                mon,
                opponent=opp_active,
                active=False,
            )

            if self.can_attack(mon):
                pv *= 1.20
            else:
                pv *= 0.85

            value += pv
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

        hp = pokemon.get("hp", 0)
        damage_taken = pokemon.get("damage", 0)
        remaining_hp = max(0, hp - damage_taken)

        score = 0.0

        # --------------------------------------------------
        # HP value
        # --------------------------------------------------
        score += remaining_hp * self.weights.hp_weight

        hp_ratio = remaining_hp / max(hp, 1)

        if active:
            if hp_ratio < 0.20:
                score -= 220
            elif hp_ratio < 0.40:
                score -= 120
            elif hp_ratio < 0.60:
                score -= 45
        else:
            if hp_ratio < 0.25:
                score -= 130
            elif hp_ratio < 0.50:
                score -= 60

        attached = pokemon.get("energies") or []
        attached_count = len(attached)

        retreat = pokemon.get("retreatCost", 0)

        cid = pokemon.get("id")

        attacks = self.gamedata.card_attacks.get(cid, [])

        # --------------------------------------------------
        # Energy investment
        # --------------------------------------------------
        score += self.energy_value(pokemon)

        # --------------------------------------------------
        # Retreat tax
        # --------------------------------------------------
        score -= self.retreat_cost(
            retreat,
            attached_count,
        )

        # --------------------------------------------------
        # Active bonus
        # --------------------------------------------------
        if active:
            score += self.weights.active_bonus

        # --------------------------------------------------
        # Attack quality
        # --------------------------------------------------
        best_attack_score = 0.0

        for attack in attacks:

            cost = self.gamedata.attack_cost(attack)

            if not self.gamedata.can_pay(
                cost,
                attached,
            ):
                continue

            damage = self.gamedata.attack_damage(attack)

            if opponent is not None:
                damage = self.gamedata.effective_damage(
                    cid,
                    damage,
                    opponent.get("id"),
                )

            attack_score = float(damage)

            attack_score += (
                self.gamedata.attack_effect_bonus(attack)
                * 1.20
            )

            atk = self.gamedata.attacks.get(attack)

            if atk is not None:

                text = (getattr(atk, "name", "") or "").lower()

                if any(x in text for x in ("bench", "each", "all")):
                    attack_score += 30

                if any(x in text for x in ("heal", "recover")):
                    attack_score += 45

                if any(x in text for x in ("switch", "swap")):
                    attack_score += 25

                if any(x in text for x in ("choose", "switch your opponent")):
                    attack_score += 45

                if any(x in text for x in ("attach", "search your deck")):
                    attack_score += 35

                if "discard" in text and "energy" in text:
                    attack_score -= 25

            if opponent is not None:

                opp_hp = max(
                    0,
                    opponent.get("hp", 0)
                    - opponent.get("damage", 0),
                )

                if damage >= opp_hp:

                    attack_score += (
                        self.gamedata.prize_value(
                            opponent.get("id")
                        )
                        * self.weights.prize_weight
                    )

                    attack_score += 250

            best_attack_score = max(
                best_attack_score,
                attack_score,
            )

        # --------------------------------------------------
        # Powered attacker bonus
        # --------------------------------------------------
        if best_attack_score > 0:

            score += self.weights.attack_ready_bonus
            score += best_attack_score * 0.50

            if hp_ratio > 0.70:
                score += best_attack_score * 0.25

            if attacks:

                hardest = max(
                    (
                        len(self.gamedata.attack_cost(a))
                        for a in attacks
                    ),
                    default=0,
                )

                if hardest > 0:

                    ratio = min(
                        attached_count / hardest,
                        1.5,
                    )

                    score += ratio * 180

        elif attacks:

            cheapest = min(
                len(self.gamedata.attack_cost(a))
                for a in attacks
            )

            if attached_count == cheapest - 1:
                score += 120

            elif attached_count == cheapest - 2:
                score += 45

        # --------------------------------------------------
        # Incoming danger
        # --------------------------------------------------
        if opponent is not None:

            incoming = self._best_attack_damage(
                opponent,
                pokemon,
            )

            if incoming >= remaining_hp:

                if active:
                    score *= 0.15
                else:
                    score *= 0.55

            elif incoming >= remaining_hp * 0.80:

                if active:
                    score *= 0.40
                else:
                    score *= 0.75

            elif incoming >= remaining_hp * 0.50:

                if active:
                    score *= 0.70
                else:
                    score *= 0.90

        # --------------------------------------------------
        # Rule-box preservation
        # --------------------------------------------------
        if not active:

            prize_value = self._pokemon_prize_value(pokemon)

            if prize_value >= 2 * self.weights.prize_weight:
                score *= 1.15

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
        if attached_count >= 4:
            best_score += 35

        if attached_count >= 6:
            best_score += 70
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

        my_active = self.active(mine)
        opp_active = self.active(opp)

        my_ready = self.can_attack(my_active)
        opp_ready = self.can_attack(opp_active)

        if my_ready:
            score += self.weights.tempo_bonus

        if opp_ready:
            score -= self.weights.tempo_bonus

        if my_ready and not opp_ready:
            score += 140

        if opp_ready and not my_ready:
            score -= 140

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

            my_hp = max(
                0,
                my_active.get("hp", 0)
                - my_active.get("damage", 0),
            )

            opp_hp = max(
                0,
                opp_active.get("hp", 0)
                - opp_active.get("damage", 0),
            )

            # ---------- immediate KO race ----------
            if my_damage >= opp_hp:
                score += (
                    self._pokemon_prize_value(opp_active)
                    + 250
                )

            if opp_damage >= my_hp:
                score -= (
                    self._pokemon_prize_value(my_active)
                    + 250
                )

            # ---------- 2HKO race ----------
            elif my_damage > opp_damage:
                score += 70

            elif opp_damage > my_damage:
                score -= 70

            # ---------- HP pressure ----------
            if my_hp < 80:
                score -= 40

            if opp_hp < 80:
                score += 40

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
        if pokemon is None:
            return False

        cid = pokemon.get("id")
        if cid is None:
            return False

        attached = pokemon.get(
            "energies",
            [],
        )

        attacks = self.gamedata.card_attacks.get(
            cid,
            [],
        )

        for attack in attacks:
            if self.gamedata.can_pay(
                self.gamedata.attack_cost(
                    attack,
                ),
                attached,
            ):
                return True

        return False
        
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