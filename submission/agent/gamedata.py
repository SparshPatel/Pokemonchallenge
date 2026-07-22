"""Authoritative runtime card/attack data, sourced from the bundled engine.
At battle time the ``cg`` package is on the path, so ``cg.api`` can give us the
ground-truth card types and per-attack damage/cost. This is far richer than the
CSV-derived ``cards.json`` (which has no attackId -> damage mapping), and it is
what makes damage-aware play and lethal detection possible.
Everything is crash-safe: if the engine import fails for any reason we fall back
to the JSON stats (types only, no attack table), and every accessor returns a
safe default. The data is loaded once and cached.
"""
from __future__ import annotations
from .enums import CardType, EnergyType
# Keywords that mark a Trainer as a "dig" card — one that searches the deck or
# otherwise digs for board pieces. Used for trainer sequencing (play these to
# find Pokémon/Energy before committing the once-per-turn Supporter).
_DIG_KEYWORDS = ("ball", "poffin", "nest", "communication", "gear", "search")
# Damage modifiers from the official rulebook (Active Pokémon only):
#   * Weakness multiplies the attacker's damage (the SV/Mega era uses ×2).
#   * Resistance subtracts a flat amount (−30 in the current standard).
# Weakness/Resistance are matched against the *attacking* Pokémon's type
# (``energyType``), never the attack's Energy cost.
WEAKNESS_MULT = 2.0
RESISTANCE_FLAT = 30
# Residual damage (in HP) a Special Condition inflicts per Pokémon Checkup,
# per the rulebook: Burned places 2 damage counters (20 HP), Poisoned places 1
# (10 HP). These are *between-turns* effects, so they add to an attack's
# *expected* value but never to immediate lethal detection.
_BURN_RESIDUAL = 20
_POISON_RESIDUAL = 10

class _Attack:
    __slots__ = (
        "attack_id",
        "name",
        "text",
        "damage",
        "energies",
        "effect_bonus",
        "draw",
        "heal",
        "self_damage",
        "bench_damage",
        "gust",
        "switch",
        "energy_acceleration",
        "energy_discard_self",
        "energy_discard_opponent",
        "spread",
        "status_burn",
        "status_poison",
        "status_paralyze",
        "status_confuse",
        "status_sleep",
    )
    def __init__(
        self,
        attack_id: int,
        name: str,
        damage: int,
        energies: list[int],
        effect_bonus: int = 0,
        text: str = "",
    ):
        self.attack_id = attack_id
        self.name = name
        self.damage = damage
        self.energies = energies
        self.effect_bonus = effect_bonus
        self.text = text.lower()
        self.draw = 0
        self.heal = 0
        self.self_damage = 0
        self.bench_damage = 0
        self.gust = False
        self.switch = False
        self.energy_acceleration = False
        self.energy_discard_self = False
        self.energy_discard_opponent = False
        self.spread = False
        self.status_burn = False
        self.status_poison = False
        self.status_paralyze = False
        self.status_confuse = False
        self.status_sleep = False

class GameData:
    """Cached card-type and attack tables for runtime decision-making."""
    _instance: "GameData | None" = None
    def __init__(self) -> None:
        self.card_type: dict[int, int] = {}
        self.card_name: dict[int, str] = {}
        self.is_mega_id: set[int] = set()         # Mega ex: 3 Prizes when KO'd
        self.is_basic_pokemon_id: set[int] = set()
        self.is_ex_id: set[int] = set()
        self.card_best_damage: dict[int, int] = {}
        self.card_attacks: dict[int, list[int]] = {}
        self.is_dig_item_id: set[int] = set()
        self.card_weakness: dict[int, int] = {}      # cardId -> EnergyType weak to
        self.card_resistance: dict[int, int] = {}    # cardId -> EnergyType resisted
        self.card_energy_type: dict[int, int] = {}   # cardId -> Pokémon's own type
        self.attacks: dict[int, _Attack] = {}
        self.ok = False

    @classmethod
    def load(cls) -> "GameData":
        if cls._instance is not None:
            return cls._instance
        gd = cls()
        gd._load_from_engine()
        cls._instance = gd
        return gd

    def _load_from_engine(self) -> None:
        """
        Load metadata from cg.api when available.
        If cg.api cannot be imported (offline testing, unit tests, future engine
        changes), fall back to cards.json so the rest of the agent still has
        Pokémon/Trainer metadata instead of an empty database.
        """
        try:
            from cg import api  # type: ignore
            self._load_engine(api)
            self.ok = True
            return
        except Exception:
            pass
        self._load_cards_json()
        
    def _load_engine(self, api) -> None:
        """
        Load runtime card and attack metadata from cg.api.
        Besides raw damage/costs we also extract lightweight tactical metadata
        from attack text so later evaluation does not need to repeatedly parse
        natural language.
        """
        import re
        draw_re = re.compile(r"draw\s+(\d+)\s+card")
        heal_re = re.compile(r"heal\s+(\d+)")
        bench_re = re.compile(r"(\d+)\s+damage.*bench")
        for a in api.all_attack():
            text = (getattr(a, "text", "") or "")
            lower = text.lower()
            bonus = 0
            if "burned" in lower:
                bonus += _BURN_RESIDUAL
            if "poisoned" in lower:
                bonus += _POISON_RESIDUAL
            atk = _Attack(
                attack_id=a.attackId,
                name=a.name,
                damage=int(a.damage or 0),
                energies=[int(e) for e in (a.energies or [])],
                effect_bonus=bonus,
                text=text,
            )
            # -------------------------
            # Status conditions
            # -------------------------
            atk.status_burn = "burned" in lower
            atk.status_poison = "poisoned" in lower
            atk.status_paralyze = "paralyzed" in lower
            atk.status_confuse = "confused" in lower
            atk.status_sleep = "asleep" in lower
            # -------------------------
            # Draw
            # -------------------------
            m = draw_re.search(lower)
            if m:
                atk.draw = int(m.group(1))
            # -------------------------
            # Healing
            # -------------------------
            m = heal_re.search(lower)
            if m:
                atk.heal = int(m.group(1))
            # -------------------------
            # Bench damage
            # -------------------------
            m = bench_re.search(lower)
            if m:
                atk.bench_damage = int(m.group(1))
                atk.spread = True
            # -------------------------
            # Gust
            # -------------------------
            if (
                "switch your opponent's active" in lower
                or "choose 1 of your opponent's benched" in lower
            ):
                atk.gust = True
            # -------------------------
            # Self switching
            # -------------------------
            if (
                "switch this pokémon" in lower
                or "switch this pokemon" in lower
                or "switch your active" in lower
            ):
                atk.switch = True
            # -------------------------
            # Energy acceleration
            # -------------------------
            if (
                "attach" in lower
                and "energy" in lower
            ):
                atk.energy_acceleration = True
            # -------------------------
            # Energy discard
            # -------------------------
            if (
                "discard an energy from this pokémon" in lower
                or "discard an energy from this pokemon" in lower
                or "discard all energy from this pokémon" in lower
                or "discard all energy from this pokemon" in lower
            ):
                atk.energy_discard_self = True
            if (
                "discard an energy from your opponent" in lower
                or "discard an energy attached to your opponent" in lower
            ):
                atk.energy_discard_opponent = True
            self.attacks[a.attackId] = atk
        for c in api.all_card_data():
            cid = c.cardId
            self.card_type[cid] = int(c.cardType)
            self.card_name[cid] = c.name
            self.card_attacks[cid] = [
                int(a)
                for a in (c.attacks or [])
            ]
            if getattr(c, "weakness", None) is not None:
                self.card_weakness[cid] = int(c.weakness)
            if getattr(c, "resistance", None) is not None:
                self.card_resistance[cid] = int(c.resistance)
            if getattr(c, "energyType", None) is not None:
                self.card_energy_type[cid] = int(c.energyType)
            if getattr(c, "basic", False) and int(c.cardType) == CardType.POKEMON:
                self.is_basic_pokemon_id.add(cid)
            if getattr(c, "ex", False) or getattr(c, "megaEx", False):
                self.is_ex_id.add(cid)
            if getattr(c, "megaEx", False):
                self.is_mega_id.add(cid)
            if int(c.cardType) == CardType.ITEM:
                name = (c.name or "").lower()
                if any(k in name for k in _DIG_KEYWORDS):
                    self.is_dig_item_id.add(cid)
            dmg = 0
            for aid in (c.attacks or []):
                atk = self.attacks.get(aid)
                if atk:
                    dmg = max(
                        dmg,
                        atk.damage,
                    )
            self.card_best_damage[cid] = dmg

    def _load_cards_json(self) -> None:
        """
        Lightweight offline fallback.
        cards.json lacks attack IDs and exact costs, but still provides enough
        metadata for evaluation, feature extraction and unit tests.
        """
        import json
        import os
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "cards.json",
        )
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cards = json.load(f)
        except Exception:
            return
        if not isinstance(cards, list):
            return
        for card in cards:
            cid = card.get("id")
            if not isinstance(cid, int):
                continue
            self.card_name[cid] = card.get("name", "")
            ctype = card.get("cardType")
            if isinstance(ctype, int):
                self.card_type[cid] = ctype
            if card.get("basic"):
                self.is_basic_pokemon_id.add(cid)
            if card.get("ex"):
                self.is_ex_id.add(cid)
            if card.get("megaEx"):
                self.is_mega_id.add(cid)
            dmg = int(card.get("bestDamage", 0) or 0)
            self.card_best_damage[cid] = dmg
        self.ok = False

    # --- accessors --------------------------------------------------------
    def type_of(self, card_id: int | None) -> int | None:
        if card_id is None:
            return None
        return self.card_type.get(card_id)

    def is_pokemon(self, card_id: int | None) -> bool:
        return self.type_of(card_id) == CardType.POKEMON

    def is_basic_pokemon(self, card_id: int | None) -> bool:
        return card_id in self.is_basic_pokemon_id

    def is_supporter(self, card_id: int | None) -> bool:
        return self.type_of(card_id) == CardType.SUPPORTER

    def is_item(self, card_id: int | None) -> bool:
        return self.type_of(card_id) == CardType.ITEM

    def is_energy(self, card_id: int | None) -> bool:
        return self.type_of(card_id) in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)

    def is_ex(self, card_id: int | None) -> bool:
        return card_id in self.is_ex_id

    def is_mega(self, card_id: int | None) -> bool:
        """True for Mega Evolution Pokémon ex (worth 3 Prizes when KO'd)."""
        return card_id in self.is_mega_id

    def prize_value(self, card_id: int | None) -> int:
        """Prizes the opponent takes when this Pokémon is KO'd: Mega 3, ex 2, else 1."""
        if card_id in self.is_mega_id:
            return 3
        if card_id in self.is_ex_id:
            return 2
        return 1

    def attack_effect_bonus(self, attack_id: int | None) -> int:
        """
        Expected value from attack effects.
        Residual damage is already included here.
        """
        atk = self.attack(attack_id)
        if atk is None:
            return 0
        bonus = atk.effect_bonus
        if atk.gust:
            bonus += 20
        if atk.energy_acceleration:
            bonus += 25
        if atk.energy_discard_opponent:
            bonus += 18
        if atk.heal:
            bonus += min(20, atk.heal // 2)
        return bonus

    def attack_damage(self, attack_id: int | None) -> int:
        atk = self.attacks.get(attack_id) if attack_id is not None else None
        return atk.damage if atk else 0
    
    def attack(self, attack_id: int | None) -> _Attack | None:
        if attack_id is None:
            return None
        return self.attacks.get(attack_id)

    def attack_draw(self, attack_id: int | None) -> int:
        atk = self.attack(attack_id)
        return atk.draw if atk else 0

    def attack_heal(self, attack_id: int | None) -> int:
        atk = self.attack(attack_id)
        return atk.heal if atk else 0

    def attack_bench_damage(self, attack_id: int | None) -> int:
        atk = self.attack(attack_id)
        return atk.bench_damage if atk else 0

    def attack_spread(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.spread if atk else False

    def attack_gust(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.gust if atk else False

    def attack_switch(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.switch if atk else False

    def attack_energy_acceleration(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.energy_acceleration if atk else False

    def attack_discards_self_energy(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.energy_discard_self if atk else False

    def attack_discards_opponent_energy(self, attack_id: int | None) -> bool:
        atk = self.attack(attack_id)
        return atk.energy_discard_opponent if atk else False

    def attack_status_score(self, attack_id: int | None) -> int:
        atk = self.attack(attack_id)
        if atk is None:
            return 0
        score = 0
        if atk.status_burn:
            score += 15
        if atk.status_poison:
            score += 18
        if atk.status_paralyze:
            score += 40
        if atk.status_sleep:
            score += 15
        if atk.status_confuse:
            score += 20
        return score
    
    def attack_utility_score(
        self,
        attack_id: int | None,
    ) -> int:
        """
        Overall tactical value of an attack independent of raw damage.
        This is intentionally conservative. Damage is evaluated elsewhere;
        this only rewards secondary effects that often win games.
        """
        atk = self.attack(attack_id)
        if atk is None:
            return 0
        score = 0
        score += atk.draw * 5
        score += atk.heal // 10
        score += atk.bench_damage // 10
        if atk.gust:
            score += 35
        if atk.switch:
            score += 15
        if atk.energy_acceleration:
            score += 30
        if atk.energy_discard_opponent:
            score += 25
        if atk.energy_discard_self:
            score -= 15
        score += self.attack_status_score(attack_id)
        return score

    def best_damage(self, card_id: int | None) -> int:
        """
        Best realistic damage this Pokémon can produce.
        Includes attack-effect bonus so evaluation prefers attacks that
        deal less immediate damage but produce stronger board impact.
        """
        if card_id is None:
            return 0
        best = 0
        for aid in self.card_attacks.get(card_id, ()):
            atk = self.attack(aid)
            if atk is None:
                continue
            dmg = atk.damage + self.attack_effect_bonus(aid)
            if dmg > best:
                best = dmg
        return best

    # --- weakness / resistance reasoning ---------------------------------
    def energy_type(self, card_id: int | None) -> int | None:
        if card_id is None:
            return None
        return self.card_energy_type.get(card_id)

    def effective_damage(
        self,
        attacker_id: int | None,
        attack_id: int | None,
        base_dmg: int,
        defender_id: int | None,
    ) -> int:
        """
        True runtime damage estimate.
        Applies:
            • Weakness
            • Resistance
            • Mad Bite scaling
            • attack effect bonus
        Never crashes.
        """
        if base_dmg <= 0:
            return base_dmg
        dmg = base_dmg
        atk = self.attack(attack_id)
        if atk is not None:
            # Bloodmoon Ursaluna — Mad Bite
            if atk.attack_id == 175:
                dmg = max(dmg, 100)
        atk_type = (
            self.card_energy_type.get(attacker_id)
            if attacker_id is not None
            else None
        )
        if atk_type is not None and defender_id is not None:
            if self.card_weakness.get(defender_id) == atk_type:
                dmg *= WEAKNESS_MULT
            if self.card_resistance.get(defender_id) == atk_type:
                dmg -= RESISTANCE_FLAT
        dmg += self.attack_effect_bonus(attack_id)
        return max(0, int(dmg))
    
    def is_dig_item(self, card_id: int | None) -> bool:
        return card_id in self.is_dig_item_id

    # --- energy-need reasoning -------------------------------------------
    def attack_cost(self, attack_id: int | None) -> list[int]:
        atk = self.attacks.get(attack_id) if attack_id is not None else None
        return list(atk.energies) if atk else []

    @staticmethod
    def can_pay(cost: list[int], attached: list[int]) -> bool:
        """Can the ``attached`` energies pay ``cost``? Colorless = any energy."""
        if len(attached) < len(cost):
            return False
        pool = list(attached)
        # Satisfy specific (typed) requirements first.
        for need in cost:
            if need == EnergyType.COLORLESS:
                continue
            if need in pool:
                pool.remove(need)
            else:
                return False
        colorless = sum(1 for c in cost if c == EnergyType.COLORLESS)
        return len(pool) >= colorless
    
    def needs_energy(self, card_id: int | None, attached: list[int]) -> bool:
        """True if ``card_id`` cannot yet pay its most expensive attack.
        A Pokémon that can already power its hardest attack does not benefit from
        more Energy — attaching to it instead of a needier target (or instead of
        attacking) is wasteful.
        """
        if card_id is None:
            return True
        costs = [self.attack_cost(a) for a in self.card_attacks.get(card_id, [])]
        costs = [c for c in costs if c]
        if not costs:
            return False  # no energy-costed attack → no need to feed it
        hardest = max(costs, key=len)
        '''This assumes
            longest cost == hardest attack
            which is true 99% of the time.
            You could theoretically use'''
        '''max(costs, key=lambda c: (
            len(c),
            sum(e != EnergyType.COLORLESS for e in c)
        ))'''
        return not self.can_pay(hardest, list(attached or []))