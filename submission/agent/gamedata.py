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
    __slots__ = ("attack_id", "name", "damage", "energies", "effect_bonus")

    def __init__(self, attack_id: int, name: str, damage: int, energies: list[int],
                 effect_bonus: int = 0):
        self.attack_id = attack_id
        self.name = name
        self.damage = damage
        self.energies = energies
        self.effect_bonus = effect_bonus


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
        try:
            from cg import api  # type: ignore
        except Exception:
            return
        try:
            for a in api.all_attack():
                text = (getattr(a, "text", "") or "").lower()
                bonus = 0
                if "burned" in text:
                    bonus += _BURN_RESIDUAL
                if "poisoned" in text:
                    bonus += _POISON_RESIDUAL
                self.attacks[a.attackId] = _Attack(
                    a.attackId, a.name, int(a.damage or 0),
                    [int(e) for e in (a.energies or [])], bonus,
                )
            for c in api.all_card_data():
                cid = c.cardId
                self.card_type[cid] = int(c.cardType)
                self.card_name[cid] = c.name
                self.card_attacks[cid] = [int(a) for a in (c.attacks or [])]
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
                        dmg = max(dmg, atk.damage)
                self.card_best_damage[cid] = dmg
            self.ok = True
        except Exception:
            # Partial load is fine; accessors degrade gracefully.
            pass

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
        """Expected residual damage (HP) from an attack's Special Condition.

        Burned ~20/checkup, Poisoned ~10/checkup. A one-turn estimate used to
        value condition-inflicting attacks; never counted toward immediate lethal
        (these resolve between turns, after the opponent may retreat/heal).
        """
        atk = self.attacks.get(attack_id) if attack_id is not None else None
        return atk.effect_bonus if atk else 0

    def attack_damage(self, attack_id: int | None) -> int:
        atk = self.attacks.get(attack_id) if attack_id is not None else None
        return atk.damage if atk else 0

    def best_damage(self, card_id: int | None) -> int:
        if card_id is None:
            return 0
        return self.card_best_damage.get(card_id, 0)

    # --- weakness / resistance reasoning ---------------------------------
    def energy_type(self, card_id: int | None) -> int | None:
        if card_id is None:
            return None
        return self.card_energy_type.get(card_id)

    def effective_damage(
        self, attacker_id: int | None, base_dmg: int, defender_id: int | None
    ) -> int:
        """Damage ``attacker_id`` deals to the Active ``defender_id``.

        Applies Weakness (×2) and Resistance (flat −) per the rulebook, matched
        against the *attacker's* Pokémon type. Degrades to ``base_dmg`` when type
        data is missing. Only meaningful for the Active Spot (callers ensure the
        defender is/will be Active).
        """
        if base_dmg <= 0:
            return base_dmg
        atk_type = self.card_energy_type.get(attacker_id) if attacker_id is not None else None
        if atk_type is None or defender_id is None:
            return base_dmg
        dmg = float(base_dmg)
        if self.card_weakness.get(defender_id) == atk_type:
            dmg *= WEAKNESS_MULT
        if self.card_resistance.get(defender_id) == atk_type:
            dmg -= RESISTANCE_FLAT
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
        return not self.can_pay(hardest, list(attached or []))
