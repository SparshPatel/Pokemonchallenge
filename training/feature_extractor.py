from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
# =====================================================================
# Feature Definition
# =====================================================================
FEATURE_NAMES = (
    # ---------------------------------------------------------------
    # Prize state
    # ---------------------------------------------------------------
    "prize_diff",
    "my_prize_left",
    "opp_prize_left",
    # ---------------------------------------------------------------
    # Active Pokémon
    # ---------------------------------------------------------------
    "my_active_hpfrac",
    "opp_active_hpfrac",
    "opp_active_dmgfrac",
    "my_ready",
    "setup_ko",
    "opp_threat",
    "active_quality",
    "active_loaded",
    # ---------------------------------------------------------------
    # Bench
    # ---------------------------------------------------------------
    "bench_frac",
    "bench_ready_frac",
    "bench_setup_ko",
    "opp_bench_dmg",
    # ---------------------------------------------------------------
    # Resources
    # ---------------------------------------------------------------
    "energy_frac",
    "hand_frac",
    # ---------------------------------------------------------------
    # Tempo
    # ---------------------------------------------------------------
    "my_can_attack",
    "opp_can_attack",
    "energy_advantage",
    "board_control",
    "supporter_available",
    "gust_available",
    "switch_available",
    "stadium_in_play",
    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------
    "multi_prize_risk",
    "bench_liability",
    "no_active",
    # ---------------------------------------------------------------
    # Bias
    # ---------------------------------------------------------------
    "bias",
)
FEATURE_DIM = len(FEATURE_NAMES)
# =====================================================================
# Feature Sample
# =====================================================================
@dataclass(slots=True)
class FeatureSample:
    replay_name: str
    turn_index: int
    terminal: bool
    features: np.ndarray
    target_value: float
    metadata: dict[str, Any]
    
# =====================================================================
# Feature Extractor
# =====================================================================
class FeatureExtractor:
    """
    Training-time feature extractor.
    IMPORTANT:
    The output feature order MUST remain exactly identical to
    agent.value_net.FEATURE_NAMES.
    The runtime ValueNet consumes exactly FEATURE_DIM values.
    """
    def __init__(
        self,
        gamedata=None,
        helpers=None,
    ):
        self.gamedata = gamedata
        self.helpers = helpers
        self.feature_names = list(
            FEATURE_NAMES
        )

    # -----------------------------------------------------------------
    @property
    def feature_dim(
        self,
    ) -> int:
        return FEATURE_DIM

    # -----------------------------------------------------------------
    def _empty_vector(
        self,
    ) -> np.ndarray:
        return np.zeros(
            self.feature_dim,
            dtype=np.float32,
        )

    # =================================================================
    # Generic Helpers
    # =================================================================
    @staticmethod
    def _players(
        state,
    ):
        if not isinstance(state, dict):
            return []
        players = (
            state.get("players")
            or []
        )
        if not isinstance(
            players,
            list,
        ):
            return []
        return players

    # -----------------------------------------------------------------
    @staticmethod
    def _player(
        state,
        index,
    ):
        players = (
            FeatureExtractor._players(
                state
            )
        )
        if (
            index < 0
            or index >= len(players)
        ):
            return {}
        player = players[index]
        if not isinstance(
            player,
            dict,
        ):
            return {}
        return player

    # -----------------------------------------------------------------
    @staticmethod
    def _active(
        player,
    ):
        if not isinstance(
            player,
            dict,
        ):
            return None
        active = (
            player.get("active")
            or []
        )
        if isinstance(
            active,
            dict,
        ):
            return active
        for pokemon in active:
            if isinstance(
                pokemon,
                dict,
            ):
                return pokemon
        return None

    # -----------------------------------------------------------------
    @staticmethod
    def _bench(
        player,
    ):
        if not isinstance(
            player,
            dict,
        ):
            return []
        bench = (
            player.get("bench")
            or []
        )
        if not isinstance(
            bench,
            list,
        ):
            return []
        return [
            pokemon
            for pokemon in bench
            if isinstance(
                pokemon,
                dict,
            )
        ]

    # -----------------------------------------------------------------
    @staticmethod
    def _prizes_left(
        player,
    ):
        if not isinstance(
            player,
            dict,
        ):
            return 0
        prize = player.get(
            "prize"
        )
        if isinstance(
            prize,
            list,
        ):
            return len(prize)
        value = player.get(
            "prizesLeft"
        )
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
        return 0

    # -----------------------------------------------------------------
    @staticmethod
    def _hand_count(
        player,
    ):
        if not isinstance(
            player,
            dict,
        ):
            return 0
        value = player.get(
            "handCount"
        )
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
        hand = player.get(
            "hand"
        )
        if isinstance(
            hand,
            list,
        ):
            return len(hand)
        return 0

    # -----------------------------------------------------------------
    @staticmethod
    def _energy_count(
        player,
    ):
        if not isinstance(
            player,
            dict,
        ):
            return 0
        energy = (
            player.get(
                "energyCards"
            )
            or player.get(
                "energy"
            )
            or []
        )
        if isinstance(
            energy,
            list,
        ):
            return len(energy)
        try:
            return int(energy)
        except Exception:
            return 0

    # -----------------------------------------------------------------
    @staticmethod
    def _hp(
        pokemon,
    ):
        if not isinstance(
            pokemon,
            dict,
        ):
            return 0
        try:
            return float(
                pokemon.get(
                    "hp"
                )
                or 0
            )
        except Exception:
            return 0.0

    # -----------------------------------------------------------------
    @staticmethod
    def _max_hp(
        pokemon,
    ):
        if not isinstance(
            pokemon,
            dict,
        ):
            return 0
        try:
            return float(
                pokemon.get(
                    "maxHp"
                )
                or 0
            )
        except Exception:
            return 0.0

    # =================================================================
    # GameData Helpers
    # =================================================================
    def _best_damage(
        self,
        pokemon,
    ):
        if (
            self.gamedata is None
            or not isinstance(
                pokemon,
                dict,
            )
        ):
            return 0.0
        pokemon_id = pokemon.get(
            "id"
        )
        if pokemon_id is None:
            return 0.0
        try:
            return float(
                self.gamedata.best_damage(
                    pokemon_id
                )
            )
        except Exception:
            return 0.0

    # -----------------------------------------------------------------
    def _prize_value(
        self,
        pokemon,
    ):
        if (
            self.gamedata is None
            or not isinstance(
                pokemon,
                dict,
            )
        ):
            return 1.0
        pokemon_id = pokemon.get(
            "id"
        )
        try:
            return float(
                self.gamedata.prize_value(
                    pokemon_id
                )
            )
        except Exception:
            return 1.0

    # -----------------------------------------------------------------
    def _card_name(
        self,
        card_id,
    ):
        if self.gamedata is None:
            return ""
        names = getattr(
            self.gamedata,
            "card_name",
            {},
        )
        try:
            return str(
                names.get(
                    card_id,
                    "",
                )
            ).lower()
        except Exception:
            return ""

    # -----------------------------------------------------------------
    def _is_supporter(
        self,
        card_id,
    ):
        if self.gamedata is None:
            return False
        try:
            return bool(
                self.gamedata.is_supporter(
                    card_id
                )
            )
        except Exception:
            return False

    # =================================================================
    # Attack / Damage Helpers
    # =================================================================
    def _can_attack(
        self,
        pokemon,
    ):
        if (
            pokemon is None
            or self.helpers is None
            or self.gamedata is None
        ):
            return False
        try:
            return bool(
                self.helpers._can_attack(
                    pokemon,
                    self.gamedata,
                )
            )
        except Exception:
            return False

    # -----------------------------------------------------------------
    def _best_affordable_damage(
        self,
        attacker,
        defender,
    ):
        if (
            attacker is None
            or defender is None
            or self.helpers is None
            or self.gamedata is None
        ):
            return 0.0
        try:
            return float(
                self.helpers._best_affordable_dmg(
                    attacker,
                    defender,
                    self.gamedata,
                )
            )
        except Exception:
            return 0.0

    # =================================================================
    # Feature Extraction
    # =================================================================
    def _extract_from_state(
        self,
        state,
        me=0,
    ):
        """
        Extract the exact 29-dimensional feature vector
        expected by the runtime ValueNet.
        `state` should represent the board from a single
        decision point.
        """
        f = {
            name: 0.0
            for name in FEATURE_NAMES
        }
        players = self._players(
            state
        )
        if len(players) < 2:
            f["bias"] = 1.0
            return np.asarray(
                [
                    f[name]
                    for name in FEATURE_NAMES
                ],
                dtype=np.float32,
            )
        try:
            me = int(me)
        except Exception:
            me = 0
        if me not in (0, 1):
            me = 0
        opp = 1 - me
        mp = self._player(
            state,
            me,
        )
        op = self._player(
            state,
            opp,
        )
        my_left = self._prizes_left(
            mp
        )
        opp_left = self._prizes_left(
            op
        )
        my_act = self._active(
            mp
        )
        opp_act = self._active(
            op
        )
        my_bench = self._bench(
            mp
        )
        opp_bench = self._bench(
            op
        )
        my_energy = self._energy_count(
            mp
        )
        opp_energy = self._energy_count(
            op
        )
        # -------------------------------------------------------------
        # Prize state
        # -------------------------------------------------------------
        f["prize_diff"] = (
            opp_left
            - my_left
        ) / 6.0
        f["my_prize_left"] = (
            my_left
            / 6.0
        )
        f["opp_prize_left"] = (
            opp_left
            / 6.0
        )
        # -------------------------------------------------------------
        # Active Pokémon
        # -------------------------------------------------------------
        if my_act:
            mhp = self._hp(
                my_act
            )
            mmax = self._max_hp(
                my_act
            )
            if mmax > 0:
                f[
                    "my_active_hpfrac"
                ] = (
                    mhp
                    / mmax
                )
            if self._can_attack(
                my_act
            ):
                f[
                    "my_ready"
                ] = 1.0
        else:
            f[
                "no_active"
            ] = 1.0
        if opp_act:
            ohp = self._hp(
                opp_act
            )
            omax = self._max_hp(
                opp_act
            )
            if omax > 0:
                f[
                    "opp_active_hpfrac"
                ] = (
                    ohp
                    / omax
                )
                f[
                    "opp_active_dmgfrac"
                ] = (
                    omax
                    - ohp
                ) / omax
        # -------------------------------------------------------------
        # KO / Threat
        # -------------------------------------------------------------
        if (
            my_act
            and opp_act
        ):
            my_dmg = (
                self._best_affordable_damage(
                    my_act,
                    opp_act,
                )
            )
            if my_dmg >= self._hp(
                opp_act
            ):
                f[
                    "setup_ko"
                ] = 1.0
            opp_dmg = (
                self._best_affordable_damage(
                    opp_act,
                    my_act,
                )
            )
            if opp_dmg >= self._hp(
                my_act
            ):
                f[
                    "opp_threat"
                ] = 1.0
        # -------------------------------------------------------------
        # Active Quality
        # -------------------------------------------------------------
        if my_act:
            team_best = 0.0
            for pokemon in (
                [my_act]
                + my_bench
            ):
                team_best = max(
                    team_best,
                    self._best_damage(
                        pokemon
                    ),
                )
            active_best = (
                self._best_damage(
                    my_act
                )
            )
            if team_best > 0:
                f[
                    "active_quality"
                ] = (
                    active_best
                    / team_best
                )
            if (
                active_best > 0
                and opp_act
            ):
                affordable = (
                    self._best_affordable_damage(
                        my_act,
                        opp_act,
                    )
                )
                f[
                    "active_loaded"
                ] = min(
                    affordable
                    / active_best,
                    1.0,
                )
        # -------------------------------------------------------------
        # Bench
        # -------------------------------------------------------------
        f[
            "bench_frac"
        ] = min(
            len(my_bench),
            5,
        ) / 5.0
        ready = 0
        for pokemon in my_bench:
            if self._can_attack(
                pokemon
            ):
                ready += 1
        f[
            "bench_ready_frac"
        ] = min(
            ready,
            5,
        ) / 5.0
        # -------------------------------------------------------------
        # Energy
        # -------------------------------------------------------------
        f[
            "energy_frac"
        ] = min(
            my_energy,
            12,
        ) / 12.0
        # -------------------------------------------------------------
        # Tempo
        # -------------------------------------------------------------
        f[
            "my_can_attack"
        ] = float(
            bool(
                my_act
                and self._can_attack(
                    my_act
                )
            )
        )
        f[
            "opp_can_attack"
        ] = float(
            bool(
                opp_act
                and self._can_attack(
                    opp_act
                )
            )
        )
        f[
            "energy_advantage"
        ] = (
            my_energy
            - opp_energy
        ) / 12.0
        my_board = len(
            my_bench
        )
        opp_board = len(
            opp_bench
        )
        f[
            "board_control"
        ] = (
            my_board
            - opp_board
        ) / 5.0
        # -------------------------------------------------------------
        # Hand
        # -------------------------------------------------------------
        hand_count = (
            self._hand_count(
                mp
            )
        )
        f[
            "hand_frac"
        ] = min(
            hand_count,
            10,
        ) / 10.0
        # -------------------------------------------------------------
        # Trainer availability
        # -------------------------------------------------------------
        hand_cards = (
            mp.get(
                "hand"
            )
            or []
        )
        supporter = 0
        gust = 0
        switch = 0
        for card in hand_cards:
            if not isinstance(
                card,
                dict,
            ):
                continue
            card_id = card.get(
                "id"
            )
            if self._is_supporter(
                card_id
            ):
                supporter = 1
            name = self._card_name(
                card_id
            )
            if (
                "boss"
                in name
                or "catcher"
                in name
            ):
                gust = 1
            if (
                "switch"
                in name
            ):
                switch = 1
        f[
            "supporter_available"
        ] = supporter
        f[
            "gust_available"
        ] = gust
        f[
            "switch_available"
        ] = switch
        # -------------------------------------------------------------
        # Opponent bench pressure
        # -------------------------------------------------------------
        bench_damage = 0.0
        bench_kos = 0
        if my_act:
            for pokemon in opp_bench:
                hp = self._hp(
                    pokemon
                )
                max_hp = self._max_hp(
                    pokemon
                )
                if max_hp > 0:
                    bench_damage += (
                        (
                            max_hp
                            - hp
                        )
                        / max_hp
                    ) * (
                        self._prize_value(
                            pokemon
                        )
                    )
                if hp > 0:
                    damage = (
                        self._best_affordable_damage(
                            my_act,
                            pokemon,
                        )
                    )
                    if damage >= hp:
                        bench_kos += 1
        f[
            "opp_bench_dmg"
        ] = min(
            bench_damage,
            5.0,
        ) / 5.0
        f[
            "bench_setup_ko"
        ] = min(
            bench_kos,
            5,
        ) / 5.0
        # -------------------------------------------------------------
        # Stadium
        # -------------------------------------------------------------
        f[
            "stadium_in_play"
        ] = float(
            bool(
                state.get(
                    "stadium"
                )
            )
        )
        # -------------------------------------------------------------
        # Multi-prize risk
        # -------------------------------------------------------------
        risk = 0.0
        for pokemon in (
            [my_act]
            + my_bench
        ):
            if not isinstance(
                pokemon,
                dict,
            ):
                continue
            risk += (
                self._prize_value(
                    pokemon
                )
            )
        f[
            "multi_prize_risk"
        ] = (
            risk
            / 10.0
        )
        # -------------------------------------------------------------
        # Bench liability
        # -------------------------------------------------------------
        liability = 0
        for pokemon in my_bench:
            hp = self._hp(
                pokemon
            )
            if hp < 90:
                liability += 1
        f[
            "bench_liability"
        ] = (
            liability
            / 5.0
        )
        # -------------------------------------------------------------
        # Bias
        # -------------------------------------------------------------
        f[
            "bias"
        ] = 1.0
        return np.asarray(
            [
                f[name]
                for name in FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        
    # =================================================================
    # Replay Adapter
    # =================================================================
    def extract(
        self,
        sample,
    ) -> FeatureSample:
        """
        Convert one replay sample into the runtime-compatible
        29-dimensional feature representation.
        The replay sample's current state is used as the board state.
        The player perspective is taken from `yourIndex` when available.
        """
        state = (
            sample.current
        )
        if state is None:
            vector = self._empty_vector()
        else:
            your_index = state.get(
                "yourIndex",
                0,
            )
            try:
                your_index = int(
                    your_index
                )
            except Exception:
                your_index = 0
            vector = (
                self._extract_from_state(
                    state,
                    me=your_index,
                )
            )
        return FeatureSample(
            replay_name=(
                sample.replay_name
            ),
            turn_index=(
                sample.turn_index
            ),
            terminal=(
                sample.terminal
            ),
            features=vector,
            target_value=float(
                sample.reward
            ),
            metadata={},
        )

    # =================================================================
    # Dataset
    # =================================================================
    def extract_dataset(
        self,
        samples,
    ):
        return [
            self.extract(
                sample
            )
            for sample in samples
        ]

    # -----------------------------------------------------------------
    def feature_matrix(
        self,
        feature_samples: list[
            FeatureSample
        ],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        if not feature_samples:
            return (
                np.empty(
                    (
                        0,
                        self.feature_dim,
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (
                        0,
                    ),
                    dtype=np.float32,
                ),
            )
        X = np.vstack(
            [
                sample.features
                for sample in feature_samples
            ]
        ).astype(
            np.float32
        )
        y = np.asarray(
            [
                sample.target_value
                for sample in feature_samples
            ],
            dtype=np.float32,
        )
        return (
            X,
            y,
        )

    # =================================================================
    # Summary
    # =================================================================
    def summary(
        self,
        feature_samples: list[
            FeatureSample
        ],
    ):
        print()
        print(
            "=" * 70
        )
        print(
            "Feature Extractor Summary"
        )
        print(
            "=" * 70
        )
        print()
        print(
            f"Samples       : "
            f"{len(feature_samples)}"
        )
        print(
            f"Feature Count : "
            f"{self.feature_dim}"
        )
        terminals = sum(
            sample.terminal
            for sample in feature_samples
        )
        print(
            f"Terminal      : "
            f"{terminals}"
        )
        if feature_samples:
            first = (
                feature_samples[0]
            )
            print()
            print(
                "First Sample Shape:",
                first.features.shape,
            )
            print()
            print(
                "Feature Order:"
            )
            for index, (
                name,
                value,
            ) in enumerate(
                zip(
                    self.feature_names,
                    first.features,
                )
            ):
                print(
                    f"{index:02d} "
                    f"{name:24s} "
                    f"{value:.6f}"
                )