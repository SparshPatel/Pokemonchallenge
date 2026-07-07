"""
evaluation.py

Core evaluation framework for the Pokemon Challenge agent.

Everything that assigns a numerical value to a board state should eventually
flow through this file.

Pipeline

Board State
      ↓
FeatureExtractor
      ↓
FeatureVector
      ↓
WeightGenerator
      ↓
Evaluator
      ↓
Scalar Score

This file intentionally contains NO Pokemon-specific heuristics.
It only defines the interfaces.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import random
import math

from torch import embedding


# ============================================================
# Feature Vector
# ============================================================

@dataclass
class FeatureVector:
    """
    Container for board features.

    Keys are intentionally dynamic so we can add/remove features
    without changing the API.
    """
    values: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

    def set(self, key: str, value: float):
        self.values[key] = float(value)

    def update(self, data: Dict[str, float]):
        for k, v in data.items():
            self.values[k] = float(v)

    def copy(self):
        return FeatureVector(dict(self.values))


# ============================================================
# Feature Extractor
# ============================================================

class FeatureExtractor:
    """
    Base feature extractor.

    Concrete implementations should override ``extract()`` and
    return a populated FeatureVector.
    """

    def extract(self, state, me: int = 0) -> FeatureVector:
        raise NotImplementedError


class PlannerFeatureExtractor(FeatureExtractor):
    """
    Default feature extractor used by TurnPlanner.

    This class intentionally computes ONLY feature values.
    It contains no evaluation weights.
    """
    def __init__(self, gamedata):
        self.gamedata = gamedata
        
    def extract(self, state, me: int = 0) -> FeatureVector:
        features = FeatureVector()
        players = state.get("players") or []
        if len(players) < 2:
            return features
        mp = players[me] if isinstance(players[me], dict) else {}
        op = players[1 - me] if isinstance(players[1 - me], dict) else {}
        # Basic counts
        features.set("my_prizes", float(len(mp.get("prize") or [])))
        features.set("opp_prizes", float(len(op.get("prize") or [])))
        features.set("my_hand", float(mp.get("handCount", 0)))
        features.set("my_bench", float(len(mp.get("bench") or [])))
        features.set("opp_bench", float(len(op.get("bench") or [])))
        # Active Pokémon
        my_active = None
        opp_active = None
        active = mp.get("active") or []
        if active and isinstance(active[0], dict):
            my_active = active[0]
        active = op.get("active") or []
        if active and isinstance(active[0], dict):
            opp_active = active[0]
        # HP
        if my_active:
            hp = float(my_active.get("hp") or 0)
            max_hp = float(my_active.get("maxHp") or 0)
            features.set("my_hp", hp)
            features.set(
                "my_hp_ratio",
                hp / max_hp if max_hp else 0.0,
            )
            features.set(
                "my_energy",
                float(len(my_active.get("energies") or []))
            )
        else:
            features.set("my_hp", 0.0)
            features.set("my_hp_ratio", 0.0)
            features.set("my_energy", 0.0)
        if opp_active:
            hp = float(opp_active.get("hp") or 0)
            max_hp = float(opp_active.get("maxHp") or 0)
            features.set("opp_hp", hp)
            features.set(
                "opp_damage",
                (max_hp - hp) / max_hp if max_hp else 0.0,
            )
        else:
            features.set("opp_hp", 0.0)
            features.set("opp_damage", 0.0)
        # Total energy in play
        energy_total = 0
        for mon in (mp.get("bench") or []):
            if isinstance(mon, dict):
                energy_total += len(mon.get("energies") or [])
        if my_active:
            energy_total += len(my_active.get("energies") or [])
        features.set(
            "energy",
            float(energy_total),
        )
        # Bench HP
        bench_hp = 0.0
        for mon in (mp.get("bench") or []):
            if isinstance(mon, dict):
                bench_hp += float(mon.get("hp") or 0)
        features.set("bench_hp", bench_hp)
        # Opponent bench HP
        opp_bench_hp = 0.0
        for mon in (op.get("bench") or []):
            if isinstance(mon, dict):
                opp_bench_hp += float(mon.get("hp") or 0)
        features.set("opp_bench_hp", opp_bench_hp)
        # Active status
        features.set(
            "has_active",
            1.0 if my_active else 0.0,
        )
        features.set(
            "opp_has_active",
            1.0 if opp_active else 0.0,
        )
        return features


# ============================================================
# Weight Generator
# ============================================================

class WeightGenerator:
    """
    Dynamically generates evaluation weights from the current
    feature vector.

    No weights are permanently stored.
    """

    def __init__(self):
        self.feature_order=[]
        self.weights={}
        self.lambda_reg=0.05
        self.A=None
        self.b=None
        self.samples=0
        self.learning_rate=0.02
        self.feature_visits={}
           
    def visit_count(self,feature:str)->int:
        return self.feature_visits.get(feature,0)
    
    def _ensure_features(
        self,
        features: FeatureVector,
    ):
        for f in features.values.keys():
            if f not in self.feature_order:
                self.feature_order.append(f)
        n = len(self.feature_order)
        if self.A is None:
            self.A = self.lambda_reg * np.eye(n)
            self.b = np.zeros(n)
            return
        old = self.A.shape[0]
        if old == n:
            return
        newA = self.lambda_reg * np.eye(n)
        newA[:old, :old] = self.A
        self.A = newA
        newb = np.zeros(n)
        newb[:old] = self.b
        self.b = newb
    
    def feature_vector(
        self,
        features: FeatureVector,
    ):
        self._ensure_features(features)
        x = np.zeros(len(self.feature_order))
        for i, name in enumerate(self.feature_order):
            x[i] = features.get(name)
        return x
     
    def fit_sample(self,features:FeatureVector,target:float):
        x=self.feature_vector(features)
        if np.linalg.norm(x)<1e-8:
            return
        if self.A is None:
            n=len(x)
            self.A=self.lambda_reg*np.eye(n)
            self.b=np.zeros(n)
        self.A+=np.outer(x,x)
        self.b+=target*x
        try:
            w=np.linalg.solve(self.A,self.b)
        except np.linalg.LinAlgError:
            w=np.linalg.pinv(self.A)@self.b
        self.weights=dict(zip(self.feature_order,w.tolist()))
        self.samples+=1
        for f in features.values:
            self.feature_visits[f]=self.feature_visits.get(f,0)+1
                
    def partial_fit(self,features:FeatureVector,target:float):
        x=self.feature_vector(features)
        if np.linalg.norm(x)<1e-8:
            return
        if self.A is None:
            n=len(x)
            self.A=self.lambda_reg*np.eye(n)
            self.b=np.zeros(n)
        self.A+=np.outer(x,x)
        self.b+=target*x
        try:
            w=np.linalg.solve(self.A,self.b)
        except np.linalg.LinAlgError:
            w=np.linalg.pinv(self.A)@self.b
        self.weights=dict(zip(self.feature_order,w.tolist()))
        self.samples+=1
        for f in features.values:
            self.feature_visits[f]=self.feature_visits.get(f,0)+1
    
    def get(self, feature: str) -> float:
        return 0.0

    def set(self, feature: str, weight: float):
        # retained only for API compatibility
        pass

    def update(self, data: Dict[str, float]):
        # retained only for API compatibility
        pass

    def generate(self,features:Optional[FeatureVector]=None)->Dict[str,float]:
        if features is None:
            return self.weights.copy()
        self._last_features=features.values.copy()
        for f in features.values:
            self.feature_visits[f]=self.feature_visits.get(f,0)+1
        out={}
        for f,v in features.values.items():
            base=self.weights.get(f,0.0)
            scale=1.0/(1.0+abs(v))
            adaptive=base+0.25*v*scale
            if f in self.momentum:
                adaptive+=0.1*self.momentum[f]
            out[f]=adaptive
        return out

    def perturb(self, sigma: float = 0.02):
        pass

    def export(self):
        return {}

    def load(self, weights):
        pass

    def reward(self,state=None,features=None,evaluation=None,lr=0.01):
        if features is None or evaluation is None:
            return
        self.partial_fit(features,evaluation)

# ============================================================
# Evaluator
# ============================================================

@dataclass
class EvaluationResult:
    score: float
    feature_scores: Dict[str,float]
    weights: Dict[str,float]
    uncertainty: float


class Evaluator:

    def __init__(self,generator:WeightGenerator):
        self.generator=generator
        self.lr=0.03
        self.regularization=0.0005
        self.momentum={}
        self.running_variance={}
        self.visit_counts={}
        self.opponent_embedding=None
        
    def set_opponent_embedding(self,embedding):
        self.opponent_embedding=embedding
        
    def estimate_uncertainty(self, features: FeatureVector) -> float:
        u = 0.0
        for name, value in features.values.items():
            n = self.visit_counts.get(name, 0)
            var = self.running_variance.get(name, 1.0)
            u += abs(value) * (var ** 0.5) / ((n + 1) ** 0.5)
        return u
    
    def evaluate(self,features:FeatureVector)->EvaluationResult:
        weights=self.generator.generate(features)
        if self.opponent_embedding is not None:
            attack_rate=self.opponent_embedding[0]
            ability_rate=self.opponent_embedding[1]
            item_rate=self.opponent_embedding[2]
            supporter_rate=self.opponent_embedding[3]
            if attack_rate>0.45:
                weights["my_hp_ratio"]=weights.get("my_hp_ratio",0)+0.6
                weights["energy"]=weights.get("energy",0)+0.15
            if ability_rate>0.25:
                weights["bench_hp"]=weights.get("bench_hp",0)+0.35
            if supporter_rate>0.35:
                weights["my_hand"]=weights.get("my_hand",0)+0.2
            if item_rate>0.45:
                weights["bench"]=weights.get("bench",0)+0.25
        total=0.0
        contrib={}
        variance=0.0
        for name,x in features.values.items():
            w=weights.get(name,0.0)
            c=w*x
            contrib[name]=c
            total+=c
            variance+=(abs(x)+1e-6)/(1+self.visit_counts.get(name,0))
        uncertainty=math.sqrt(variance)
        return EvaluationResult(
            score=total,
            feature_scores=contrib,
            weights=weights,
            uncertainty=uncertainty,
        )
    
    def temporal_difference_update(
        self,
        previous: FeatureVector,
        current: FeatureVector,
        reward: float,
        gamma: float = 0.98,
    ):
        previous_value = self.evaluate(previous).score
        current_value = self.evaluate(current).score
        target = reward + gamma * current_value
        td_error = target - previous_value
        self.generator.fit_sample(
            previous,
            previous_value + td_error,
        )