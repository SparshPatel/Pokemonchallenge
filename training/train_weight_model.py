"""
train_weight_model.py
Learns evaluation weights from replay data.
Pipeline
ReplayParser
      ↓
ReplayLoader
      ↓
DatasetBuilder
      ↓
FeatureExtractor
      ↓
WeightModel
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from replay_parser import ReplayParser
from replay_loader import ReplayLoader
from dataset_builder import DatasetBuilder
from feature_extractor import FeatureExtractor

# =====================================================================
# Config
# =====================================================================
@dataclass(slots=True)
class WeightTrainingConfig:
    replay_root: Path
    output_model: Path
    gamma: float = 0.90
    random_seed: int = 42
    test_fraction: float = 0.20
    n_estimators: int = 300
    max_depth: int | None = 16
    min_samples_leaf: int = 2
    
# =====================================================================
# Weight Model
# =====================================================================
class WeightModel:
    def __init__(
        self,
        config: WeightTrainingConfig,
    ):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []
        
    def _discount_targets(self, samples):
        """
        Compute discounted Monte-Carlo return for every replay position.
        Terminal rewards are propagated backwards using gamma.
        """
        targets = np.zeros(len(samples), dtype=np.float32)
        gamma = self.config.gamma
        running_return = 0.0
        for i in range(len(samples) - 1, -1, -1):
            sample = samples[i]
            if sample.terminal:
                running_return = float(sample.reward)
            else:
                running_return *= gamma
            targets[i] = running_return
        return targets

    # -------------------------------------------------------------
    def _load_dataset(self):
        parser = ReplayParser(self.config.replay_root)
        parser.parse()
        loader = ReplayLoader()
        games = loader.load_from_parser(parser)
        builder = DatasetBuilder()
        samples = builder.build(games)
        extractor = FeatureExtractor()
        self.feature_names = extractor.feature_names
        X = []
        for sample in samples:
            feature_sample = extractor.extract(sample)
            X.append(feature_sample.features)
        X = np.asarray(X, dtype=np.float32)
        y = self._discount_targets(samples)
        return X, y

    # -------------------------------------------------------------
    def train(self):
        X, y = self._load_dataset()
        print()
        print("=" * 70)
        print("Weight Model")
        print("=" * 70)
        print()
        print("Samples :", len(X))
        print("Features:", X.shape[1])
        print("Gamma   :", self.config.gamma)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_fraction,
            random_state=self.config.random_seed,
            shuffle=True,
        )
        self.model.fit(
            X_train,
            y_train,
        )
        prediction = self.model.predict(X_test)
        mse = mean_squared_error(
            y_test,
            prediction,
        )
        print("Test MSE :", mse)
        return mse

    # -------------------------------------------------------------
    def save(self):
        self.config.output_model.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        with open(
            self.config.output_model,
            "wb",
        ) as f:
            pickle.dump(
                payload,
                f,
            )
        print()
        print(
            "Saved model to:",
            self.config.output_model,
        )

    # -------------------------------------------------------------
    def export_feature_importance(
        self,
        json_path: Path,
    ):
        importance = dict(
            zip(
                self.feature_names,
                self.model.feature_importances_,
            )
        )
        importance = dict(
            sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                importance,
                f,
                indent=4,
            )
        print(
            "Feature importance exported:",
            json_path,
        )

# =====================================================================
# Main
# =====================================================================
def main():
    config = WeightTrainingConfig(
        replay_root=Path("../replays"),
        output_model=Path("../agent/weight_model.pkl"),
        gamma=0.90,
    )
    trainer = WeightModel(
        config,
    )
    trainer.train()
    trainer.save()
    trainer.export_feature_importance(
        Path("../agent/feature_importance.json"),
    )

if __name__ == "__main__":
    main()