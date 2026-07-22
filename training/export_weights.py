"""
Exports learned feature importances into planner-compatible weights.
Pipeline
feature_importance.json
        ↓
normalize
        ↓
learned_weights.json
"""
from __future__ import annotations
import json
from pathlib import Path
# =====================================================================
# Configuration
# =====================================================================
INPUT_FILE = Path("../agent/feature_importance.json")
OUTPUT_FILE = Path("../agent/learned_weights.json")
# =====================================================================
# Exporter
# =====================================================================
class WeightExporter:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.weights: dict[str, float] = {}

    # -------------------------------------------------------------
    def load(self):
        with open(
            self.input_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.weights = json.load(f)

    # -------------------------------------------------------------
    def normalize(self):
        if not self.weights:
            return
        maximum = max(self.weights.values())
        if maximum <= 0:
            return
        for key in self.weights:
            self.weights[key] /= maximum

    # -------------------------------------------------------------
    def save(self):
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.weights,
                f,
                indent=4,
                sort_keys=True,
            )

    # -------------------------------------------------------------
    def summary(self):
        print()
        print("=" * 70)
        print("Weight Export Summary")
        print("=" * 70)
        print()
        print(
            "Features :",
            len(self.weights),
        )
        print()
        top = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        for name, value in top:
            print(
                f"{name:<30} {value:.4f}"
            )

# =====================================================================
# Main
# =====================================================================
def main():
    exporter = WeightExporter(
        INPUT_FILE,
        OUTPUT_FILE,
    )
    exporter.load()
    exporter.normalize()
    exporter.save()
    exporter.summary()

if __name__ == "__main__":
    main()