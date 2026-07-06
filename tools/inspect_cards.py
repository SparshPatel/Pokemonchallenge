"""Quick one-off inspection of the card CSV schema and value distributions."""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "EN_Card_Data.csv"

df = pd.read_csv(CSV, encoding="utf-8", dtype=str).fillna("")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

stage_col = "Stage (Pokémon)/Type (Energy and Trainer)"
print("\n--- Category counts ---")
print(df["Category"].value_counts())

print("\n--- Stage/Type counts ---")
print(df[stage_col].value_counts())

print("\n--- Rule counts (non-empty) ---")
print(df[df["Rule"] != ""]["Rule"].value_counts())

print("\n--- Unique cards (by Card ID) ---")
print("unique card ids:", df["Card ID"].nunique())

print("\n--- Sample Cost values ---")
print(df[df["Cost"] != ""]["Cost"].drop_duplicates().head(20).tolist())

print("\n--- Sample Damage values ---")
print(df[df["Damage"] != ""]["Damage"].drop_duplicates().head(20).tolist())

print("\n--- Type values ---")
print(df[df["Type"] != ""]["Type"].value_counts())
