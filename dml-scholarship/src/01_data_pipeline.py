"""Phase 1 — Load, clean, and feature-engineer the UCI dataset."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"


def load_raw() -> pd.DataFrame:
    """Read the raw UCI CSV.  Update the filename to match your download."""
    path = RAW_DIR / "dataset.csv"          # ← rename to match actual file
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with any missing value — revisit after EDA
    df = df.dropna()
    # Add cleaning steps here (e.g. outlier removal, dtype coercion)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Add derived columns here (e.g. log transforms, interaction terms)
    return df


def run() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"[01] Saved {len(df):,} rows → {out}")
    return df


if __name__ == "__main__":
    run()
