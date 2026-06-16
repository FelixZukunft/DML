"""Shared constants used across all pipeline phases."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SEED = 42
N_FOLDS = 5
N_REP = 10

# ── Dataset selection ──────────────────────────────────────────────────────
# Two encodings of the same data are produced by 01_data_pipeline.py. The
# modelling phases (specs, tuning, estimation, inference, sensitivity) can run
# on either one — pick which here, via CLI arg or the $DML_DATASET env var.
ROOT = Path(__file__).parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

DATA_FILES = {
    "raw":      PROCESSED_DIR / "dml_ready_raw.csv",       # 231 one-hot X features
    "semantic": PROCESSED_DIR / "dml_ready_semantic.csv",  # 59 named-group X features
}
DEFAULT_DATASET = "raw"
DATASET_ENV_VAR = "DML_DATASET"


def resolve_dataset(which: str | None = None) -> tuple[str, Path]:
    """Resolve which processed dataset the modelling phases run on.

    Priority: explicit ``which`` arg > ``$DML_DATASET`` env var > DEFAULT_DATASET.
    Returns ``(key, path)``. Raises if the key is unknown or the file is missing.
    """
    key = (which or os.environ.get(DATASET_ENV_VAR) or DEFAULT_DATASET).strip().lower()
    if key not in DATA_FILES:
        raise ValueError(f"Unknown dataset {key!r}; choose from {sorted(DATA_FILES)}")
    path = DATA_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Run `python src/01_data_pipeline.py {key}` to generate it."
        )
    return key, path


def dataset_from_argv(argv: list[str]) -> str | None:
    """Pull a dataset selector ('raw'|'semantic') out of CLI args, if present."""
    for arg in argv:
        token = arg.strip().lower()
        if token in DATA_FILES:
            return token
    return None

# ── Variable specification ─────────────────────────────────────────────────
# Actual column names in data/processed/dml_ready.csv (headers already stripped).
Y_COL = "Curricular units 2nd sem (grade)"   # outcome — continuous
D_COL = "Scholarship holder"                 # treatment — binary 0/1

# Legacy aliases kept for backward compatibility with earlier scaffold code.
OUTCOME = Y_COL
TREATMENT = D_COL
CONTROLS: list[str] = []   # empty = use all remaining columns


def get_x_cols_encoded(df: pd.DataFrame) -> list[str]:
    """Return the encoded confounder columns from a processed dataframe.

    X = every column except the outcome (Y) and treatment (D). Works on the
    fully one-hot-encoded dml_ready.csv. Column names are stripped defensively
    in case the file was read without cleaning headers.
    """
    df.columns = df.columns.str.strip()
    assert Y_COL in df.columns, f"Outcome column {Y_COL!r} not found"
    assert D_COL in df.columns, f"Treatment column {D_COL!r} not found"
    return [c for c in df.columns if c not in (Y_COL, D_COL)]
