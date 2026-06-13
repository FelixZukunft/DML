"""Shared constants used across all pipeline phases."""
from __future__ import annotations

import pandas as pd

SEED = 42
N_FOLDS = 5
N_REP = 10

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
