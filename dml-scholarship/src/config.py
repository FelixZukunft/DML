"""Shared constants used across all pipeline phases."""

SEED = 42
N_FOLDS = 5
N_REP = 1

# ── Variable specification ─────────────────────────────────────────────────
# Set these after EDA to match your processed dataset's column names.
OUTCOME = "Y"
TREATMENT = "D"
CONTROLS: list[str] = []   # empty = use all remaining columns
