"""Phase 1 — Clean, encode, and scale UCI student data for DML estimation.

Two encoding modes are available:

  "raw"      — pd.get_dummies(drop_first=True) on raw integer codes.
               Produces many columns with numeric suffixes (e.g. Course_9119).
               Reference category is dropped per variable (standard for regression).
               Output: data/processed/dml_ready_raw.csv

  "semantic" — Named group dummies built from the Excel recoding scheme.
               Produces interpretable column names (e.g. Course_Health).
               All groups kept (no drop_first); reference is chosen at model time.
               Output: data/processed/dml_ready_semantic.csv

Steps 1–4 (load, assertions, description, column selection) and
steps 6–8 (scale, validate, save) are identical for both modes.
Only step 5 (encoding) differs.

Usage
─────
Run both modes at once (default):
    python src/01_data_pipeline.py

Run one mode only:
    python src/01_data_pipeline.py raw
    python src/01_data_pipeline.py semantic

Import in downstream scripts:
    from src.data_pipeline import run
    df_raw      = run("raw")
    df_semantic = run("semantic")

    # or simply read the saved CSVs:
    df = pd.read_csv("data/processed/dml_ready_raw.csv")
    df = pd.read_csv("data/processed/dml_ready_semantic.csv")
"""
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parents[1]
EXCEL_PATH = ROOT / "data" / "raw" / "categorical_recoding_scheme_1.xlsx"
RAW_CSV    = ROOT / "data" / "raw" / "students_raw.csv"
SCALER_PKL = ROOT / "outputs" / "models" / "scaler.pkl"

OUT_PATHS = {
    "raw":      ROOT / "data" / "processed" / "dml_ready_raw.csv",
    "semantic": ROOT / "data" / "processed" / "dml_ready_semantic.csv",
}

# ── variable definitions (shared by both modes) ────────────────────────────────
Y_COL = "Curricular units 2nd sem (grade)"
D_COL = "Scholarship holder"

# Note: CSV stores "Marital Status" with capital S.
X_COLS = [
    "Marital Status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender",
    "Age at enrollment", "International",
    "GDP", "Inflation rate", "Unemployment rate",
]

CONTINUOUS_X = [
    "Previous qualification (grade)", "Admission grade",
    "Age at enrollment", "GDP", "Inflation rate", "Unemployment rate",
]

CATEGORICAL_X = [col for col in X_COLS if col not in CONTINUOUS_X]

# ── semantic encoding: recoding maps (from Excel Sheet 2) ─────────────────────
#    Each dict maps raw integer codes → group integer defined in Sheet 1.

_marital_status_map = {1: 1, 2: 2, 5: 2, 3: 3, 4: 3, 6: 3}

_application_mode_map = {
    1: 1, 17: 1, 18: 1, 39: 2, 42: 3, 43: 3, 51: 3, 57: 3,
    7: 4, 44: 4, 53: 4, 5: 5, 16: 5, 10: 5, 2: 5, 26: 5, 27: 5, 15: 5,
}

_application_order_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 9: 3}

_course_map = {
    9085: 1, 9500: 1, 9556: 1,
    9147: 2, 9254: 2, 9670: 2, 9991: 2,
    171:  3, 9070: 3, 9773: 3,
    8014: 4, 9238: 4, 9853: 4,
    33:   5, 9003: 5, 9119: 5, 9130: 5,
}

_previous_qualification_map = {
    9: 1, 10: 1, 12: 1, 14: 1, 15: 1, 19: 1, 38: 1,
    1: 2, 39: 3, 42: 3,
    2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 40: 4, 43: 4,
}

_nacionality_map = {
    1: 1,
    2: 2, 6: 2, 11: 2, 13: 2, 14: 2, 17: 2, 21: 2, 22: 2, 24: 2,
    25: 2, 26: 2, 32: 2, 41: 2, 62: 2, 100: 2, 101: 2, 103: 2,
    105: 2, 108: 2, 109: 2,
}

_mothers_qualification_map = {
    35: 1, 36: 1, 37: 1,
    11: 2, 26: 2, 30: 2, 38: 2,
    9: 3, 10: 3, 12: 3, 14: 3, 19: 3, 27: 3, 29: 3,
    1: 4, 18: 4, 22: 4, 39: 4, 42: 4,
    2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 40: 5, 41: 5, 43: 5, 44: 5,
    34: 99,
}

_fathers_qualification_map = {
    35: 1, 36: 1, 37: 1,
    11: 2, 26: 2, 30: 2, 38: 2,
    9: 3, 10: 3, 12: 3, 13: 3, 14: 3, 19: 3, 25: 3, 27: 3, 29: 3,
    1: 4, 18: 4, 20: 4, 22: 4, 31: 4, 33: 4, 39: 4, 42: 4,
    2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 40: 5, 41: 5, 43: 5, 44: 5,
    34: 99,
}

_mothers_occupation_map = {
    1: 1, 2: 1, 122: 1, 123: 1, 125: 1,
    3: 2, 131: 2, 132: 2, 134: 2,
    4: 3, 5: 3, 141: 3, 143: 3, 144: 3, 151: 3, 152: 3, 153: 3,
    6: 4, 7: 4, 8: 4, 10: 4, 171: 4, 173: 4, 175: 4,
    9: 5, 191: 5, 192: 5, 193: 5, 194: 5,
    0: 6, 90: 6, 99: 6,
}

_fathers_occupation_map = {
    1: 1, 2: 1, 112: 1, 114: 1, 121: 1, 122: 1, 123: 1, 124: 1,
    3: 2, 131: 2, 132: 2, 134: 2, 135: 2,
    4: 3, 5: 3, 141: 3, 143: 3, 144: 3, 151: 3, 152: 3, 153: 3, 154: 3,
    6: 4, 7: 4, 8: 4, 10: 4, 101: 4, 102: 4, 103: 4, 161: 4, 163: 4,
    171: 4, 172: 4, 174: 4, 175: 4, 181: 4, 182: 4, 183: 4,
    9: 5, 192: 5, 193: 5, 194: 5, 195: 5,
    0: 6, 90: 6, 99: 6,
}

# (csv_column_name, recode_map, Sheet1_variable_name)
RECODINGS = [
    ("Marital Status",         _marital_status_map,         "Marital Status"),
    ("Application mode",       _application_mode_map,       "Application mode"),
    ("Application order",      _application_order_map,      "Application order"),
    ("Course",                 _course_map,                 "Course"),
    ("Previous qualification", _previous_qualification_map, "Previous qualification"),
    ("Nacionality",            _nacionality_map,            "Nacionality"),
    ("Mother's qualification", _mothers_qualification_map,  "Mother's qualification"),
    ("Father's qualification", _fathers_qualification_map,  "Father's qualification"),
    ("Mother's occupation",    _mothers_occupation_map,     "Mother's occupation"),
    ("Father's occupation",    _fathers_occupation_map,     "Father's occupation"),
]

SEP = "-" * 60


# ── helpers for semantic encoding ─────────────────────────────────────────────

def _clean_name(s: str) -> str:
    """Convert a label to a valid column-name fragment."""
    s = str(s).replace(" ", "_")
    s = re.sub(r"['/&()–\-,.]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _load_group_names() -> dict:
    """Return {sheet1_variable: {group_int: group_name}} from Excel Sheet 1."""
    sheet1 = pd.read_excel(EXCEL_PATH, sheet_name=0)
    sheet1 = sheet1.dropna(subset=["Variable", "New Group (integer)", "New Group Name"])
    result: dict = {}
    for _, row in sheet1.iterrows():
        var      = row["Variable"]
        grp_int  = int(row["New Group (integer)"])
        grp_name = str(row["New Group Name"])
        result.setdefault(var, {})[grp_int] = grp_name
    return result


# ── encoding functions (step 5) ───────────────────────────────────────────────

def _encode_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    pd.get_dummies(drop_first=True) on every column in CATEGORICAL_X.
    Produces columns like 'Course_9119', 'Marital Status_2', etc.
    Reference category is dropped per variable (standard for regression).
    Binary variables (Displaced, Gender, …) each produce a single dummy
    identical to the original column.
    """
    dummies_list = []
    for col in CATEGORICAL_X:
        dummies = pd.get_dummies(df[col], drop_first=True, prefix=col).astype("int8")
        dummies_list.append(dummies)
        print(f"  {col!r:45s} → {dummies.shape[1]} dummies")
    df = df.drop(columns=CATEGORICAL_X)
    df = pd.concat([df] + dummies_list, axis=1)
    return df


def _encode_semantic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Named group dummies from the Excel recoding scheme (Sheet 1 + Sheet 2).
    Produces columns like 'Course_Health', 'Mothers_qualification_Unknown', etc.
    All groups are kept (no drop_first); choose the reference at model time.
    The 6 remaining binary variables in CATEGORICAL_X that are not in RECODINGS
    (Daytime/evening attendance, Displaced, Educational special needs, Debtor,
    Tuition fees up to date, Gender, International) stay as raw 0/1 integers.
    """
    group_names = _load_group_names()
    recoded_cols = {csv_col for csv_col, _, _ in RECODINGS}

    for csv_col, recode_map, sheet1_var in RECODINGS:
        col_prefix = _clean_name(csv_col)
        recoded    = df[csv_col].map(recode_map)

        unmapped = recoded.isna().sum()
        if unmapped:
            raise ValueError(
                f"Column '{csv_col}': {unmapped} values not in recode map. "
                f"Raw values present: {sorted(df[csv_col].unique())}"
            )

        grp_lookup = group_names[sheet1_var]
        new_cols   = []
        for grp_int, grp_name in sorted(grp_lookup.items()):
            col_name = f"{col_prefix}_{_clean_name(grp_name)}"
            df[col_name] = (recoded == grp_int).astype("int8")
            new_cols.append(col_name)

        df = df.drop(columns=[csv_col])
        print(f"  {csv_col!r:45s} → {len(new_cols)} named groups: {new_cols}")

    # report the binary vars that stayed as-is
    binary_passthrough = [c for c in CATEGORICAL_X if c not in recoded_cols]
    if binary_passthrough:
        print(f"  Binary vars kept as 0/1 integers: {binary_passthrough}")

    return df


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(encoding: str = "raw") -> pd.DataFrame:
    """
    Run the full data pipeline.

    Parameters
    ----------
    encoding : "raw" | "semantic"
        Selects the categorical encoding strategy for step 5.
    """
    if encoding not in OUT_PATHS:
        raise ValueError(f"encoding must be 'raw' or 'semantic', got {encoding!r}")

    out_csv = OUT_PATHS[encoding]

    print(SEP)
    print(f"PIPELINE — encoding = '{encoding}'")

    # ── step 1: load ──────────────────────────────────────────────────────────
    print(SEP)
    print("STEP 1 — LOAD")
    df = pd.read_csv(RAW_CSV)
    print(f"Raw shape: {df.shape}")

    # ── step 2: safety assertions ─────────────────────────────────────────────
    print(SEP)
    print("STEP 2 — SAFETY ASSERTIONS")
    bad = [c for c in X_COLS if ("1st sem" in c or "2nd sem" in c)]
    if bad:
        raise ValueError(f"POST-TREATMENT VARIABLE DETECTED: {bad}")
    assert D_COL in df.columns, f"Missing treatment column: {D_COL}"
    assert Y_COL in df.columns, f"Missing outcome column: {Y_COL}"
    print("OK — no post-treatment variables in X_COLS")
    print(f"OK — '{D_COL}' and '{Y_COL}' present")

    # ── step 3: sample description (no rows dropped) ──────────────────────────
    print(SEP)
    print("STEP 3 — SAMPLE DESCRIPTION (full sample, no filtering)")
    Y, D = df[Y_COL], df[D_COL]
    zero_mask = Y == 0

    print(f"\na) Total N = {len(df):,}")

    print(f"\nb) Y = 0  (withdrawal / failure)")
    print(f"   Count : {zero_mask.sum():,}  ({zero_mask.mean()*100:.1f}%)")

    print(f"\nc) Treatment distribution — {D_COL}")
    for val, cnt in D.value_counts().sort_index().items():
        print(f"   D={val}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    print(f"\nd) Y describe — full sample")
    print(Y.describe().to_string())

    print(f"\ne) Y describe — conditional on Y > 0  (reference only)")
    print(Y[Y > 0].describe().to_string())

    print(f"\nf) Cross-tab: scholarship vs zero-grade rate")
    print(pd.crosstab(D.rename("D"), zero_mask.rename("Y==0"), margins=True))
    print("\n   Row-normalised:")
    print(pd.crosstab(D.rename("D"), zero_mask.rename("Y==0"), normalize="index").round(4))

    # ── step 4: build clean dataframe ─────────────────────────────────────────
    print(SEP)
    print("STEP 4 — SELECT COLUMNS  (Y, D, X only — no 1st-sem, no Target)")
    df_clean = df[[Y_COL, D_COL] + X_COLS].copy()
    print(f"Shape: {df_clean.shape}")

    # ── step 5: encode categoricals ───────────────────────────────────────────
    print(SEP)
    print(f"STEP 5 — ENCODE CATEGORICALS  (mode: {encoding})")
    if encoding == "raw":
        df_clean = _encode_raw(df_clean)
    else:
        df_clean = _encode_semantic(df_clean)
    print(f"Shape after encoding: {df_clean.shape}")

    # ── step 6: scale continuous X only ───────────────────────────────────────
    print(SEP)
    print("STEP 6 — SCALE CONTINUOUS X  (Y and D untouched)")
    scaler = StandardScaler()
    df_clean[CONTINUOUS_X] = scaler.fit_transform(df_clean[CONTINUOUS_X])
    SCALER_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Continuous X scaled: {CONTINUOUS_X}")
    print(f"Scaler saved → {SCALER_PKL}")

    # ── step 7: final validation ──────────────────────────────────────────────
    print(SEP)
    print("STEP 7 — FINAL VALIDATION")
    print(f"Final shape           : {df_clean.shape}")
    print(f"Y mean  (should ~10)  : {df_clean[Y_COL].mean():.4f}")
    print(f"D mean  (should ~0.25): {df_clean[D_COL].mean():.4f}")

    x_cols_final = [c for c in df_clean.columns if c not in (Y_COL, D_COL)]
    print(f"X columns after encoding: {len(x_cols_final)} total")

    bad_final = [c for c in df_clean.columns if "1st sem" in c]
    assert not bad_final, f"POST-TREATMENT LEAK: {bad_final}"
    print("ASSERTION PASSED — no '1st sem' column in output")

    # ── step 8: save ──────────────────────────────────────────────────────────
    print(SEP)
    print("STEP 8 — SAVE")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    return df_clean


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("raw", "semantic"):
        run(mode)
    else:
        run("raw")
        print()
        run("semantic")

#Run both modes (default):
#python /Users/felixzukunft/DML/dml-scholarship/src/01_data_pipeline.py 

#python /Users/felixzukunft/DML/dml-scholarship/src/01_data_pipeline.py raw
#python /Users/felixzukunft/DML/dml-scholarship/src/01_data_pipeline.py semantic
#semantic = combining columns
