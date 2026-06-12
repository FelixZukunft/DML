"""Phase 1 — Clean, encode, and scale UCI student data for DML estimation."""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parents[1]
RAW_CSV    = ROOT / "data" / "raw" / "students_raw.csv"
OUT_CSV    = ROOT / "data" / "processed" / "dml_ready.csv"
SCALER_PKL = ROOT / "outputs" / "models" / "scaler.pkl"

# ── variable definitions ───────────────────────────────────────────────────────
Y_COL = "Curricular units 2nd sem (grade)"
D_COL = "Scholarship holder"

# Note: CSV uses "Marital Status" (capital S) — corrected from spec.
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

SEP = "-" * 60


def run() -> pd.DataFrame:
    # ── step 1: load ──────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_CSV)
    print(SEP)
    print("STEP 1 — LOAD")
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
    print(f"OK — '{D_COL}' present")
    print(f"OK — '{Y_COL}' present")

    # ── step 3: sample description (no rows dropped) ──────────────────────────
    print(SEP)
    print("STEP 3 — SAMPLE DESCRIPTION (full sample, no filtering)")

    Y = df[Y_COL]
    D = df[D_COL]

    # a) total N
    print(f"\na) Total N = {len(df):,}")

    # b) Y == 0
    zero_mask = Y == 0
    print(f"\nb) Y = 0  (withdrawal / failure)")
    print(f"   Count : {zero_mask.sum():,}")
    print(f"   Share : {zero_mask.mean()*100:.1f}%")

    # c) D distribution
    print(f"\nc) Treatment distribution — {D_COL}")
    d_counts = D.value_counts().sort_index()
    for val, cnt in d_counts.items():
        print(f"   D={val}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    # d) Y describe — full sample
    print(f"\nd) Y describe — full sample")
    print(Y.describe().to_string())

    # e) Y describe — conditional on Y > 0
    print(f"\ne) Y describe — conditional on Y > 0  (reference only)")
    print(Y[Y > 0].describe().to_string())

    # f) cross-tab D vs (Y == 0)
    print(f"\nf) Cross-tab: scholarship vs zero-grade rate")
    ct = pd.crosstab(
        D.rename("Scholarship holder"),
        zero_mask.rename("Y == 0"),
        margins=True,
    )
    print(ct)
    print("\n   Row-normalised (share with Y=0 by scholarship status):")
    ct_norm = pd.crosstab(
        D.rename("Scholarship holder"),
        zero_mask.rename("Y == 0"),
        normalize="index",
    ).round(4)
    print(ct_norm)

    # ── step 4: build clean dataframe ─────────────────────────────────────────
    print(SEP)
    print("STEP 4 — BUILD CLEAN DATAFRAME")
    df_clean = df[[Y_COL, D_COL] + X_COLS].copy()
    print(f"Shape after column selection: {df_clean.shape}")

    # ── step 5: encode categoricals ───────────────────────────────────────────
    print(SEP)
    print("STEP 5 — ENCODE CATEGORICALS")
    dummies_list = []
    for col in CATEGORICAL_X:
        dummies = pd.get_dummies(df_clean[col], drop_first=True, prefix=col).astype("int8")
        dummies_list.append(dummies)
        print(f"  {col!r:45s} → {dummies.shape[1]} dummies")

    df_clean = df_clean.drop(columns=CATEGORICAL_X)
    df_clean = pd.concat([df_clean] + dummies_list, axis=1)
    print(f"Shape after encoding: {df_clean.shape}")

    # ── step 6: scale continuous X only ───────────────────────────────────────
    print(SEP)
    print("STEP 6 — SCALE CONTINUOUS X  (Y and D untouched)")
    scaler = StandardScaler()
    df_clean[CONTINUOUS_X] = scaler.fit_transform(df_clean[CONTINUOUS_X])
    SCALER_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler fitted on: {CONTINUOUS_X}")
    print(f"Saved → {SCALER_PKL}")

    # ── step 7: final validation ──────────────────────────────────────────────
    print(SEP)
    print("STEP 7 — FINAL VALIDATION")
    print(f"Final shape           : {df_clean.shape}")
    print(f"Y mean  (should ~10)  : {df_clean[Y_COL].mean():.4f}")
    print(f"D mean  (should ~0.25): {df_clean[D_COL].mean():.4f}")

    x_cols_final = [c for c in df_clean.columns if c not in (Y_COL, D_COL)]
    print(f"\nX columns after encoding ({len(x_cols_final)} total):")
    for c in x_cols_final:
        print(f"  {c}")

    bad_final = [c for c in df_clean.columns if "1st sem" in c]
    assert not bad_final, f"POST-TREATMENT LEAK IN FINAL DF: {bad_final}"
    print("\nFINAL ASSERTION PASSED — no '1st sem' column in output")

    # ── step 8: save ──────────────────────────────────────────────────────────
    print(SEP)
    print("STEP 8 — SAVE")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")

    return df_clean


if __name__ == "__main__":
    run()
