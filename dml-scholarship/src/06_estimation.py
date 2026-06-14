"""Phase 6 — Fit all tuned DML models and save results.

  python src/06_estimation.py     # fits all 4 models, writes results

Downstream scripts import load_results():
  from estimation import load_results
  df = load_results()
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hyperparameter import get_tuned_models   # noqa: E402
from config import D_COL                      # noqa: E402

ROOT       = Path(__file__).parents[1]
MODELS_DIR = ROOT / "outputs" / "models"
TABLES_DIR = ROOT / "outputs" / "tables"

_DISPLAY = {
    "plr_lasso":         "PLR Lasso",
    "plr_random_forest": "PLR Random Forest",
    "plr_xgboost":       "PLR XGBoost",
    "irm_random_forest": "IRM Random Forest",
}
_PKL = {
    "plr_lasso":         "plr_lasso_fitted.pkl",
    "plr_random_forest": "plr_rf_fitted.pkl",
    "plr_xgboost":       "plr_xgb_fitted.pkl",
    "irm_random_forest": "irm_rf_fitted.pkl",
}
_EST_MIN = {
    "plr_lasso":         "5–8",
    "plr_random_forest": "10–15",
    "plr_xgboost":       "5–8",
    "irm_random_forest": "10–15",
}


def _sig(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return "ns"


def _nuisance_rmse(model) -> dict[str, float]:
    """Extract mean RMSE per learner from model.nuisance_loss."""
    try:
        out = {}
        for learner, treat_dict in model.nuisance_loss.items():
            vals = [v for rep in treat_dict.get(D_COL, []) for v in rep
                    if v is not None and not np.isnan(v)]
            out[learner] = round(float(np.mean(vals)), 4) if vals else float("nan")
        return out
    except Exception:
        return {}


def load_results() -> pd.DataFrame:
    """Return the ATE results table written by the __main__ block."""
    return pd.read_csv(TABLES_DIR / "ate_results.csv")


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("06_estimation.py starting — this will take 20-40 minutes", flush=True)
    print("Do not close the terminal.", flush=True)
    print("=" * 60, flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # OLS baseline
    registry = json.loads((MODELS_DIR / "model_registry.json").read_text())
    ols_coef_d = registry["ols_baseline"]["coef_d"]

    # Load all tuned models
    print("\nLoading tuned models...", flush=True)
    bundle = get_tuned_models()
    models = bundle["models"]
    print(f"Loaded: {list(models.keys())}", flush=True)

    # ── Fit loop ──────────────────────────────────────────────────────────
    order = ["plr_lasso", "plr_random_forest", "plr_xgboost", "irm_random_forest"]
    rows  = []
    total_start = time.time()

    for idx, name in enumerate(order, 1):
        model   = models[name]
        display = _DISPLAY[name]

        print(f"\n{'='*60}", flush=True)
        print(f"Fitting: {name}  [{idx}/{len(order)}]", flush=True)
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"Model:   {display} (tuned)", flush=True)
        print(f"This will take approximately {_EST_MIN[name]} minutes...", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        try:
            model.fit()
            elapsed = time.time() - t0

            print(model,         flush=True)
            print(model.summary, flush=True)
            print(f"\nElapsed: {elapsed/60:.1f} minutes", flush=True)

            ate   = float(model.coef.flatten()[0])
            se    = float(model.se.flatten()[0])
            ci    = model.confint()
            ci_lo = float(ci.iloc[0, 0])
            ci_hi = float(ci.iloc[0, 1])
            tstat = float(model.t_stat.flatten()[0])
            pval  = float(model.pval.flatten()[0])
            stab  = float(np.std(model.all_coef.flatten()))
            rmse  = _nuisance_rmse(model)

            rows.append(dict(
                model=name, display=display,
                ATE=ate, SE=se, CI_lower=ci_lo, CI_upper=ci_hi,
                t_stat=tstat, p_value=pval, coef_std=stab,
                nuisance=json.dumps(rmse),
                elapsed_min=round(elapsed / 60, 2),
            ))

            pkl_path = MODELS_DIR / _PKL[name]
            with open(pkl_path, "wb") as fh:
                pickle.dump(model, fh)
            print(f"Saved → {pkl_path}", flush=True)

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR fitting {name}: {exc}", flush=True)
            rows.append(dict(
                model=name, display=display,
                ATE=None, SE=None, CI_lower=None, CI_upper=None,
                t_stat=None, p_value=None, coef_std=None,
                nuisance=None, elapsed_min=round(elapsed / 60, 2),
            ))

    total_elapsed = time.time() - total_start

    # ── Build and save results table ──────────────────────────────────────
    results_df = pd.concat([
        pd.DataFrame([dict(
            model="ols_baseline", display="OLS (naive)",
            ATE=ols_coef_d, SE=None, CI_lower=None, CI_upper=None,
            t_stat=None, p_value=None, coef_std=None,
            nuisance=None, elapsed_min=None,
        )]),
        pd.DataFrame(rows),
    ], ignore_index=True)

    csv_path = TABLES_DIR / "ate_results.csv"
    results_df.to_csv(csv_path, index=False)

    # ── Final comparison table ────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS — ATE of Scholarship on 2nd Semester Grade", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n{'Model':<22} {'ATE':>7} {'SE':>7} {'CI_lo':>7} {'CI_hi':>7} {'p_val':>8}  sig", flush=True)
    print("─" * 67, flush=True)

    for _, row in results_df.iterrows():
        f = lambda v, fmt=".3f": f"{v:{fmt}}" if pd.notna(v) else "  —  "
        stars = _sig(row["p_value"]) if pd.notna(row["p_value"]) else "—"
        print(
            f"{row['display']:<22} {f(row['ATE']):>7} {f(row['SE']):>7} "
            f"{f(row['CI_lower']):>7} {f(row['CI_upper']):>7} "
            f"{f(row['p_value'], '.4f'):>8}  {stars}",
            flush=True,
        )

    print(f"\nSignificance: *** p<0.01  ** p<0.05  * p<0.1  ns = not significant", flush=True)
    print(f"\nOLS naive estimate: {ols_coef_d:.4f} (biased — no confounder adjustment)", flush=True)
    print("DML corrects for selection bias via ML partialling out.", flush=True)

    print(f"\nStability check — std dev of ATE across 10 repetitions:", flush=True)
    for _, row in results_df.iterrows():
        if pd.notna(row.get("coef_std")):
            print(f"  {row['display']:<22} {row['coef_std']:.3f}  (low = stable)", flush=True)

    print(f"\nNuisance model performance (RMSE):", flush=True)
    for _, row in results_df.iterrows():
        if pd.notna(row.get("nuisance")):
            rmse = json.loads(row["nuisance"])
            parts = "  ".join(f"{k} RMSE={v:.4f}" for k, v in rmse.items())
            print(f"  {row['display']:<22} {parts}", flush=True)

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes", flush=True)
    print(f"Results saved → {csv_path}", flush=True)
    print(f"{'='*60}", flush=True)
