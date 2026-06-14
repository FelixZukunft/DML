"""Phase 5 — Optuna hyperparameter tuning for DML nuisance learners.

Tunes PLR (RF), PLR (XGBoost), and IRM (RF) using tune_ml_models().
Lasso PLR is self-tuning via LassoCV/LogisticRegressionCV — no Optuna needed.

  python src/05_hyperparameter.py     # runs all tuning, writes best_hyperparams.json

Downstream phases import the tuned model factory:
  from hyperparameter import get_tuned_models
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import doubleml as dml
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from config import D_COL, Y_COL, get_x_cols_encoded   # noqa: E402
from ml_methods import get_learners                    # noqa: E402

ROOT      = Path(__file__).parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dml_ready.csv"
OUT_DIR   = ROOT / "outputs" / "models"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Param extractors ───────────────────────────────────────────────────────

def extract_best_params(model: dml.DoubleML, learner_key: str) -> dict:
    """Return the best params dict for learner_key from a tuned DoubleML model.

    After tune_ml_models() the structure is:
      model.params[learner_key][D_COL][rep_idx][fold_idx]
    All folds share the same tuned params, so [0][0] is sufficient.
    """
    return model.params[learner_key][D_COL][0][0]


# ── Param-space definitions ────────────────────────────────────────────────

def ml_l_rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }


def ml_m_rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }


def ml_l_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def ml_m_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def ml_g_irm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }


def ml_m_irm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
    }


# ── Tuned model factory (imported by 06_estimation.py) ────────────────────

def get_tuned_models() -> dict:
    """Instantiate all 4 DML models with tuned hyperparameters (unfitted).

    Loads best_hyperparams.json written by this script's __main__ block.
    Lasso PLR is included with its original self-tuning spec (no Optuna params).

    Returns:
      {'models': {name: unfitted_tuned_model}, 'dml_data': DoubleMLData}
    """
    params_path = OUT_DIR / "best_hyperparams.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found — run `python src/05_hyperparameter.py` first."
        )
    best = json.loads(params_path.read_text())

    df = pd.read_csv(DATA_PATH)
    x_cols = get_x_cols_encoded(df)
    data_obj = dml.DoubleMLData(
        df, y_col=Y_COL, d_cols=D_COL, x_cols=x_cols,
        use_other_treat_as_covariate=False,
    )
    learners = get_learners()

    # PLR Lasso — self-tuning, no Optuna params
    plr_lasso = dml.DoubleMLPLR(
        data_obj,
        ml_l=clone(learners["lasso"]["ml_l"]),
        ml_m=clone(learners["lasso"]["ml_m"]),
        n_folds=5, n_rep=10, score="partialling out",
    )

    # PLR Random Forest — tuned params applied
    plr_rf = dml.DoubleMLPLR(
        data_obj,
        ml_l=RandomForestRegressor(random_state=42, n_jobs=-1),
        ml_m=RandomForestClassifier(random_state=42, n_jobs=-1),
        n_folds=5, n_rep=10, score="partialling out",
    )
    plr_rf.set_ml_nuisance_params("ml_l", D_COL, best["plr_random_forest"]["ml_l"])
    plr_rf.set_ml_nuisance_params("ml_m", D_COL, best["plr_random_forest"]["ml_m"])

    # PLR XGBoost — tuned params applied
    plr_xgb = dml.DoubleMLPLR(
        data_obj,
        ml_l=XGBRegressor(random_state=42, verbosity=0),
        ml_m=XGBClassifier(random_state=42, verbosity=0),
        n_folds=5, n_rep=10, score="partialling out",
    )
    plr_xgb.set_ml_nuisance_params("ml_l", D_COL, best["plr_xgboost"]["ml_l"])
    plr_xgb.set_ml_nuisance_params("ml_m", D_COL, best["plr_xgboost"]["ml_m"])

    # IRM Random Forest — tuned params applied
    irm_rf = dml.DoubleMLIRM(
        data_obj,
        ml_g=RandomForestRegressor(random_state=42, n_jobs=-1),
        ml_m=RandomForestClassifier(random_state=42, n_jobs=-1),
        n_folds=5, n_rep=10, score="ATE",
    )
    irm_rf.set_ml_nuisance_params("ml_g0", D_COL, best["irm_random_forest"]["ml_g0"])
    irm_rf.set_ml_nuisance_params("ml_g1", D_COL, best["irm_random_forest"]["ml_g1"])
    irm_rf.set_ml_nuisance_params("ml_m",  D_COL, best["irm_random_forest"]["ml_m"])

    return {
        "models": {
            "plr_lasso":         plr_lasso,
            "plr_random_forest": plr_rf,
            "plr_xgboost":       plr_xgb,
            "irm_random_forest": irm_rf,
        },
        "dml_data": data_obj,
    }


# ── Main: run tuning ───────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_SETTINGS = {"n_trials": 30, "verbosity": optuna.logging.WARNING}

    # ─── STEP 1: Load data ────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    x_cols = get_x_cols_encoded(df)
    obj_dml_data = dml.DoubleMLData(
        df, y_col=Y_COL, d_cols=D_COL, x_cols=x_cols,
        use_other_treat_as_covariate=False,
    )
    print(f"Data loaded: {df.shape[0]} obs × {len(x_cols)} X columns")

    # ─── STEP 2: Tune PLR — Random Forest ────────────────────────────────
    plr_rf = dml.DoubleMLPLR(
        obj_dml_data,
        ml_l=RandomForestRegressor(random_state=42, n_jobs=-1),
        ml_m=RandomForestClassifier(random_state=42, n_jobs=-1),
        n_folds=5, n_rep=3, score="partialling out",
    )
    print("\nTuning PLR — Random Forest (30 Optuna trials)...")
    t0 = time.time()
    plr_rf.tune_ml_models(
        ml_param_space={"ml_l": ml_l_rf_params, "ml_m": ml_m_rf_params},
        optuna_settings=OPTUNA_SETTINGS,
    )
    elapsed = time.time() - t0
    print(f"RF tuning done in {elapsed/60:.1f} minutes")
    print("Best RF params found:")
    print(json.dumps(plr_rf.params, indent=2, default=str))

    # ─── STEP 3: Tune PLR — XGBoost ──────────────────────────────────────
    plr_xgb = dml.DoubleMLPLR(
        obj_dml_data,
        ml_l=XGBRegressor(random_state=42, verbosity=0),
        ml_m=XGBClassifier(random_state=42, verbosity=0),
        n_folds=5, n_rep=3, score="partialling out",
    )
    print("\nTuning PLR — XGBoost (30 Optuna trials)...")
    t0 = time.time()
    plr_xgb.tune_ml_models(
        ml_param_space={"ml_l": ml_l_xgb_params, "ml_m": ml_m_xgb_params},
        optuna_settings=OPTUNA_SETTINGS,
    )
    elapsed = time.time() - t0
    print(f"XGB tuning done in {elapsed/60:.1f} minutes")
    print("Best XGB params found:")
    print(json.dumps(plr_xgb.params, indent=2, default=str))

    # ─── STEP 4: Tune IRM — Random Forest ────────────────────────────────
    irm_rf = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=RandomForestRegressor(random_state=42, n_jobs=-1),
        ml_m=RandomForestClassifier(random_state=42, n_jobs=-1),
        n_folds=5, n_rep=3, score="ATE",
    )
    print("\nTuning IRM — Random Forest (30 Optuna trials)...")
    t0 = time.time()
    irm_rf.tune_ml_models(
        ml_param_space={"ml_g": ml_g_irm_params, "ml_m": ml_m_irm_params},
        optuna_settings=OPTUNA_SETTINGS,
    )
    elapsed = time.time() - t0
    print(f"IRM RF tuning done in {elapsed/60:.1f} minutes")
    print("Best IRM RF params found:")
    print(json.dumps(irm_rf.params, indent=2, default=str))

    # ─── STEP 5: Extract and save best params ────────────────────────────
    best_params = {
        "plr_random_forest": {
            "ml_l": extract_best_params(plr_rf,  "ml_l"),
            "ml_m": extract_best_params(plr_rf,  "ml_m"),
        },
        "plr_xgboost": {
            "ml_l": extract_best_params(plr_xgb, "ml_l"),
            "ml_m": extract_best_params(plr_xgb, "ml_m"),
        },
        "irm_random_forest": {
            "ml_g0": extract_best_params(irm_rf, "ml_g0"),
            "ml_g1": extract_best_params(irm_rf, "ml_g1"),
            "ml_m":  extract_best_params(irm_rf, "ml_m"),
        },
    }

    params_path = OUT_DIR / "best_hyperparams.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"\nBest params saved → {params_path}")
    print(json.dumps(best_params, indent=2))

    # ─── STEP 7: Summary table ────────────────────────────────────────────
    print(f"\n{'Model':<22} {'Learner':<8} {'Parameter':<22} Value")
    print("-" * 68)
    for model_key, learners_dict in best_params.items():
        for learner_key, params in learners_dict.items():
            for param, val in params.items():
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                print(f"{model_key:<22} {learner_key:<8} {param:<22} {val_str}")

    print(
        "\nTuning complete. Best params saved to:\n"
        f"  {params_path}\n\n"
        "Next step: python src/06_estimation.py\n"
        "06_estimation.py will import get_tuned_models() from this file\n"
        "and run .fit() on all 4 tuned models.\n"
        "Expected runtime: 20-40 minutes."
    )
