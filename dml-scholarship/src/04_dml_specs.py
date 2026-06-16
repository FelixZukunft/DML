"""Phase 4 — DoubleML model specifications.

Defines the data backend, the PLR models (primary) and IRM model (robustness),
a naive OLS/Ridge baseline, and a JSON model registry. No model is fitted here.

  python src/04_dml_specs.py            # configure on default dataset (raw)
  python src/04_dml_specs.py semantic   # configure on the semantic dataset

Downstream phases import the model factory:
  from dml_specs import get_all_models
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
import doubleml as dml
from sklearn.base import clone
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from config import (                                   # noqa: E402
    Y_COL, D_COL, get_x_cols_encoded, resolve_dataset,
)
from ml_methods import get_learners                    # noqa: E402

ROOT = Path(__file__).parents[1]


# ── Model factory (imported by 05_estimation.py) ───────────────────────────
def get_all_models(dataset: str | None = None) -> dict:
    """Instantiate all DML models fresh (unfitted).

    Args:
      dataset — 'raw' or 'semantic' (default: $DML_DATASET or 'raw'). Selects
                which processed CSV the DoubleMLData backend is built from.

    Returns dict with keys:
      'models'   — {'plr_lasso', 'plr_random_forest', 'plr_xgboost',
                    'irm_random_forest'} → unfitted DoubleML model objects
      'dml_data' — the DoubleMLData backend
      'dataset'  — the resolved dataset key ('raw' | 'semantic')

    Usage in 05_estimation.py:
      from dml_specs import get_all_models
      bundle = get_all_models()
      dml_data = bundle['dml_data']
      for name, model in bundle['models'].items():
          print(f"Fitting {name}...")
          model.fit()
          print(model.summary)
    """
    dataset_key, data_path = resolve_dataset(dataset)
    df = pd.read_csv(data_path)
    x_cols = get_x_cols_encoded(df)
    data_obj = dml.DoubleMLData(
        df, y_col=Y_COL, d_cols=D_COL, x_cols=x_cols,
        use_other_treat_as_covariate=False,
    )
    learners = get_learners()

    all_models = {
        "plr_lasso": dml.DoubleMLPLR(
            data_obj,
            ml_l=clone(learners["lasso"]["ml_l"]),
            ml_m=clone(learners["lasso"]["ml_m"]),
            n_folds=5, n_rep=10, score="partialling out"),
        "plr_random_forest": dml.DoubleMLPLR(
            data_obj,
            ml_l=clone(learners["random_forest"]["ml_l"]),
            ml_m=clone(learners["random_forest"]["ml_m"]),
            n_folds=5, n_rep=10, score="partialling out"),
        "plr_xgboost": dml.DoubleMLPLR(
            data_obj,
            ml_l=clone(learners["xgboost"]["ml_l"]),
            ml_m=clone(learners["xgboost"]["ml_m"]),
            n_folds=5, n_rep=10, score="partialling out"),
        "irm_random_forest": dml.DoubleMLIRM(
            data_obj,
            ml_g=clone(learners["random_forest"]["ml_g"]),
            ml_m=clone(learners["random_forest"]["ml_m"]),
            n_folds=5, n_rep=10, score="ATE"),
    }

    return {"models": all_models, "dml_data": data_obj, "dataset": dataset_key}


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.feature_selection import SelectKBest, f_regression
    from config import dataset_from_argv

    # ─── STEP 1: Load data and build DoubleMLData object ─────────────────────
    dataset_key, data_path = resolve_dataset(dataset_from_argv(sys.argv[1:]))
    print(f"Dataset: {dataset_key}  ({data_path.name})")
    df = pd.read_csv(data_path)
    x_cols = get_x_cols_encoded(df)

    # Safety check: confirm no post-treatment variables leaked in
    assert not any("1st sem" in c for c in x_cols), "POST-TREATMENT LEAK DETECTED"
    min_x = 200 if dataset_key == "raw" else 40
    assert len(x_cols) > min_x, f"Expected >{min_x} X cols for {dataset_key}, got {len(x_cols)}"

    obj_dml_data = dml.DoubleMLData(
        df,
        y_col=Y_COL,
        d_cols=D_COL,
        x_cols=x_cols,
        use_other_treat_as_covariate=False,  # only one treatment variable
    )

    print(obj_dml_data)

    # ─── STEP 2: Define PLR models (primary) ─────────────────────────────────
    learners = get_learners()
    models = {}

    for config_name in ["lasso", "random_forest", "xgboost"]:
        config = learners[config_name]

        model = dml.DoubleMLPLR(
            obj_dml_data,
            ml_l=clone(config["ml_l"]),
            ml_m=clone(config["ml_m"]),
            n_folds=5,
            n_rep=10,
            score="partialling out",
        )

        key = f"plr_{config_name}"
        models[key] = model

        print(f"\n{'='*50}")
        print(f"PLR ({config_name}) — CONFIGURED (not fitted)")
        print(model)
        print(f"ml_l: {type(config['ml_l']).__name__}")
        print(f"ml_m: {type(config['ml_m']).__name__}")

    # ─── STEP 3: Define IRM model (robustness check) ─────────────────────────
    rf_config = learners["random_forest"]

    irm_model = dml.DoubleMLIRM(
        obj_dml_data,
        ml_g=clone(rf_config["ml_g"]),
        ml_m=clone(rf_config["ml_m"]),
        n_folds=5,
        n_rep=10,
        score="ATE",
    )

    models["irm_random_forest"] = irm_model

    print("\n" + "=" * 50)
    print("IRM (random_forest) — CONFIGURED (not fitted)")
    print(irm_model)
    print("Note: IRM fits ml_g0 for D=0 group and ml_g1 for D=1 group")
    print("      This allows fully heterogeneous treatment effects")

    # ─── STEP 4: Naive OLS baseline ──────────────────────────────────────────
    X_arr = df[x_cols].values
    y_arr = df[Y_COL].values
    d_arr = df[D_COL].values

    # Include D in X for the baseline (naive: no partialling out)
    X_with_d = np.column_stack([d_arr, X_arr])

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_with_d, y_arr)

    ols_coef_d = ridge.coef_[0]  # coefficient on D (scholarship)

    # For std error and CI, run OLS on top 20 most important X only
    # (to avoid multicollinearity in statsmodels)
    selector = SelectKBest(f_regression, k=20)
    X_top20 = selector.fit_transform(X_arr, y_arr)
    X_sm = sm.add_constant(np.column_stack([d_arr, X_top20]))
    ols_sm = sm.OLS(y_arr, X_sm).fit()

    ci = ols_sm.conf_int()  # shape (k, 2): row 1 is D → [lower, upper]
    print("\n" + "=" * 50)
    print("NAIVE OLS BASELINE (biased — no partialling out)")
    print(f"  Ridge coefficient on D (scholarship): {ols_coef_d:.4f}")
    print(f"  Statsmodels OLS on D (top-20 X): {ols_sm.params[1]:.4f}")
    print(f"  Std Error: {ols_sm.bse[1]:.4f}")
    print(f"  95% CI: [{ci[1][0]:.4f}, {ci[1][1]:.4f}]")
    print(f"  p-value: {ols_sm.pvalues[1]:.4f}")
    print("  Interpretation: this is the BIASED estimate before")
    print("  DML removes confounding. DML ATE will differ from this.")

    # ─── STEP 5: Save model registry ─────────────────────────────────────────
    registry = {
        "dataset": dataset_key,
        "plr_lasso": {
            "type": "PLR",
            "learner": "lasso",
            "ml_l": "LassoCV",
            "ml_m": "LogisticRegressionCV",
            "n_folds": 5,
            "n_rep": 10,
            "score": "partialling out",
            "fitted": False,
            "purpose": "linear baseline",
        },
        "plr_random_forest": {
            "type": "PLR",
            "learner": "random_forest",
            "ml_l": "RandomForestRegressor",
            "ml_m": "RandomForestClassifier",
            "n_folds": 5,
            "n_rep": 10,
            "score": "partialling out",
            "fitted": False,
            "purpose": "primary non-linear estimator",
        },
        "plr_xgboost": {
            "type": "PLR",
            "learner": "xgboost",
            "ml_l": "XGBRegressor",
            "ml_m": "XGBClassifier",
            "n_folds": 5,
            "n_rep": 10,
            "score": "partialling out",
            "fitted": False,
            "purpose": "gradient boosting alternative",
        },
        "irm_random_forest": {
            "type": "IRM",
            "learner": "random_forest",
            "ml_g": "RandomForestRegressor",
            "ml_m": "RandomForestClassifier",
            "n_folds": 5,
            "n_rep": 10,
            "score": "ATE",
            "fitted": False,
            "purpose": "robustness check — heterogeneous effects",
        },
        "ols_baseline": {
            "type": "OLS",
            "learner": "Ridge+Statsmodels",
            "fitted": True,
            "coef_d": round(float(ols_coef_d), 6),
            "purpose": "naive biased baseline for comparison",
        },
    }

    out_dir = ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nRegistry saved → {out_dir / 'model_registry.json'}")

    # ─── FINAL OUTPUT ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("04_dml_specs.py — COMPLETE")
    print(f"  Dataset: {dataset_key}  ({data_path.name})")
    print(f"  Data: {df.shape[0]} obs × {len(x_cols)} X columns")
    print(f"  Models defined: PLR×3 + IRM×1")
    print(f"  OLS baseline: fitted (coef_D = {ols_coef_d:.4f})")
    print(f"  Registry: outputs/models/model_registry.json")
    print("  Next step: python src/05_estimation.py")
    print("  WARNING: estimation will take 15-30 minutes")
    print("=" * 60)

#cd /Users/felixzukunft/DML/dml-scholarship
#python src/04_dml_specs.py