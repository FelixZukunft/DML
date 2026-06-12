"""Phase 5 — Hyperparameter tuning for nuisance learners."""
from __future__ import annotations

import doubleml as dml

# Tuning is delegated to DoubleML's built-in tune() method which wraps
# sklearn GridSearchCV / RandomizedSearchCV internally.

RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth":    [3, 5, 7],
    "min_samples_leaf": [3, 5, 10],
}

XGB_PARAM_GRID = {
    "n_estimators":     [200, 400],
    "max_depth":        [3, 5],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}

LASSO_PARAM_GRID = {}   # LassoCV tunes alpha internally via CV


def tune_model(model: dml.DoubleML, learner: str = "rf") -> dml.DoubleML:
    """Call DoubleML's tune() with the appropriate grid."""
    grid = {"rf": RF_PARAM_GRID, "xgb": XGB_PARAM_GRID}.get(learner, {})
    if not grid:
        print(f"[05] No external grid for '{learner}'; skipping tune()")
        return model
    # param_grids keys must match the ml_l / ml_m names in DoubleML
    param_grids = {"ml_l": grid, "ml_m": grid}
    model.tune(param_grids, search_mode="grid_search", n_folds_tune=3)
    print(f"[05] Tuned '{learner}' learner")
    return model


def run(models: dict) -> dict:
    """Tune each model in the dict (key = spec name)."""
    tuned = {}
    for name, model in models.items():
        learner_key = name.split("_")[1] if "_" in name else "rf"
        tuned[name] = tune_model(model, learner_key)
    return tuned
