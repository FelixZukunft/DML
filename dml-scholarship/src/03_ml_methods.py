"""Phase 3 — ML learner configurations for DML nuisance estimation.

Each config supplies three fresh (unfitted) sklearn-compatible estimators:
  ml_l  — regressor  for E[Y | X]       (outcome nuisance, used in PLR + IRM)
  ml_m  — classifier for E[D | X]       (propensity nuisance)
  ml_g  — regressor  for E[Y | X, D]    (IRM-specific outcome nuisance; same spec as ml_l)

Usage:
    from src.03_ml_methods import get_learners
    configs = get_learners()
    ml_l = configs['random_forest']['ml_l']   # fresh unfitted instance
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from xgboost import XGBClassifier, XGBRegressor

# ── Learner blueprints (never mutate these; clone() before use) ────────────

_LASSO_L = LassoCV(
    cv=5, random_state=42, max_iter=10_000,
)
_LOGISTIC_M = LogisticRegressionCV(
    cv=5, random_state=42, max_iter=1_000, penalty="l1", solver="liblinear",
)

_RF_L = RandomForestRegressor(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)
_RF_M = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)

_XGB_L = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    random_state=42, verbosity=0,
)
_XGB_M = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    random_state=42, verbosity=0,
)


# ── Public API ─────────────────────────────────────────────────────────────

def get_learners() -> dict[str, dict]:
    """Return a dict of DML learner configs, each with fresh cloned instances.

    Keys: 'lasso', 'random_forest', 'xgboost'
    Each value: {'ml_l': <regressor>, 'ml_m': <classifier>,
                 'ml_g': <regressor>, 'note': <str>}
    """
    return {
        "lasso": {
            "ml_l": clone(_LASSO_L),
            "ml_m": clone(_LOGISTIC_M),
            "ml_g": clone(_LASSO_L),
            "note": "Linear baseline — shows what DML corrects vs OLS",
        },
        "random_forest": {
            "ml_l": clone(_RF_L),
            "ml_m": clone(_RF_M),
            "ml_g": clone(_RF_L),
            "note": "Primary non-linear learner",
        },
        "xgboost": {
            "ml_l": clone(_XGB_L),
            "ml_m": clone(_XGB_M),
            "ml_g": clone(_XGB_L),
            "note": "Gradient boosting — secondary non-linear learner",
        },
    }


def get_naive_ols() -> str:
    """Return a description of the naive OLS specification used as a benchmark.

    Not a fitted model — consumed by 04_dml_specs.py to document the comparison.
    Fit via statsmodels: smf.ols('Y ~ D + X1 + X2 + ...', data=df).fit()
    """
    return (
        "OLS: Y ~ D + X  (statsmodels OLS, no cross-fitting, no regularisation)\n"
        "Coefficient on D is the naive estimate — conflates causal effect with\n"
        "selection bias. Included as a benchmark, not as a causal estimate."
    )


# ── Smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError

    configs = get_learners()

    for name, cfg in configs.items():
        ml_l_cls = type(cfg["ml_l"]).__name__
        ml_m_cls = type(cfg["ml_m"]).__name__
        ml_g_cls = type(cfg["ml_g"]).__name__

        # verify all three are unfitted
        for role, est in (("ml_l", cfg["ml_l"]), ("ml_m", cfg["ml_m"]), ("ml_g", cfg["ml_g"])):
            try:
                check_is_fitted(est)
                status = "FITTED (unexpected!)"
            except NotFittedError:
                status = "unfitted ✓"
            print(f"  [{name}] {role}: {type(est).__name__:<35} {status}")

        print(f"  [{name}] note: {cfg['note']}\n")

    print(get_naive_ols())
    print("\n03_ml_methods.py — OK")
