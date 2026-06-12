"""Phase 3 — Base learner configurations for DML nuisance models."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from xgboost import XGBRegressor, XGBClassifier

from src.config import SEED

# ── Regression learners (for continuous Y or D) ───────────────────────────
def rf_regressor(**kw) -> RandomForestRegressor:
    defaults = dict(n_estimators=300, max_depth=5, min_samples_leaf=5,
                    n_jobs=-1, random_state=SEED)
    return RandomForestRegressor(**{**defaults, **kw})


def xgb_regressor(**kw) -> XGBRegressor:
    defaults = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="rmse", random_state=SEED,
                    verbosity=0, n_jobs=-1)
    return XGBRegressor(**{**defaults, **kw})


def lasso_regressor(**kw) -> LassoCV:
    defaults = dict(cv=5, n_jobs=-1, random_state=SEED, max_iter=5000)
    return LassoCV(**{**defaults, **kw})


# ── Classification learners (for binary D) ───────────────────────────────
def rf_classifier(**kw) -> RandomForestClassifier:
    defaults = dict(n_estimators=300, max_depth=5, min_samples_leaf=5,
                    n_jobs=-1, random_state=SEED)
    return RandomForestClassifier(**{**defaults, **kw})


def xgb_classifier(**kw) -> XGBClassifier:
    defaults = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=SEED,
                    verbosity=0, n_jobs=-1)
    return XGBClassifier(**{**defaults, **kw})


def logistic_classifier(**kw) -> LogisticRegressionCV:
    defaults = dict(cv=5, n_jobs=-1, random_state=SEED, max_iter=1000)
    return LogisticRegressionCV(**{**defaults, **kw})


# ── Learner catalogue ─────────────────────────────────────────────────────
REGRESSION_LEARNERS = {
    "rf":    rf_regressor,
    "xgb":   xgb_regressor,
    "lasso": lasso_regressor,
}

CLASSIFICATION_LEARNERS = {
    "rf":      rf_classifier,
    "xgb":     xgb_classifier,
    "logistic": logistic_classifier,
}
