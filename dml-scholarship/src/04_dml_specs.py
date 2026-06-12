"""Phase 4 — DoubleML model assembly."""
from __future__ import annotations

import doubleml as dml
import pandas as pd

from src.config import OUTCOME, TREATMENT, CONTROLS, N_FOLDS, N_REP
from src.ml_methods import REGRESSION_LEARNERS, CLASSIFICATION_LEARNERS


def make_data_backend(df: pd.DataFrame) -> dml.DoubleMLData:
    x_cols = CONTROLS if CONTROLS else [
        c for c in df.columns if c not in (OUTCOME, TREATMENT)
    ]
    return dml.DoubleMLData(df, y_col=OUTCOME, d_cols=TREATMENT, x_cols=x_cols)


def make_plr(
    data: dml.DoubleMLData,
    ml_l: str = "rf",
    ml_m: str = "rf",
) -> dml.DoubleMLPLR:
    return dml.DoubleMLPLR(
        data,
        ml_l=REGRESSION_LEARNERS[ml_l](),
        ml_m=REGRESSION_LEARNERS[ml_m](),
        n_folds=N_FOLDS,
        n_rep=N_REP,
        draw_sample_splitting=True,
    )


def make_irm(
    data: dml.DoubleMLData,
    ml_g: str = "rf",
    ml_m: str = "rf",
    binary_treatment: bool = True,
) -> dml.DoubleMLIRM:
    m = (CLASSIFICATION_LEARNERS[ml_m]()
         if binary_treatment else REGRESSION_LEARNERS[ml_m]())
    return dml.DoubleMLIRM(
        data,
        ml_g=REGRESSION_LEARNERS[ml_g](),
        ml_m=m,
        n_folds=N_FOLDS,
        n_rep=N_REP,
        draw_sample_splitting=True,
    )


SPECS: dict[str, callable] = {
    "plr_rf":    lambda d: make_plr(d, "rf", "rf"),
    "plr_xgb":   lambda d: make_plr(d, "xgb", "xgb"),
    "plr_lasso": lambda d: make_plr(d, "lasso", "lasso"),
    "irm_rf":    lambda d: make_irm(d, "rf", "rf"),
}
