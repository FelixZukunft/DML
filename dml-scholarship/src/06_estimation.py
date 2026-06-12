"""Phase 6 — Fit DML models and persist them."""
from __future__ import annotations

import joblib
import doubleml as dml
from pathlib import Path

MODELS_DIR = Path(__file__).parents[1] / "outputs" / "models"


def fit_all(models: dict[str, dml.DoubleML]) -> dict[str, dml.DoubleML]:
    fitted = {}
    for name, model in models.items():
        print(f"[06] Fitting {name} …")
        model.fit()
        fitted[name] = model
        print(f"     θ̂ = {model.coef[0]:.4f}")
    return fitted


def save_models(fitted: dict[str, dml.DoubleML]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"[06] Saved {path}")


def load_model(name: str) -> dml.DoubleML:
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def run(models: dict[str, dml.DoubleML]) -> dict[str, dml.DoubleML]:
    fitted = fit_all(models)
    save_models(fitted)
    return fitted
