"""Phase 8 — Robustness and sensitivity checks."""
from __future__ import annotations

import pandas as pd
import doubleml as dml
from pathlib import Path

TABLES_DIR = Path(__file__).parents[1] / "outputs" / "tables"


def sensitivity_riesz(fitted: dict[str, dml.DoubleML]) -> pd.DataFrame:
    """Sensitivity analysis via DoubleML's built-in sensitivity framework."""
    rows = []
    for name, model in fitted.items():
        if not hasattr(model, "sensitivity_analysis"):
            print(f"[08] {name}: sensitivity_analysis not available, skipping")
            continue
        model.sensitivity_analysis()
        sa = model.sensitivity_params
        rows.append({"spec": name, **sa})
    return pd.DataFrame(rows)


def alternative_specs(
    data: dml.DoubleMLData,
    fitted: dict[str, dml.DoubleML],
) -> pd.DataFrame:
    """Placeholder — add alternative control sets, subsamples, etc."""
    print("[08] Alternative spec checks: implement per research design")
    return pd.DataFrame()


def run(
    fitted: dict[str, dml.DoubleML],
    data: dml.DoubleMLData | None = None,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    riesz = sensitivity_riesz(fitted)
    if not riesz.empty:
        riesz.to_csv(TABLES_DIR / "sensitivity_riesz.csv", index=False)
        print("[08] Sensitivity table saved → outputs/tables/sensitivity_riesz.csv")

    if data is not None:
        alternative_specs(data, fitted)

    print("[08] Sensitivity checks complete")
