"""Phase 7 — CIs, p-values, coefficient tables, and coefficient plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import doubleml as dml
from pathlib import Path

TABLES_DIR = Path(__file__).parents[1] / "outputs" / "tables"
FIGURES_DIR = Path(__file__).parents[1] / "outputs" / "figures"


def extract_results(fitted: dict[str, dml.DoubleML]) -> pd.DataFrame:
    rows = []
    for name, model in fitted.items():
        summary = model.summary
        rows.append({
            "spec":   name,
            "coef":   model.coef[0],
            "se":     model.se[0],
            "ci_lo":  model.confint().iloc[0, 0],
            "ci_hi":  model.confint().iloc[0, 1],
            "pval":   model.pval[0],
            "t_stat": model.t_stat[0],
        })
    return pd.DataFrame(rows).sort_values("spec")


def save_tables(results: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "coef_table.csv", index=False)
    results.to_latex(TABLES_DIR / "coef_table.tex", index=False, float_format="%.4f")
    print("[07] Tables saved → outputs/tables/")


def plot_coefs(results: pd.DataFrame, save: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(results) * 0.8)))
    y = range(len(results))
    ax.errorbar(results["coef"], y,
                xerr=[results["coef"] - results["ci_lo"],
                      results["ci_hi"] - results["coef"]],
                fmt="o", color="#4C72B0", capsize=4)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(results["spec"])
    ax.set_xlabel("Estimated treatment effect (95 % CI)")
    ax.set_title("DML coefficient estimates across specifications")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "coef_plot.png", dpi=150)
        print("[07] Coefficient plot saved → outputs/figures/coef_plot.png")
    plt.close()


def run(fitted: dict[str, dml.DoubleML]) -> pd.DataFrame:
    results = extract_results(fitted)
    save_tables(results)
    plot_coefs(results)
    print(results.to_string(index=False))
    return results
