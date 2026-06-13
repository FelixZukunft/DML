"""Phase 2 — Causal DAG and variable selection for the DML scholarship study.

Draws the identification DAG and saves it to outputs/figures/causal_dag.png.
Variable-set helper (get_variable_sets) is used by downstream phases.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"

# ── Variable specification (actual column names in processed CSVs) ─────────
Y_COL = "Curricular units 2nd sem (grade)"
D_COL = "Scholarship holder"
X_COLS = [
    "Marital Status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification (grade)",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "International", "Age at enrollment", "Admission grade",
    "GDP", "Inflation rate", "Unemployment rate",
]


def get_variable_sets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return (Y, D, X) slices from a processed dataframe.

    X is everything except Y and D — works for both raw and semantic encodings.
    """
    y = df[Y_COL]
    d = df[D_COL]
    x = df[[c for c in df.columns if c not in (Y_COL, D_COL)]]
    return y, d, x


# ── DAG definition (networkx graph for structure; rendered manually below) ─
def build_dag() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(["X\n(Confounders)", "D\n(Scholarship)", "Y\n(2nd Sem Grade)", "U\n(Unobserved)"])
    G.add_edges_from([
        ("X\n(Confounders)",  "D\n(Scholarship)"),
        ("X\n(Confounders)",  "Y\n(2nd Sem Grade)"),
        ("D\n(Scholarship)",  "Y\n(2nd Sem Grade)"),
        ("U\n(Unobserved)",   "D\n(Scholarship)"),
        ("U\n(Unobserved)",   "Y\n(2nd Sem Grade)"),
    ])
    return G


# ── Manual layout (left-to-right; no graphviz dependency needed) ───────────
#   D  ──────────────→  Y
#   ↑                   ↑
#   X (bottom-center)   U (bottom-right, dashed)
_POS = {
    "D": (0.80,  0.42),
    "Y": (2.50,  0.42),
    "X": (1.65, -0.28),
    "U": (2.50, -0.28),
}

_NODE = {
    "D": dict(label="D\n(Scholarship)",   fc="#F4C430", ec="#9e7e00", lw=2.0, r=0.22, dashed=False, tc="#2a2a2a"),
    "Y": dict(label="Y\n(2nd Sem Grade)", fc="#4C72B0", ec="#2a4f8f", lw=2.0, r=0.22, dashed=False, tc="white"),
    "X": dict(label="X\n(Confounders)",   fc="#E07B39", ec="#a04010", lw=2.0, r=0.22, dashed=False, tc="white"),
    "U": dict(label="U\n(Unobserved)",    fc="#CC3333", ec="#881111", lw=1.5, r=0.16, dashed=True,  tc="white"),
}

# (src, dst, color, lw, dashed, label, (label_x_offset, label_y_offset))
_EDGES = [
    ("X", "D", "#888888", 1.5, False, "Selection\nbias",         (-0.32,  0.14)),
    ("X", "Y", "#888888", 1.5, False, "",                         ( 0.00,  0.00)),
    ("D", "Y", "#4C72B0", 3.0, False, "θ (ATE)",                 ( 0.00,  0.14)),
    ("U", "D", "#CC3333", 1.5, True,  "Unobserved\nconfounding", (-0.22,  0.14)),
    ("U", "Y", "#CC3333", 1.5, True,  "",                         ( 0.00,  0.00)),
]


def _draw_arrow(ax, x0, y0, x1, y1, r_src, r_dst, color, lw, dashed):
    """Arrow from the border of src circle to the border of dst circle."""
    dx, dy = x1 - x0, y1 - y0
    dist   = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    sx, sy = x0 + ux * r_src, y0 + uy * r_src
    ex, ey = x1 - ux * r_dst, y1 - uy * r_dst
    ax.annotate(
        "", xy=(ex, ey), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=(0, (5, 4)) if dashed else "solid",
            mutation_scale=14,
        ),
        zorder=3,
    )


def plot_dag(save: bool = True) -> Path | None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.set_xlim(0.20, 3.30)
    ax.set_ylim(-1.00, 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Nodes
    for key, cfg in _NODE.items():
        x, y = _POS[key]
        # filled circle
        ax.add_patch(plt.Circle((x, y), cfg["r"], color=cfg["fc"], zorder=3))
        # border (dashed for U)
        ax.add_patch(plt.Circle(
            (x, y), cfg["r"], fill=False,
            edgecolor=cfg["ec"], lw=cfg["lw"],
            linestyle=(0, (4, 3)) if cfg["dashed"] else "solid",
            zorder=4,
        ))
        ax.text(x, y, cfg["label"], ha="center", va="center",
                fontsize=9, fontweight="bold", color=cfg["tc"],
                zorder=5, multialignment="center")

    # Edges
    for src, dst, color, lw, dashed, label, (lox, loy) in _EDGES:
        x0, y0 = _POS[src]
        x1, y1 = _POS[dst]
        _draw_arrow(ax, x0, y0, x1, y1,
                    _NODE[src]["r"], _NODE[dst]["r"],
                    color, lw, dashed)
        if label:
            mx = (x0 + x1) / 2 + lox
            my = (y0 + y1) / 2 + loy
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=7.5, color=color, zorder=6, multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # Text box 1 — identification assumption
    ax.text(
        0.25, -0.52,
        "Identification:  E[U | D, X] = 0\n"
        "Conditional on X, scholarship assignment\n"
        "is as good as random (selection on observables)",
        ha="left", va="top", fontsize=8.5, color="#2a2a2a",
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBEA", ec="#CCAA44", lw=1.3),
        zorder=6,
    )

    # Text box 2 — DML implementation
    ax.text(
        1.58, -0.52,
        "DML implementation:  DoubleMLPLR (primary)\n"
        "+ DoubleMLIRM (robustness check)\n"
        "Nuisance learners: Lasso, Random Forest, XGBoost",
        ha="left", va="top", fontsize=8.5, color="#2a2a2a",
        bbox=dict(boxstyle="round,pad=0.45", fc="#EEF4FF", ec="#4C72B0", lw=1.3),
        zorder=6,
    )

    fig.suptitle("Causal DAG — DML Scholarship Study",
                 fontsize=13, fontweight="bold", y=0.98)

    if save:
        out = FIGURES_DIR / "causal_dag.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[02] DAG saved → {out}")
        return out
    plt.close()
    return None


def run(df: pd.DataFrame | None = None) -> None:
    path = plot_dag(save=True)
    if df is not None:
        y, d, x = get_variable_sets(df)
        print(f"[02] Y={Y_COL!r}  D={D_COL!r}  X features={x.shape[1]}")


if __name__ == "__main__":
    run()
