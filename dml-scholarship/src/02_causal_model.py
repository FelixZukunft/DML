"""Phase 2 — DAG definition and variable selection."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path

from src.config import OUTCOME, TREATMENT, CONTROLS

FIGURES_DIR = Path(__file__).parents[1] / "outputs" / "figures"


def get_variable_sets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return (Y, D, X) as Series / DataFrame slices."""
    y = df[OUTCOME]
    d = df[TREATMENT]
    x = df[CONTROLS] if CONTROLS else df.drop(columns=[OUTCOME, TREATMENT])
    return y, d, x


# ── DAG ───────────────────────────────────────────────────────────────────
def build_dag() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([
        ("X", "D"),   # controls → treatment
        ("X", "Y"),   # controls → outcome
        ("D", "Y"),   # treatment → outcome  (the causal effect we estimate)
    ])
    return G


def plot_dag(G: nx.DiGraph | None = None, save: bool = True) -> None:
    G = G or build_dag()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pos = {"X": (0, 0), "D": (1, 0), "Y": (2, 0)}
    fig, ax = plt.subplots(figsize=(5, 2))
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=1200,
                     node_color="#4C72B0", font_color="white",
                     arrows=True, arrowsize=20)
    ax.axis("off")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "dag.png", dpi=150)
        print("[02] DAG saved → outputs/figures/dag.png")
    plt.close()


def run(df: pd.DataFrame | None = None) -> None:
    G = build_dag()
    plot_dag(G)
    if df is not None:
        y, d, x = get_variable_sets(df)
        print(f"[02] Y={OUTCOME}, D={TREATMENT}, X cols={list(x.columns)}")


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_pipeline", __file__.replace("02_causal_model.py", "01_data_pipeline.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run(mod.run())
