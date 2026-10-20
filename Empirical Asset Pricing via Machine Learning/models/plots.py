
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


# ---- bar chart of OOS R^2   --------------------
def plot_oos_r2_bar(
    oos_r2: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "Out-of-sample R^2 by model",
) -> plt.Figure:
    order = ["OLS", "Huber", "ElasticNet", "PCR", "PLS", "RandomForest", "BoostedTrees"]
    names = [n for n in order if n in oos_r2]
    values = [oos_r2[n] for n in names]
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Out-of-sample R^2")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, v in zip(bars, values):
        offset = 0.01 * (max(values) - min(values) + 1e-9)
        ax.annotate(
            f"{v:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3 if v >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ---- heatmap of the Diebold-Mariano matrix ---
def plot_dm_heatmap(
    dm_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Diebold-Mariano statistics (row vs column)",
) -> plt.Figure:
    import numpy as np

    names = list(dm_matrix.index)
    data = dm_matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(np.nanmax(np.abs(data))) else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
