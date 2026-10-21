

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from models.post_lasso_regression import CATEGORIES, NEWS_PREFIX


# ---- horizontal bar of top-N news by |t-stat| 
def plot_top_news_by_tstat(
    coefficients: pd.DataFrame,
    n: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Top news dummies by |t-stat| (post-LASSO OLS)",
) -> plt.Figure:
    
    news = coefficients[coefficients["variable"].str.startswith(NEWS_PREFIX)]
    top = news.sort_values("abs_t", ascending=False).head(n)
    top = top.iloc[::-1]

    labels = top["variable"].str[len(NEWS_PREFIX):].tolist()
    tstats = top["t_stat"].to_numpy()
    colors = ["tab:blue" if t >= 0 else "tab:red" for t in tstats]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels) + 1)))
    ax.barh(labels, tstats, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t-stat")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ---- vertical bar of omega_GKW per category  
def plot_omega_gkw_by_category(
    omega_by_category: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "omega_GKW by news category",
) -> plt.Figure:
    
    values = [float(omega_by_category.get(cat, 0.0)) for cat in CATEGORIES]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(CATEGORIES, values, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("omega_GKW")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            ax.annotate(f"{v:.4f}", (x, v), ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
        else:
            ax.annotate(f"{v:.4f}", (x, v), ha="center", va="top",
                        xytext=(0, -3), textcoords="offset points")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
