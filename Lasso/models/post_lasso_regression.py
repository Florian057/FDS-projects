"""Post-LASSO regression + omega_GKW."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


NEWS_PREFIX = "news_"
CATEGORIES = ["Macro Release", "Central Bank", "Auction", "Ad Hoc"]


@dataclass
class PostLassoResult:
    selected_news: List[str]
    ols_results: sm.regression.linear_model.RegressionResultsWrapper
    coefficients: pd.DataFrame
    omega_gkw: float
    omega_gkw_by_category: Dict[str, float] = field(default_factory=dict)


def infer_category(var: str) -> str:
    """news_<category>_... -> category. Returns 'Other' if no match."""
    if not var.startswith(NEWS_PREFIX):
        return "Other"
    tail = var[len(NEWS_PREFIX):]
    for cat in CATEGORIES:
        if tail.startswith(cat + "_") or tail == cat:
            return cat
    return "Other"


def extract_coefficients(results) -> pd.DataFrame:
    """OLS results -> DataFrame with variable, coef, std_err, t_stat, p_value, abs_t."""
    coefs = pd.DataFrame({
        "variable": results.params.index,
        "coef": results.params.values,
        "std_err": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values,
    })
    coefs["abs_t"] = coefs["t_stat"].abs()
    return coefs


# ---- K-fold LASSO on news_* columns  ----------
def select_news_via_lasso(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> List[str]:
    
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    news_cols = X.columns[X.columns.str.startswith(NEWS_PREFIX)].tolist()
    X_news = X[news_cols]
    scaler = StandardScaler()
    X_news_std = scaler.fit_transform(X_news)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    lasso = LassoCV(cv=cv, max_iter=10000)
    lasso.fit(X_news_std, y)

    selected = [col for col, coef in zip(news_cols, lasso.coef_) if abs(coef) > 1e-10]

    print(f"lasso.alpha_ (CV-picked penalty): {lasso.alpha_}")
    print(f"lasso.alphas_[0] (largest penalty in grid): {lasso.alphas_[0]}")
    print(f"Number of selected news variables: {len(selected)}")
    assert lasso.alpha_ < lasso.alphas_[0], (
        "CV picked the 'nothing selected' corner of the alpha grid."
    )

    return selected


# ----  post-LASSO OLS with HC1 SEs   -------------
def fit_post_lasso_ols(X, y, selected_news: List[str], cov_type: str = "HC1"):
    
    non_news = [c for c in X.columns if not c.startswith(NEWS_PREFIX)]
    cols = non_news + list(selected_news)
    X_sub = X[cols].astype(float)

    return sm.OLS(y, X_sub).fit(cov_type=cov_type)


# ----  overall omega_GKW  -----------------------
def compute_omega_gkw(X: pd.DataFrame, y: pd.Series, coefficients: pd.DataFrame) -> float:
    # TODO: implement -- return float in [0, 1]
    news_cols = [v for v in coefficients["variable"] if v.startswith(NEWS_PREFIX) and v in X.columns]
    if not news_cols:
        return 0.0
    active = X[news_cols].to_numpy().any(axis=1)
    y_arr = np.asarray(y)
    total = y_arr.sum()
    if total == 0:
        return 0.0
    return float(y_arr[active].sum() / total)


# ---- per-category omega_GKW, winner-takes-all  ----
def compute_omega_gkw_by_category(X, y, coefficients) -> Dict[str, float]:
   
    shares = {cat: 0.0 for cat in CATEGORIES}

    news = coefficients[coefficients["variable"].str.startswith(NEWS_PREFIX)]
    news = news[news["variable"].isin(X.columns)]
    news = news.sort_values("abs_t", ascending=False)
    ranked_vars = news["variable"].tolist()
    if not ranked_vars:
        return shares

    y_arr = np.asarray(y)
    total = y_arr.sum()
    if total == 0:
        return shares

    active = X[ranked_vars].to_numpy().astype(bool)
    winner_idx = np.where(active.any(axis=1), active.argmax(axis=1), -1)

    for i, var in enumerate(ranked_vars):
        cat = infer_category(var)
        if cat not in shares:
            continue
        mask = winner_idx == i
        shares[cat] += float(y_arr[mask].sum() / total)

    return shares


# --- Orchestrator  -------------------------------------------
def run_post_lasso(X, y, cv_folds: int = 5, cov_type: str = "HC1") -> PostLassoResult:
    selected = select_news_via_lasso(X, y, cv_folds=cv_folds)
    ols = fit_post_lasso_ols(X, y, selected_news=selected, cov_type=cov_type)
    coefs = extract_coefficients(ols)
    return PostLassoResult(
        selected_news=selected,
        ols_results=ols,
        coefficients=coefs,
        omega_gkw=compute_omega_gkw(X, y, coefs),
        omega_gkw_by_category=compute_omega_gkw_by_category(X, y, coefs),
    )
