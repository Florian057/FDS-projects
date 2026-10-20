

from __future__ import annotations

from dataclasses import dataclass, field
from pyexpat import model
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

TARGET = "excess_return"
FEATURE_PREFIX = "z"
MODEL_NAMES = ["OLS", "Huber", "ElasticNet", "PCR", "PLS", "RandomForest", "BoostedTrees"]
ELASTIC_ALPHAS = [0.01, 0.1, 1.0, 10.0]
ELASTIC_L1_RATIOS = [0.1, 0.5, 0.9]
MAX_COMPONENTS = 15
RANDOM_STATE = 42


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class MLResult:
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    oos_r2: Dict[str, float] = field(default_factory=dict)
    dm_matrix: pd.DataFrame = None
    best_elastic: Tuple[float, float] = (float("nan"), float("nan"))
    best_pcr_components: int = 0
    best_pls_components: int = 0


# ---- TASK 2: train/val/test split  (see README Task 2) ---------------------
def make_splits(X: np.ndarray, y: np.ndarray, test_size: float = 0.3,
                val_size: float = 0.2857, random_state: int = RANDOM_STATE) -> Splits:
    n = X.shape[0]
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_test = int(np.ceil(n * test_size))
    test_idx = perm[:n_test]
    trainval_idx = perm[n_test:]

    m = trainval_idx.shape[0]
    rng2 = np.random.RandomState(random_state)
    perm2 = rng2.permutation(m)
    n_val = int(np.ceil(m * val_size))
    val_idx = trainval_idx[perm2[:n_val]]
    train_idx = trainval_idx[perm2[n_val:]]

    return Splits(
        X_train=X[train_idx], X_val=X[val_idx], X_test=X[test_idx],
        y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
    )


# ---- OLS + Huber baselines --------------------
def fit_linear_baselines(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    # TODO: implement -- return {"OLS": ols_model, "Huber": huber_model}
    ols_model = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    huber_model= sklearn.linear_model.HuberRegressor(max_iter=200).fit(X_train, y_train)

    return {"OLS": ols_model, "Huber": huber_model}

# ---- ElasticNet grid search on the validation set --------
def fit_elasticnet_cv(X_train, y_train, X_val, y_val,
                      alphas: List[float] = ELASTIC_ALPHAS,
                      l1_ratios: List[float] = ELASTIC_L1_RATIOS):
    best_mse = np.inf
    best_alpha = alphas[0]
    best_l1_ratio = l1_ratios[0]
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            candidate = sklearn.linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                max_iter=1000, random_state=RANDOM_STATE,
            ).fit(X_train, y_train)
            mse = mean_squared_error(y_val, candidate.predict(X_val))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    final_model = sklearn.linear_model.ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True,
        max_iter=1000, random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    print(f"best alpha, {best_alpha}")
    print(f"best l1, {best_l1_ratio}")

    return (final_model, best_alpha, best_l1_ratio)
    


# ---- PCR + PLS with validation-set n_components search ---
def fit_pcr_pls_cv(X_train, y_train, X_val, y_val,
                   max_components: int = MAX_COMPONENTS):
    k_max = min(X_train.shape[1], max_components)

    best_pcr_k = 1
    best_pcr_mse = np.inf
    for k in range(1, k_max + 1):
        pca = sklearn.decomposition.PCA(n_components=k).fit(X_train)
        Xt_tr = pca.transform(X_train)
        Xt_val = pca.transform(X_val)
        lr = sklearn.linear_model.LinearRegression().fit(Xt_tr, y_train)
        mse = mean_squared_error(y_val, lr.predict(Xt_val))
        if mse < best_pcr_mse:
            best_pcr_mse = mse
            best_pcr_k = k

    pca = sklearn.decomposition.PCA(n_components=best_pcr_k).fit(X_train)
    lr = sklearn.linear_model.LinearRegression().fit(pca.transform(X_train), y_train)

    best_pls_k = 1
    best_pls_mse = np.inf
    for k in range(1, k_max + 1):
        pls_candidate = PLSRegression(n_components=k).fit(X_train, y_train)
        mse = mean_squared_error(y_val, pls_candidate.predict(X_val).flatten())
        if mse < best_pls_mse:
            best_pls_mse = mse
            best_pls_k = k

    pls = PLSRegression(n_components=best_pls_k).fit(X_train, y_train)

    print(f"best pcr comp, {best_pcr_k}")
    print(f"best pls comp, {best_pls_k}")

    return ({"PCR": (pca, lr), "PLS": pls}, best_pcr_k, best_pls_k)


# ---- tree ensembles ---------------------------
def fit_tree_ensembles(X_train, y_train, random_state: int = RANDOM_STATE) -> Dict[str, object]:
    rf = sklearn.ensemble.RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_split=5,
        random_state=random_state).fit(X_train, y_train)

    brt = sklearn.ensemble.GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1,
        max_depth=5, random_state=random_state).fit(X_train, y_train)

    return {"RandomForest": rf, "BoostedTrees": brt}

# ---- out-of-sample R^2 -----------------------
def compute_oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ---- Newey-West standard error --------------------
def newey_west_se(d: np.ndarray, lag: int = 1) -> float:
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    dbar = d.mean()
    e = d - dbar

    gamma0 = np.dot(e, e) / n
    sigma2 = gamma0
    for i in range(1, lag + 1):
        gamma_i = np.dot(e[:n - i], e[i:]) / n
        sigma2 += 2.0 * (1.0 - i / (lag + 1)) * gamma_i

    return float(np.sqrt(sigma2 / n))


# ---- Diebold-Mariano matrix ------------------
def compute_dm_matrix(errors: Dict[str, np.ndarray], lag: int = 1) -> pd.DataFrame:
    names = list(errors.keys())
    dm = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            if i == j:
                continue
            d = errors[i] ** 2 - errors[j] ** 2
            se = newey_west_se(d, lag=lag)
            dm.loc[i, j] = float(np.mean(d)) / se if se > 0 else np.nan
    return dm


# ---- Orchestrator ------------------------------------------
def run_ml_pipeline(X: np.ndarray, y: np.ndarray) -> MLResult:
    splits = make_splits(X, y)

    linear = fit_linear_baselines(splits.X_train, splits.y_train)
    enet, best_alpha, best_l1 = fit_elasticnet_cv(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    dim_models, pcr_k, pls_k = fit_pcr_pls_cv(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    trees = fit_tree_ensembles(splits.X_train, splits.y_train)

    pca_pcr, lr_pcr = dim_models["PCR"]
    pls_model = dim_models["PLS"]

    preds = {
        "OLS": linear["OLS"].predict(splits.X_test),
        "Huber": linear["Huber"].predict(splits.X_test),
        "ElasticNet": enet.predict(splits.X_test),
        "PCR": lr_pcr.predict(pca_pcr.transform(splits.X_test)),
        "PLS": pls_model.predict(splits.X_test).flatten(),
        "RandomForest": trees["RandomForest"].predict(splits.X_test),
        "BoostedTrees": trees["BoostedTrees"].predict(splits.X_test),
    }

    r2 = {name: compute_oos_r2(splits.y_test, p) for name, p in preds.items()}
    errors = {name: splits.y_test - p for name, p in preds.items()}
    dm = compute_dm_matrix(errors, lag=1)

    return MLResult(
        predictions=preds,
        oos_r2=r2,
        dm_matrix=dm,
        best_elastic=(best_alpha, best_l1),
        best_pcr_components=pcr_k,
        best_pls_components=pls_k,
    )
