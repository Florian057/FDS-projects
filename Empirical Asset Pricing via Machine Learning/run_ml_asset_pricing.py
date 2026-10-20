

"""Reads the pre-built panel dataset (data/panel_data.parquet) and runs:

    train/val/test split ->
    linear + regularized + dim-reduction + tree models ->
    out-of-sample R^2 + Diebold-Mariano matrix ->
    plots.

Dataset already contains 15 features `z1`..`z15` (characteristics x macro
factors, flattened via Kronecker product) and the target `excess_return`.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from models.ml_models import TARGET, run_ml_pipeline
from models.plots import plot_dm_heatmap, plot_oos_r2_bar


DATA_PATH = Path(__file__).parent / "data" / "panel_data.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_xy() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("z")]
    y = df[TARGET].to_numpy(dtype=float)
    X = df[feature_cols].to_numpy(dtype=float)
    return X, y, feature_cols


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y, feature_cols = load_xy()
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"  # feature columns: {len(feature_cols)}")
    print(f"  y stats: mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")

    # ---- Task 1 diagnostic: in-sample OLS R^2 on the full panel ----
    ols_full = LinearRegression().fit(X, y)
    in_sample_r2 = r2_score(y, ols_full.predict(X))
    print(f"  in-sample OLS R^2 on the full panel: {in_sample_r2:.4f} "
          f"(target signal R^2 was calibrated to ~0.05)")

    result = run_ml_pipeline(X, y)

    print()
    print("=" * 60)
    print(f"ElasticNet best (alpha, l1_ratio) = {result.best_elastic}")
    print(f"PCR best n_components   = {result.best_pcr_components}")
    print(f"PLS best n_components   = {result.best_pls_components}")
    print("=" * 60)

    print()
    print("Out-of-sample R^2:")
    r2_df = pd.DataFrame(
        {"Model": list(result.oos_r2.keys()),
         "OOS_R2": list(result.oos_r2.values())}
    )
    print(r2_df.to_string(index=False))
    r2_df.to_csv(OUTPUT_DIR / "oos_r2.csv", index=False)

    print()
    print("Diebold-Mariano matrix (row minus column, positive => row worse):")
    print(result.dm_matrix.round(3).to_string())
    result.dm_matrix.to_csv(OUTPUT_DIR / "dm_matrix.csv")

    plot_oos_r2_bar(
        result.oos_r2,
        save_path=OUTPUT_DIR / "oos_r2_bar.png",
        title="Out-of-sample R^2 by model (PS 6)",
    )
    plot_dm_heatmap(
        result.dm_matrix,
        save_path=OUTPUT_DIR / "dm_heatmap.png",
        title="Diebold-Mariano statistics (row vs column)",
    )
    plt.show()
    print(f"\nSaved CSVs + plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
