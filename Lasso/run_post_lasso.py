

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from models.plots import plot_omega_gkw_by_category, plot_top_news_by_tstat
from models.post_lasso_regression import run_post_lasso


DATA_PATH = Path(__file__).parent / "data" / "ready_data.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"
TARGET = "y_diff_squared"
MATURITY_LABEL = "10-day"


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_PATH).set_index("loctimestamp")
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y = load_xy()
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    print(f"  # news_* columns: {sum(c.startswith('news_') for c in X.columns)}")
    print(f"  # controls (non-news): {sum(not c.startswith('news_') for c in X.columns)}")

    result = run_post_lasso(X, y, cv_folds=5)

    print()
    print("=" * 60)
    print(f"LASSO (5-fold CV) selected {len(result.selected_news)} news variables")
    print("=" * 60)

    print()
    print("Post-LASSO OLS coefficients:")
    print(result.coefficients.to_string(index=False))
    result.coefficients.to_csv(OUTPUT_DIR / "ols_coefficients.csv", index=False)

    print()
    print(f"omega_GKW (overall) = {result.omega_gkw:.6f}")
    print("omega_GKW by category:")
    for cat, val in result.omega_gkw_by_category.items():
        print(f"  {cat:16s} {val:.6f}")

    pd.DataFrame(
        {"category": list(result.omega_gkw_by_category.keys()),
         "omega_gkw": list(result.omega_gkw_by_category.values())}
    ).to_csv(OUTPUT_DIR / "omega_by_category.csv", index=False)

    plot_top_news_by_tstat(
        result.coefficients,
        n=10,
        save_path=OUTPUT_DIR / "top10_news_tstat.png",
        title=f"Top 10 news dummies by |t-stat|  (maturity {MATURITY_LABEL})",
    )
    plot_omega_gkw_by_category(
        result.omega_gkw_by_category,
        save_path=OUTPUT_DIR / "omega_by_category.png",
        title=f"omega_GKW by news category  (maturity {MATURITY_LABEL})",
    )
    plt.show()
    print(f"\nSaved plots + CSVs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
