from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")


# ---------- Basic metrics & plot ----------

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_real_vs_pred(y_true, y_pred, title: str, figures_dir: Path, filename: str):
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("Real value (M€)")
    plt.ylabel("Predicted value (M€)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / filename)
    plt.close()


# ---------- Results table ----------

def save_results_table(results_df: pd.DataFrame, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "model_comparison.csv"
    results_df.to_csv(path, index=False)
    return path


# ---------- Error analysis ----------

def analyze_errors(names_test, y_test, y_pred, figures_dir: Path, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    errors_df = pd.DataFrame(
        {
            "short_name": names_test.values,
            "real_value_M": y_test.values,
            "pred_value_M": y_pred,
        }
    )
    errors_df["error_M"] = errors_df["pred_value_M"] - errors_df["real_value_M"]
    errors_df["abs_error_M"] = errors_df["error_M"].abs()

    # Save errors table
    errors_path = results_dir / "prediction_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    # Error distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(errors_df["error_M"], bins=40, kde=True)
    plt.title("Distribution of Errors (Predicted - Real, in M€)")
    plt.xlabel("Error (M€)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_distribution.png")
    plt.close()

    # Error vs real value
    plt.figure(figsize=(8, 5))
    plt.scatter(errors_df["real_value_M"], errors_df["error_M"], alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Real value (M€)")
    plt.ylabel("Error (M€)")
    plt.title("Prediction Error vs Real Value")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_vs_real_value.png")
    plt.close()

    return errors_df


# ---------- Feature importance (tree models) ----------

def plot_feature_importance(best_pipe, numerical_cols, categorical_cols, figures_dir: Path):
    """
    Plot top-20 feature importances for tree-based models.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    tree_model = None
    model_name = type(best_pipe.named_steps["model"]).__name__
    if model_name in ["RandomForestRegressor", "GradientBoostingRegressor"]:
        tree_model = best_pipe.named_steps["model"]

    if tree_model is None:
        print("No tree-based model, skipping feature importance.")
        return

    # Get feature names after preprocessing
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    num_features = numerical_cols
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = num_features + cat_feature_names

    importances = tree_model.feature_importances_
    feat_imp = pd.DataFrame(
        {"feature": all_feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(20), x="importance", y="feature")
    plt.title("Top 20 Most Important Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_top20.png")
    plt.close()

    # Save table as well
    feat_imp.to_csv(figures_dir / "feature_importances.csv", index=False)
