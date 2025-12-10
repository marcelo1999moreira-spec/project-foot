from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

from src.evaluation import compute_metrics, plot_real_vs_pred


def train_and_select_model(
    X,
    y,
    player_names,
    preprocessor,
    models: Dict[str, object],
    figures_dir: Path,
):
    """
    Train all candidate models, perform CV, evaluate on test set,
    plot real vs predicted, and return the best model.
    """
    # Train/test split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, player_names, test_size=0.2, random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    best_model_name = None
    best_rmse = np.inf
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        print(f"\n===== Model: {name} =====")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Cross-validation on training set
        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        print(f"CV RMSE (mean): {cv_rmse_mean:.4f} M€ (+/- {cv_rmse_std:.4f})")

        # Fit and evaluate on test set
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)

        mae, rmse, r2 = compute_metrics(y_test, y_pred_test)
        print(f"Test MAE  : {mae:.4f} M€")
        print(f"Test RMSE : {rmse:.4f} M€")
        print(f"Test R²   : {r2:.4f}")

        results.append(
            {
                "model": name,
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
            }
        )

        # Real vs predicted plot
        filename = f"real_vs_pred_{name.replace(' ', '_').lower()}.png"
        plot_real_vs_pred(y_test, y_pred_test, f"{name}: Real vs Predicted", figures_dir, filename)

        # Keep best
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipe = pipe
            best_y_pred = y_pred_test

    results_df = pd.DataFrame(results).sort_values("test_rmse")
    print("\n=== Comparative results ===")
    print(results_df)

    print(f"\n>>> Best base model (RMSE): {best_model_name} with RMSE = {best_rmse:.4f} M€")

    return (
        best_pipe,
        best_model_name,
        best_y_pred,
        best_rmse,
        results_df,
        X_train,
        X_test,
        y_train,
        y_test,
        names_test,
    )


def tune_best_random_forest(
    best_pipe,
    best_model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    figures_dir: Path,
):
    """
    If the best base model is Random Forest, run a small GridSearchCV.
    """
    if best_model_name != "Random Forest":
        print("Best base model is not Random Forest -> skipping tuning.")
        return best_pipe, best_model_name, best_pipe.predict(X_test), None

    print("\nRunning GridSearchCV on Random Forest...")

    preprocessor = best_pipe.named_steps["preprocessor"]

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [15, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    grid_search = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest RandomForest parameters:")
    print(grid_search.best_params_)

    tuned_pipe = grid_search.best_estimator_
    tuned_y_pred = tuned_pipe.predict(X_test)

    mae, tuned_rmse, tuned_r2 = compute_metrics(y_test, tuned_y_pred)
    print("\nPerformance after tuning:")
    print(f"MAE  : {mae:.4f} M€")
    print(f"RMSE : {tuned_rmse:.4f} M€")
    print(f"R²   : {tuned_r2:.4f}")

    plot_real_vs_pred(
        y_test,
        tuned_y_pred,
        "Random Forest (tuned): Real vs Predicted",
        figures_dir,
        "real_vs_pred_random_forest_tuned.png",
    )

    print(f"\n>>> Final selected model: Random Forest (tuned), RMSE = {tuned_rmse:.4f} M€")

    return tuned_pipe, "Random Forest (tuned)", tuned_y_pred, tuned_rmse


def save_model(best_pipe, models_dir: Path, model_name: str = "player_value_model.pkl") -> Path:
    """
    Save the final trained pipeline to results/models.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name
    joblib.dump(best_pipe, model_path)
    return model_path
