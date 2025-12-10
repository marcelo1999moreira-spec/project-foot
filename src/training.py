import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from .evaluation import compute_metrics, plot_real_vs_pred

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Build the ColumnTransformer for scaling numeric features
    and one-hot encoding categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_models(models, preprocessor, X_train, y_train, X_test, y_test, names_test):
    """
    Train and evaluate all models with cross-validation.
    Returns:
      best_pipeline, best_model_name, best_y_pred, results_df
    """
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_rmse = np.inf
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        print(f"\n===== Model: {name} =====")

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        print(f"CV RMSE (mean): {cv_rmse_mean:.4f} M€ (+/- {cv_rmse_std:.4f})")

        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)

        mae, rmse, r2 = compute_metrics(y_test, y_pred_test)
        print(f"Test MAE  : {mae:.4f} M€")
        print(f"Test RMSE : {rmse:.4f} M€")
        print(f"Test R²   : {r2:.4f}")

        results.append({
            "model": name,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipe = pipe
            best_y_pred = y_pred_test

        plot_real_vs_pred(y_test, y_pred_test, title=f"{name}: Real vs Predicted")

    results_df = pd.DataFrame(results)
    print("\n=== Comparative results ===")
    display(results_df.sort_values("test_rmse"))

    print(f"\n>>> Best model (RMSE): {best_model_name} with RMSE = {best_rmse:.4f} M€")

    return best_pipe, best_model_name, best_y_pred, results_df

def tune_random_forest_if_best(best_model_name, preprocessor, X_train, y_train, X_test, y_test):
    """
    If the best model is a Random Forest, perform a light GridSearch.
    Returns the tuned pipeline and predictions.
    """
    tuned_pipe = None
    tuned_y_pred = None

    if best_model_name != "Random Forest":
        return None, None

    print("\nRunning GridSearchCV on Random Forest...")

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [15, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    grid_search = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest RandomForest parameters:")
    print(grid_search.best_params_)

    tuned_pipe = grid_search.best_estimator_
    tuned_y_pred = tuned_pipe.predict(X_test)

    return tuned_pipe, tuned_y_pred
