from pathlib import Path

from src.data_loader import load_raw_data
from src.features import add_basic_features, prepare_model_dataset
from src.eda import run_basic_eda
from src.modeling import get_preprocessor, get_models
from src.training import train_and_select_model, tune_best_random_forest, save_model
from src.evaluation import analyze_errors, plot_feature_importance, save_results_table
from src.prediction import demo_predictions


def main():
    base_dir = Path(__file__).resolve().parent

    # Create results folders
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"
    for d in [results_dir, figures_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading raw data...")
    df = load_raw_data(base_dir)
    print("Raw data shape:", df.shape)

    print("\n[2] Feature engineering...")
    df = add_basic_features(df)
    df_model, numerical_cols, categorical_cols, X, y, player_names = prepare_model_dataset(df)
    print("Modeling dataframe shape:", df_model.shape)

    print("\n[3] Running basic EDA (figures saved in results/figures)...")
    run_basic_eda(df, figures_dir)

    print("\n[4] Preparing models and preprocessor...")
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    models = get_models()

    print("\n[5] Training and evaluating all models...")
    (best_pipe,
     best_model_name,
     best_y_pred,
     best_rmse,
     results_df,
     X_train,
     X_test,
     y_train,
     y_test,
     names_test) = train_and_select_model(
        X, y, player_names, preprocessor, models, figures_dir
    )

    print("\n[6] Hyperparameter tuning (if best base model is Random Forest)...")
    best_pipe, best_model_name, best_y_pred, best_rmse = tune_best_random_forest(
        best_pipe, best_model_name, X_train, y_train, X_test, y_test, figures_dir
    )

    print("\n[7] Saving results table and final model...")
    save_results_table(results_df, results_dir)
    model_path = save_model(best_pipe, models_dir, model_name="player_value_model.pkl")
    print(f"Final model saved to: {model_path}")

    print("\n[8] Error analysis and feature importance...")
    errors_df = analyze_errors(names_test, y_test, best_y_pred, figures_dir, results_dir)
    plot_feature_importance(best_pipe, numerical_cols, categorical_cols, figures_dir)

    print("\n[9] Demo predictions on a few players...")
    demo_predictions(df_model, best_pipe, numerical_cols, categorical_cols)

    print("\n✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()
