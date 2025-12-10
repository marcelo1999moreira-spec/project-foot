import pandas as pd
import joblib

from src.data_loader import load_raw_data
from src.features import build_base_features, build_model_dataset
from src.eda import run_basic_eda
from src.modeling import get_models
from src.training import build_preprocessor, train_and_evaluate_models, tune_random_forest_if_best
from src.interpretation import analyze_errors, plot_feature_importance
from src.prediction import predict_player_value, compare_two_players

from sklearn.model_selection import train_test_split

def main():
    # 1. Load raw data
    df_raw = load_raw_data()

    # 2. Feature engineering
    df_fe = build_base_features(df_raw)

    # (Optional) EDA – usually in a notebook, but can be called here
    # run_basic_eda(df_fe)

    # 3. Build modeling dataset
    df_model, X, y, player_names, numerical_cols, categorical_cols = build_model_dataset(df_fe)

    # 4. Train/test split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, player_names,
        test_size=0.2,
        random_state=42
    )

    # 5. Preprocessor and models
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    models = get_models()

    # 6. Train and evaluate all models
    best_pipe, best_model_name, best_y_pred, results_df = train_and_evaluate_models(
        models, preprocessor,
        X_train, y_train,
        X_test, y_test,
        names_test
    )

    # 7. Optional RandomForest tuning
    tuned_pipe, tuned_y_pred = tune_random_forest_if_best(
        best_model_name,
        preprocessor,
        X_train, y_train,
        X_test, y_test
    )

    if tuned_pipe is not None:
        best_pipe = tuned_pipe
        best_y_pred = tuned_y_pred
        best_model_name = "Random Forest (tuned)"

    # 8. Error analysis & feature importance
    errors_df = analyze_errors(names_test, y_test, best_y_pred)
    feat_imp = plot_feature_importance(best_model_name, best_pipe, numerical_cols, categorical_cols)

    # 9. Save final model
    joblib.dump(best_pipe, "results/models/player_value_model.pkl")
    print("\n✅ Final model saved to 'results/models/player_value_model.pkl'")

    # 10. Example predictions
    print("\n=== Example predictions ===")
    example_names = df_model["short_name"].sample(3, random_state=1).tolist()
    for name in example_names:
        predict_player_value(name, df_model, best_pipe, numerical_cols, categorical_cols)

    if len(df_model) >= 2:
        name1, name2 = df_model["short_name"].iloc[0], df_model["short_name"].iloc[1]
        compare_two_players(name1, name2, df_model, best_pipe, numerical_cols, categorical_cols)

if __name__ == "__main__":
    main()




