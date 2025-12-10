import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_errors(names_test, y_test, y_pred):
    """
    Build an error dataframe and plot error distributions.
    """
    errors_df = pd.DataFrame({
        "short_name": names_test.values,
        "real_value_M": y_test.values,
        "pred_value_M": y_pred,
    })

    errors_df["error_M"] = errors_df["pred_value_M"] - errors_df["real_value_M"]
    errors_df["abs_error_M"] = errors_df["error_M"].abs()

    print("\nSample of prediction errors:")
    display(errors_df.head())

    under_valued = errors_df.sort_values("error_M", ascending=True).head(15)
    over_valued = errors_df.sort_values("error_M", ascending=False).head(15)

    print("\nTop 15 under-valued players:")
    display(under_valued)

    print("\nTop 15 over-valued players:")
    display(over_valued)

    plt.figure(figsize=(8, 4))
    sns.histplot(errors_df["error_M"], bins=40, kde=True)
    plt.title("Distribution of errors (Predicted - Real, in M€)")
    plt.xlabel("Error (M€)")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(errors_df["real_value_M"], errors_df["error_M"], alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Real value (M€)")
    plt.ylabel("Error (M€)")
    plt.title("Prediction error as a function of real value")
    plt.show()

    return errors_df

def plot_feature_importance(best_model_name, best_pipeline, numerical_cols, categorical_cols):
    """
    Plot feature importances if the final model is tree-based.
    """
    tree_model = None

    if "Random Forest" in best_model_name:
        tree_model = best_pipeline.named_steps["model"]
    elif best_model_name == "Gradient Boosting":
        tree_model = best_pipeline.named_steps["model"]

    if tree_model is None:
        print("\nNo tree-based model selected, cannot plot feature importance.")
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"]
    num_features = numerical_cols
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = num_features + cat_feature_names

    importances = tree_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\nTop 20 most important features:")
    display(feat_imp.head(20))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(20), x="importance", y="feature")
    plt.title(f"Top 20 feature importances ({best_model_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return feat_imp
