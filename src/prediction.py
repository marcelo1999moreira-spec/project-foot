import random
import pandas as pd


def get_player_row(player_name: str, df_source: pd.DataFrame):
    """
    Return the row corresponding to a player by short_name, or None.
    """
    mask = df_source["short_name"].str.lower() == player_name.lower()
    if mask.sum() == 0:
        return None
    return df_source[mask].iloc[0]


def predict_player_value(player_name, df_source, pipeline, num_cols, cat_cols):
    """
    Predict a player's market value in M€ from their name.
    """
    row = get_player_row(player_name, df_source)
    if row is None:
        print(f"❌ Player '{player_name}' not found in the dataset.")
        return

    data = pd.DataFrame(
        [row[num_cols + cat_cols].values],
        columns=num_cols + cat_cols,
    )

    pred = pipeline.predict(data)[0]

    print(f"\n🎯 Prediction for {player_name}:")
    print(f"Predicted value: {pred:.2f} M€")

    if "value_million" in row:
        real = row["value_million"]
        print(f"Real value     : {real:.2f} M€")
        print(f"Error          : {pred - real:.2f} M€")


def compare_two_players(name1, name2, df_source, pipeline, num_cols, cat_cols):
    """
    Compare two players using the trained model.
    """
    row1 = get_player_row(name1, df_source)
    row2 = get_player_row(name2, df_source)

    if row1 is None:
        print(f"❌ Player '{name1}' not found.")
        return
    if row2 is None:
        print(f"❌ Player '{name2}' not found.")
        return

    data = pd.DataFrame(
        [row1[num_cols + cat_cols].values, row2[num_cols + cat_cols].values],
        columns=num_cols + cat_cols,
    )

    preds = pipeline.predict(data)

    for name, row, pred in zip([name1, name2], [row1, row2], preds):
        print(f"\n{name}:")
        print(f" - Position : {row['main_position']}")
        print(f" - Age      : {row['age']}")
        print(f" - Overall  : {row['overall']}")
        print(f" - Potential: {row['potential']}")
        if "offensive_index" in row:
            print(f" - Offensive index: {row['offensive_index']:.2f}")
        if "defensive_index" in row:
            print(f" - Defensive index: {row['defensive_index']:.2f}")
        print(f" - Predicted value: {pred:.2f} M€")
        if "value_million" in row:
            print(f" - Real value     : {row['value_million']:.2f} M€")

    if preds[0] > preds[1]:
        print(f"\n👉 According to the model, {name1} has a higher value.")
    elif preds[1] > preds[0]:
        print(f"\n👉 According to the model, {name2} has a higher value.")
    else:
        print("\n👉 According to the model, both players have the same predicted value.")


def demo_predictions(df_model, best_pipe, numerical_cols, categorical_cols):
    """
    Sample a few random players and show predictions + one comparison.
    """
    if len(df_model) == 0:
        print("No data in df_model, skipping demo predictions.")
        return

    example_names = df_model["short_name"].sample(3, random_state=1).tolist()
    print("\nDemo players:", example_names)

    for name in example_names:
        predict_player_value(name, df_model, best_pipe, numerical_cols, categorical_cols)

    if len(df_model) >= 2:
        name1 = df_model["short_name"].iloc[0]
        name2 = df_model["short_name"].iloc[1]
        print("\nComparison demo:")
        compare_two_players(name1, name2, df_model, best_pipe, numerical_cols, categorical_cols)
