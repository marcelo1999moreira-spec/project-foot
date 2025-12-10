import numpy as np
import pandas as pd

def extract_main_position(pos):
    if isinstance(pos, str):
        return pos.split(",")[0].strip()
    return "Unknown"

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning and feature engineering to the raw dataframe.
    Returns a new dataframe with engineered features.
    """
    # Keep only rows with known market value
    df = df.dropna(subset=["value_eur"]).copy()

    # Main position
    df["main_position"] = df["player_positions"].apply(extract_main_position)

    # Growth potential
    df["growth_potential"] = df["potential"] - df["overall"]

    # Value in millions
    df["value_million"] = df["value_eur"] / 1_000_000

    # BMI
    if "height_cm" in df.columns and "weight_kg" in df.columns:
        df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    else:
        df["bmi"] = np.nan

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[15, 20, 25, 30, 35, 45],
        labels=["<20", "20-25", "25-30", "30-35", "35+"]
    )

    # Offensive / defensive indices
    offensive_stats = [c for c in ["pace", "shooting", "passing", "dribbling"] if c in df.columns]
    defensive_stats = [c for c in ["defending", "physic"] if c in df.columns]

    if offensive_stats:
        df["offensive_index"] = df[offensive_stats].mean(axis=1)
    else:
        df["offensive_index"] = np.nan

    if defensive_stats:
        df["defensive_index"] = df[defensive_stats].mean(axis=1)
    else:
        df["defensive_index"] = np.nan

    return df

def build_model_dataset(df: pd.DataFrame):
    """
    Build the modeling dataset: df_model, X, y, player_names,
    and the lists of numerical and categorical columns.
    """
    base_cols_to_keep = ["short_name", "main_position", "value_million"]

    numerical_cols = [
        "age", "height_cm", "weight_kg",
        "overall", "potential", "growth_potential",
        "pace", "shooting", "passing",
        "dribbling", "defending", "physic",
        "bmi", "offensive_index", "defensive_index"
    ]

    # Keep only existing numerical columns
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    categorical_cols = ["main_position"]

    df_model = df[base_cols_to_keep + numerical_cols].dropna().copy()

    X = df_model[numerical_cols + categorical_cols]
    y = df_model["value_million"]
    player_names = df_model["short_name"]

    print("\nModeling dataframe shape:", df_model.shape)
    print("Numerical columns used:", numerical_cols)
    print("Categorical columns used:", categorical_cols)

    return df_model, X, y, player_names, numerical_cols, categorical_cols
