import matplotlib.pyplot as plt
import seaborn as sns

def run_basic_eda(df):
    """
    Produce basic EDA plots (distributions, scatterplots, heatmap).
    """
    sns.set(style="whitegrid")

    numeric_for_eda = [
        "age", "overall", "potential", "growth_potential",
        "value_million", "offensive_index", "defensive_index"
    ]
    numeric_for_eda = [c for c in numeric_for_eda if c in df.columns]

    # Histograms
    for col in numeric_for_eda:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # Scatterplots
    if "overall" in df.columns and "value_million" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="overall", y="value_million", alpha=0.3)
        plt.title("Value (M€) vs Overall")
        plt.xlabel("Overall")
        plt.ylabel("Value (M€)")
        plt.show()

    if "potential" in df.columns and "value_million" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="potential", y="value_million", alpha=0.3, color="green")
        plt.title("Value (M€) vs Potential")
        plt.xlabel("Potential")
        plt.ylabel("Value (M€)")
        plt.show()

    # Boxplot by position (top 12)
    if "main_position" in df.columns and "value_million" in df.columns:
        top_positions = df["main_position"].value_counts().head(12).index
        df_top_pos = df[df["main_position"].isin(top_positions)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top_pos, x="main_position", y="value_million")
        plt.xticks(rotation=45)
        plt.title("Value (M€) by Main Position (Top 12)")
        plt.xlabel("Main position")
        plt.ylabel("Value (M€)")
        plt.show()

    # Correlation heatmap
    corr_cols = [c for c in [
        "age", "overall", "potential", "growth_potential",
        "offensive_index", "defensive_index", "value_million"
    ] if c in df.columns]

    if len(corr_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation matrix")
        plt.show()
