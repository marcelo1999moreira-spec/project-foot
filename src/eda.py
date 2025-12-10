from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")


def run_basic_eda(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Run a few exploratory plots and save them to results/figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Distributions
    numeric_for_eda = [
        "age", "overall", "potential", "growth_potential",
        "value_million", "offensive_index", "defensive_index",
    ]
    numeric_for_eda = [c for c in numeric_for_eda if c in df.columns]

    for col in numeric_for_eda:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(figures_dir / f"dist_{col}.png")
        plt.close()

    # 2) Value vs overall
    if {"overall", "value_million"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="overall", y="value_million", alpha=0.3)
        plt.title("Market Value (M€) vs Overall")
        plt.xlabel("Overall")
        plt.ylabel("Value (M€)")
        plt.tight_layout()
        plt.savefig(figures_dir / "value_vs_overall.png")
        plt.close()

    # 3) Value vs potential
    if {"potential", "value_million"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="potential", y="value_million", alpha=0.3)
        plt.title("Market Value (M€) vs Potential")
        plt.xlabel("Potential")
        plt.ylabel("Value (M€)")
        plt.tight_layout()
        plt.savefig(figures_dir / "value_vs_potential.png")
        plt.close()

    # 4) Boxplot by main position
    if {"main_position", "value_million"}.issubset(df.columns):
        top_positions = df["main_position"].value_counts().head(12).index
        df_top_pos = df[df["main_position"].isin(top_positions)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top_pos, x="main_position", y="value_million")
        plt.xticks(rotation=45)
        plt.title("Market Value (M€) by Main Position (Top 12)")
        plt.xlabel("Main position")
        plt.ylabel("Value (M€)")
        plt.tight_layout()
        plt.savefig(figures_dir / "value_by_position.png")
        plt.close()

    # 5) Correlation heatmap
    corr_cols = [
        c for c in [
            "age", "overall", "potential",
            "growth_potential", "offensive_index",
            "defensive_index", "value_million",
        ]
        if c in df.columns
    ]

    if len(corr_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(figures_dir / "correlation_matrix.png")
        plt.close()
