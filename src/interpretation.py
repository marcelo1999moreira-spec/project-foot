from pathlib import Path

from .evaluation import plot_feature_importance


def run_interpretation(best_pipe, numerical_cols, categorical_cols, figures_dir: Path):
    """
    Wrapper to run model interpretation (feature importance).
    """
    plot_feature_importance(best_pipe, numerical_cols, categorical_cols, figures_dir)
