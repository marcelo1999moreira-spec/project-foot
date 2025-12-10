from pathlib import Path
import pandas as pd


def load_raw_data(base_dir: Path) -> pd.DataFrame:
    """
    Load the raw FIFA players dataset from data/raw/players_21.csv.
    """
    data_path = base_dir / "data" / "raw" / "players_21.csv"
    df = pd.read_csv(data_path)
    return df
