import pandas as pd

def load_raw_data(path: str = "data/raw/players_21.csv") -> pd.DataFrame:
    """
    Load the raw FIFA 21 players dataset from CSV.
    """
    df = pd.read_csv(path)
    print("Initial dataset shape:", df.shape)
    return df
