import pandas as pd
import os

def load_data(data_path="../data/titanic", filename="train.csv"):
    """
    Loads a CSV file into a pandas DataFrame.
    
    Args:
        data_path (str): Path to the data folder.
        filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = os.path.join(data_path, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"{file_path} not found.")

if __name__ == "__main__":
    df = load_data()
    print(df.head())
