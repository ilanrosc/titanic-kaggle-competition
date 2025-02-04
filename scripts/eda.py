import matplotlib.pyplot as plt
import seaborn as sns
from .data_loader import load_data

def summarize_data(df):
    """Prints dataset info, missing values, and summary statistics."""
    print("📌 Dataset Info:")
    df.info()

    print("\n📌 Missing Values:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print(f"\n📌 Duplicate Rows: {duplicate_count}")

    print("\n📌 Summary Statistics:")
    print(df.describe())


def plot_distributions(df, exclude_columns=None):
    """
    Plots distributions for all numerical columns.

    Args:
        df (pd.DataFrame): Dataset
        exclude_columns (list, optional): List of columns to exclude from visualization.
    """
    exclude_columns = exclude_columns if exclude_columns else []
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [col for col in num_cols if col.lower() not in exclude_columns]  # Exclude specified columns
    
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

if __name__ == "__main__":
    df = load_data()  
    summarize_data(df)
    plot_distributions(df, exclude_columns=["passengerid", "survived"])

