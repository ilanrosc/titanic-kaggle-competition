from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from data_loader import load_data

def fill_missing_values(df: pd.DataFrame, 
                        num_strategy: str = "median", 
                        cat_strategy: str = "mode", 
                        fill_value=None, 
                        exclude_columns=None) -> pd.DataFrame:
    """
    Handles missing values separately for numerical and categorical features.

    Args:
        df (pd.DataFrame): The dataset.
        num_strategy (str): Strategy for numerical columns - "mean", "median", "mode", "constant", "ffill", "bfill", or "drop".
        cat_strategy (str): Strategy for categorical columns - "mode", "constant", "ffill", "bfill", or "drop".
        fill_value: Value to use when strategy="constant".
        exclude_columns (list, optional): List of columns to exclude from processing.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """

    df_copy = df.copy()

    # Exclude specified columns
    if exclude_columns:
        cols_to_process = [col for col in df_copy.columns if col not in exclude_columns]
    else:
        cols_to_process = df_copy.columns

    # Identify numerical and categorical columns
    num_cols = df_copy[cols_to_process].select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_copy[cols_to_process].select_dtypes(include=["object", "category"]).columns

    # Handle missing values for numerical columns
    for col in num_cols:
        if df_copy[col].isnull().sum() > 0:
            if num_strategy == "mean":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif num_strategy == "median":
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif num_strategy == "mode":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif num_strategy == "constant":
                if fill_value is not None:
                    df_copy[col] = df_copy[col].fillna(fill_value)
                else:
                    raise ValueError("For strategy='constant', a fill_value must be provided.")
            elif num_strategy == "ffill":
                df_copy[col] = df_copy[col].ffill()
            elif num_strategy == "bfill":
                df_copy[col] = df_copy[col].bfill()
            elif num_strategy == "drop":
                df_copy.dropna(subset=[col], inplace=True)

    # Handle missing values for categorical columns
    for col in cat_cols:
        if df_copy[col].isnull().sum() > 0:
            if cat_strategy == "mode":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif cat_strategy == "constant":
                if fill_value is not None:
                    df_copy[col] = df_copy[col].fillna(fill_value)
                else:
                    raise ValueError("For strategy='constant', a fill_value must be provided.")
            elif cat_strategy == "ffill":
                df_copy[col] = df_copy[col].ffill()
            elif cat_strategy == "bfill":
                df_copy[col] = df_copy[col].bfill()
            elif cat_strategy == "drop":
                df_copy.dropna(subset=[col], inplace=True)

    return df_copy

def encode_categorical(df):
    """Encodes categorical features dynamically."""
    df = df.copy() 
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def scale_features(df):
    """Standardizes numerical features dynamically."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [col for col in num_cols if col.lower() not in ["passengerid", "survived"]]  # Exclude IDs & target
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    return df

if __name__ == "__main__":
    df = load_data()  
    
    # Example usage:
    print("Fill missing values using different strategies for numerical & categorical columns")
    df_cleaned_v1 = fill_missing_values(df, num_strategy="median", cat_strategy="mode", exclude_columns=["PassengerId"])
    print(df_cleaned_v1.head())
    print("Forward fill for categorical, mean fill for numerical")
    df_cleaned_v2 = fill_missing_values(df, num_strategy="mean", cat_strategy="ffill")
    print(df_cleaned_v2.head())
    print("Drop missing values in numerical features but use mode for categorical")
    df_cleaned_v3 = fill_missing_values(df, num_strategy="drop", cat_strategy="mode")
    print(df_cleaned_v3.head())
    print("Fill all missing categorical values with 'Unknown' and numeric with median")
    df_cleaned_v4 = fill_missing_values(df, num_strategy="median", cat_strategy="constant", fill_value="Unknown")
    print(df_cleaned_v4.head())
