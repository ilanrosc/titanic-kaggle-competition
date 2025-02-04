from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_loader import load_data

def fill_missing_values(df, strategy="median", constant_value=None):
    """
    Handles missing values using different strategies.

    Args:
        df (pd.DataFrame): The dataset.
        strategy (str): Strategy to use - 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill', or 'drop'.
        constant_value (optional): Value to use when strategy is 'constant'.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in df.columns:
        if df[col].isnull().sum() > 0: 
            if col in num_cols:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "constant":
                    if constant_value is not None:
                        df[col] = df[col].fillna(constant_value)
                    else:
                        raise ValueError(f"Must provide `constant_value` when strategy is 'constant'")
                elif strategy == "ffill":
                    df[col] = df[col].fillna(method="ffill")
                elif strategy == "bfill":
                    df[col] = df[col].fillna(method="bfill")
                elif strategy == "drop":
                    df.dropna(subset=[col], inplace=True)
                else:
                    raise ValueError(f"Invalid strategy '{strategy}' for numerical column '{col}'.")

            elif col in cat_cols:
                if strategy in ["mode", "ffill", "bfill", "constant"]:
                    if strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif strategy == "constant":
                        if constant_value is not None:
                            df[col] = df[col].fillna(constant_value)
                        else:
                            raise ValueError(f"Must provide `constant_value` when strategy is 'constant'")
                    elif strategy in ["ffill", "bfill"]:
                        df[col] = df[col].fillna(method=strategy)
                else:
                    raise ValueError(f"Strategy '{strategy}' is not valid for categorical column '{col}'. Use 'mode', 'constant', 'ffill', or 'bfill'.")
    
    return df

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
    df = fill_missing_values(df, strategy="mode") 
    df = encode_categorical(df)
    df = scale_features(df)
    
    print(df.head())