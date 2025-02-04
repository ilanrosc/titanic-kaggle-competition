from sklearn.model_selection import train_test_split
from .data_loader import load_data  # Import reusable data loader

def split_data(df, target_column, test_size=0.2, random_state=42, exclude_columns=None):
    """
    Splits a dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        target_column (str): The column to predict.
        test_size (float): Proportion of the dataset to use as the test set.
        random_state (int): Seed for reproducibility.
        exclude_columns (list, optional): Columns to exclude from training features.

    Returns:
        tuple: X_train, X_test, y_train, y_test (pandas DataFrames)
    """
    df = df.copy()  # Prevent modifications to original dataset
    exclude_columns = exclude_columns if exclude_columns else []
    
    # Ensure the target column is in the dataset
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Select features and target variable
    feature_columns = [col for col in df.columns if col != target_column and col in df.columns and col not in exclude_columns]
    
    X = df[feature_columns]
    y = df[target_column]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data() 
    X_train, X_test, y_train, y_test = split_data(df, target_column="Survived", exclude_columns=["Name", "Ticket", "Cabin", "PassengerId"])
    
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
