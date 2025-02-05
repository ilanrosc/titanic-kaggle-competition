from .data_loader import load_data  

def extract_title(df, name_column="Name"):
    """
    Extracts the title from the passenger's name and creates a new 'Title' column.

    Args:
        df (pd.DataFrame): The dataset.
        name_column (str): The column name containing passenger names.

    Returns:
        pd.DataFrame: Updated DataFrame with the new 'Title' column.
    """
    df = df.copy()  

    df["Title"] = df[name_column].str.extract(r" ([A-Za-z]+)\.")  # Extract title from name

    # Standardize rare/misspelled titles
    title_replacements = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", 
        "Dr": "Rare", "Rev": "Rare", "Major": "Rare", "Col": "Rare", 
        "Capt": "Rare", "Sir": "Rare", "Lady": "Rare", "Jonkheer": "Rare", 
        "Don": "Rare", "Dona": "Rare", "Countess": "Rare"
    }

    df["Title"] = df["Title"].replace(title_replacements)

    return df

def create_family_size(df, sibsp_col="SibSp", parch_col="Parch"):
    """Creates a new feature dynamically for family size."""
    df = df.copy()

    if sibsp_col in df.columns and parch_col in df.columns:
        df["FamilySize"] = df[sibsp_col] + df[parch_col] + 1
    return df


if __name__ == "__main__":
    df = load_data()  
    df = extract_title(df, name_column="Name")
    df = create_family_size(df, sibsp_col="SibSp", parch_col="Parch")
    print(df[["Name", "Title", "FamilySize"]].head())
