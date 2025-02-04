from data_loader import load_data  

def extract_title(df, name_column="Name"):
    """Extracts passenger titles from a specified name column."""
    df = df.copy()
    
    if name_column in df.columns:
        df["Title"] = df[name_column].str.extract(" ([A-Za-z]+)\.")
        title_replacements = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Don": "Other", "Rev": "Other", "Dr": "Other",
            "Major": "Other", "Col": "Other", "Capt": "Other",
            "Sir": "Other", "Jonkheer": "Other", "Dona": "Other",
            "Countess": "Other", "Lady": "Other"
        }
        df["Title"].replace(title_replacements, inplace=True)
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
