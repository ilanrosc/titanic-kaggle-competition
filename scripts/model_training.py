import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_data
from data_split import split_data
from preprocessing import fill_missing_values, encode_categorical, scale_features

def stack_models(df, target_column="Survived", test_size=0.2, random_state=42, exclude_columns=None):
    """
    Implements model stacking with two classifiers and averages their predictions.

    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The column to predict.
        test_size (float): Proportion of dataset to use as test set.
        random_state (int): Seed for reproducibility.
        exclude_columns (list, optional): Columns to exclude from training.

    Returns:
        tuple: (final_model, accuracy_score)
    """
    # Step 1: Preprocess Data
    df = fill_missing_values(df, strategy="mode")  # Fill missing values
    df = encode_categorical(df)  # Convert categorical to numeric
    df = scale_features(df)  # Standardize numeric features

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state, exclude_columns)
    
    # Step 3: Define Models
    model_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model_gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    
    # Step 4: Train Models
    model_rf.fit(X_train, y_train)
    model_gb.fit(X_train, y_train)
    
    # Step 5: Predict Probabilities
    preds_rf = model_rf.predict_proba(X_test)[:, 1]
    preds_gb = model_gb.predict_proba(X_test)[:, 1]
    
    # Step 6: Average Predictions (Blending Technique)
    final_preds = (preds_rf + preds_gb) / 2  
    final_preds = np.round(final_preds)  # Convert to 0 or 1

    # Step 7: Evaluate Model
    accuracy = accuracy_score(y_test, final_preds)
    
    return (model_rf, model_gb, accuracy)  # Return both models and accuracy

if __name__ == "__main__":
    df = load_data()
    
    # Apply Model Stacking with Preprocessing
    model_rf, model_gb, accuracy = stack_models(df, exclude_columns=["Name", "Ticket", "Cabin", "PassengerId"])
    
    print(f"✅ Model stacking completed successfully!")
    print(f"📊 Stacked Model Accuracy: {accuracy:.4f}")  # Explicitly prints accuracy
