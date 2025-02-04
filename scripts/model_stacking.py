import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_loader import load_data

def stack_models(df):
    """Uses two models and averages their predictions."""
    X = df.drop(columns=["Survived", "Name", "Ticket", "Cabin"])
    y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    preds_rf = rf.predict_proba(X_test)[:, 1]
    preds_gb = gb.predict_proba(X_test)[:, 1]
    
    final_preds = (preds_rf + preds_gb) / 2  # Averaging predictions
    final_preds = np.round(final_preds)  # Convert to 0 or 1
    
    accuracy = accuracy_score(y_test, final_preds)
    print(f"Stacked Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    df = load_data()
    stack_models(df)
