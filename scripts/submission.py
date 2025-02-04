import pandas as pd
from .data_loader import load_data
from model_training import train_model

def generate_submission(model, test_data, filename="submission.csv"):
    """
    Creates a CSV file for Kaggle submission.

    Args:
        model: Trained model used for prediction.
        test_data (pd.DataFrame): The dataset for making predictions.
        filename (str): Output CSV filename.

    Returns:
        None
    """
    # Ensure only the features are passed
    feature_columns = [col for col in test_data.columns if col.lower() not in ["passengerid"]]
    
    predictions = model.predict(test_data[feature_columns])

    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

    # Save to CSV
    submission.to_csv(filename, index=False)
    print(f"✅ Submission file saved as {filename}")

if __name__ == "__main__":
    # Load and train model
    df_train = load_data(filename="train.csv")
    model, _ = train_model(df_train, exclude_columns=["Name", "Ticket", "Cabin", "PassengerId"])

    # Load test data
    df_test = load_data(filename="test.csv")

    # Generate submission
    generate_submission(model, df_test)
