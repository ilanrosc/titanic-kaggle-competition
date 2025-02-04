from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred):
    """Computes classification performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return accuracy, f1, roc_auc
