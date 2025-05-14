from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

def predictions(model, data):
    """
    Make predictions using the trained model.

    Parameters:
    - model: The trained model.
    - data: The input data for making predictions.

    Returns:
    - predictions: The predicted values.
    """
    predictions = model.predict(data)
    # Convert predictions to probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[:, 1]  # Get the probability of the positive (bad loans) class
    
    return predictions, probabilities

def model_evaluation(y_true, y_pred, probabilities):
    """
    Evaluate the model's performance using various metrics.

    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.
    - probabilities: The predicted probabilities.

    Returns:
    - evaluation_results: A dictionary containing various evaluation metrics.
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probabilities)

    # Create a dictionary to store the results
    evaluation_results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

    return evaluation_results

def model_report(y_true, y_pred):
    """
    Generate a report of the model's performance.

    Returns:
    - Classification Report and Confusion Matrix.
    """
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    return f"Classification Report:\n{report}\nConfusion Matrix:\n{matrix}"