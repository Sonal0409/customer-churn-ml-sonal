"""Model evaluation utilities"""


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    return metrics


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
    plt.show()
