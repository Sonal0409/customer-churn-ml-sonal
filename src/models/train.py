"""Simple customer churn model training script"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_data():
    """Load and prepare data"""
    # For demo: Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    data = {
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'support_calls': np.random.randint(0, 10, n_samples)
    }

    df = pd.DataFrame(data)

    # Create target
    churn_prob = (
        0.3 * (df['support_calls'] > 5).astype(int) +
        0.2 * (df['tenure_months'] < 12).astype(int)
    )
    df['churn'] = (churn_prob > 0.3).astype(int)

    return df

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    # Load data
    print("Loading data...")
    df = load_data()

    # Split features and target
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"
Model Accuracy: {accuracy:.2%}")
    print("
Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("
Model saved to models/churn_model.pkl")

if __name__ == "__main__":
    main()