"""XGBoost model training script"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_xgboost_model():
    """Train XGBoost model for churn prediction"""

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    data = {
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'support_calls': np.random.randint(0, 10, n_samples)
    }

    df = pd.DataFrame(data)
    churn_prob = (
        0.3 * (df['support_calls'] > 5).astype(int) +
        0.2 * (df['tenure_months'] < 12).astype(int)
    )
    df['churn'] = (churn_prob > 0.3).astype(int)

    # Split data
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"XGBoost Model Accuracy: {accuracy:.2%}")

    # Save model
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_xgboost_model()