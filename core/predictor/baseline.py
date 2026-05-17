"""Module 2: Retention Predictor - Baseline (Logistic Regression)

Quick sanity-check model before the stacking ensemble.
Handles class imbalance via class_weight='balanced'.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    'tenure_months', 'monthly_contracts', 'retention_rate_13m',
    'commission_rate', 'customer_satisfaction', 'cross_sell_count',
    'complaint_count', 'training_hours', 'monthly_premium_10k',
    'commission_cost_10k',
]

TARGET = 'churned_12m'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> tuple:
    """Load synthetic data and split into train/test."""
    path = PROJECT_ROOT / 'data' / 'synthetic' / 'agent_performance.csv'
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Fit Logistic Regression on scaled features."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_sc, y_train)
    return model, scaler


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series,
                   scaler=None, label: str = 'Logistic Regression') -> dict:
    """Print classification report and return metrics dict."""
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {label} ---")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {auc:.4f}")

    return {'model': label, 'roc_auc': round(auc, 4)}


def cross_validate_auc(model, X: pd.DataFrame, y: pd.Series,
                       scaler=None) -> float:
    """5-fold stratified CV ROC-AUC."""
    X_arr = scaler.transform(X) if scaler else X.values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_arr, y, cv=cv, scoring='roc_auc')
    return scores.mean(), scores.std()


def save_baseline(model, scaler):
    model_dir = PROJECT_ROOT / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'baseline_lr.joblib')
    joblib.dump(scaler, model_dir / 'predictor_scaler.joblib')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Churn rate - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    model, scaler = train_baseline(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, scaler)

    cv_mean, cv_std = cross_validate_auc(model, X_train, y_train, scaler)
    print(f"CV ROC-AUC: {cv_mean:.4f} +/- {cv_std:.4f}")

    save_baseline(model, scaler)
    print("\nSaved: models/baseline_lr.joblib, models/predictor_scaler.joblib")
