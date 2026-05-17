"""Module 2: Retention Predictor - Stacking Ensemble

XGBoost + LightGBM + CatBoost as base learners,
LogisticRegression meta-learner via StackingClassifier.
Compares all models (baseline LR, 3 trees, stacking) in one table.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor.baseline import FEATURES, TARGET, RANDOM_STATE, load_data


def build_base_learners(pos_weight: float = 1.0) -> list:
    """Create base learner instances with class imbalance handling."""
    return [
        ('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
        )),
        ('cat', CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            auto_class_weights='Balanced',
            random_state=RANDOM_STATE,
            verbose=0,
        )),
    ]


def build_stacking(pos_weight: float = 1.0) -> StackingClassifier:
    """Build stacking classifier with LR meta-learner."""
    base = build_base_learners(pos_weight)
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1,
    )


def evaluate_single(model, X_test: np.ndarray, y_test: np.ndarray,
                    label: str) -> dict:
    """Evaluate a fitted model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'model': label,
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
    }


def compare_all(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train and evaluate all models, return comparison table."""
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    results = []

    # Baseline LR
    from core.predictor.baseline import train_baseline, evaluate_model
    lr_model, scaler = train_baseline(X_train, y_train)
    X_test_sc = scaler.transform(X_test)
    results.append(evaluate_single(lr_model, X_test_sc, y_test, 'LogisticReg'))

    # Individual tree models
    base_learners = build_base_learners(pos_weight)
    best_tree_name, best_tree_model, best_tree_auc = None, None, 0

    for name, model in base_learners:
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        m = evaluate_single(model, X_test, y_test, name.upper())
        results.append(m)

        if m['roc_auc'] > best_tree_auc:
            best_tree_name = name
            best_tree_model = model
            best_tree_auc = m['roc_auc']

    # Stacking
    print("  Training Stacking...")
    stack = build_stacking(pos_weight)
    stack.fit(X_train, y_train)
    results.append(evaluate_single(stack, X_test, y_test, 'Stacking'))

    comp = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    return comp, stack, best_tree_model, best_tree_name


def save_models(stack, best_tree, best_tree_name: str):
    model_dir = PROJECT_ROOT / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, model_dir / 'stacking_model.joblib')
    joblib.dump(best_tree, model_dir / 'best_tree_model.joblib')

    # Save test data for explainer
    return best_tree_name


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Churn rate - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")
    print(f"Pos weight: {(y_train == 0).sum() / (y_train == 1).sum():.2f}\n")

    comp, stack, best_tree, best_tree_name = compare_all(
        X_train, X_test, y_train, y_test,
    )

    print("\nModel comparison:")
    print(comp.to_string(index=False))

    # Classification report for stacking
    y_pred = stack.predict(X_test)
    print(f"\n--- Stacking detail ---")
    print(classification_report(y_test, y_pred, digits=3))

    # Cross-validation for stacking
    print("Running 5-fold CV for Stacking (takes a moment)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(stack, X_train, y_train,
                                cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    save_models(stack, best_tree, best_tree_name)

    # Save test split for explainer
    test_dir = PROJECT_ROOT / 'data' / 'processed'
    test_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(test_dir / 'X_test.csv', index=False)
    y_test.to_csv(test_dir / 'y_test.csv', index=False, header=True)

    print(f"\nSaved: models/stacking_model.joblib")
    print(f"Saved: models/best_tree_model.joblib ({best_tree_name.upper()})")
    print(f"Saved: data/processed/X_test.csv, y_test.csv")
