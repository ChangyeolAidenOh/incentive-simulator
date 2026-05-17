"""Module 2: Retention Predictor - TreeSHAP + PDP

Loads the best individual tree model from stacking.py,
generates SHAP summary/waterfall and PDP for top features.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    'tenure_months', 'monthly_contracts', 'retention_rate_13m',
    'commission_rate', 'customer_satisfaction', 'cross_sell_count',
    'complaint_count', 'training_hours', 'monthly_premium_10k',
    'commission_cost_10k',
]


def load_model_and_data() -> tuple:
    """Load best tree model and test set."""
    model_dir = PROJECT_ROOT / 'models'
    data_dir = PROJECT_ROOT / 'data' / 'processed'

    model = joblib.load(model_dir / 'best_tree_model.joblib')
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()

    return model, X_test, y_test


def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """Compute TreeSHAP values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: Path):
    """SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_bar(shap_values, save_path: Path):
    """SHAP mean absolute value bar plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(shap_values, idx: int, save_path: Path):
    """SHAP waterfall for a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pdp(model, X: pd.DataFrame, top_n: int = 4, save_path: Path = None):
    """Partial Dependence Plot for top-N features by SHAP importance."""
    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features=list(range(top_n)),
        feature_names=FEATURES, ax=axes,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_top_features(shap_values, n: int = 4) -> list:
    """Return top-N feature indices by mean |SHAP|."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    return top_idx.tolist()


if __name__ == '__main__':
    model, X_test, y_test = load_model_and_data()
    print(f"Test set: {len(X_test)} samples")
    print(f"Model type: {type(model).__name__}")

    print("Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_test)

    fig_dir = PROJECT_ROOT / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # SHAP summary (beeswarm)
    plot_shap_summary(shap_values, X_test, fig_dir / 'shap_summary.png')
    print("Saved: figures/shap_summary.png")

    # SHAP bar (feature importance)
    plot_shap_bar(shap_values, fig_dir / 'shap_bar.png')
    print("Saved: figures/shap_bar.png")

    # SHAP waterfall (highest-risk churned agent)
    y_prob = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_prob)
    plot_shap_waterfall(shap_values, high_risk_idx,
                        fig_dir / 'shap_waterfall_high_risk.png')
    print(f"Saved: figures/shap_waterfall_high_risk.png "
          f"(idx={high_risk_idx}, prob={y_prob[high_risk_idx]:.3f})")

    # PDP for top-4 features
    top_idx = get_top_features(shap_values, n=4)
    top_names = [FEATURES[i] for i in top_idx]
    print(f"Top-4 features: {top_names}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    PartialDependenceDisplay.from_estimator(
        model, X_test, features=top_idx,
        feature_names=FEATURES, ax=axes,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / 'pdp_top4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/pdp_top4.png")

    print("\nModule 2 explainer complete.")
