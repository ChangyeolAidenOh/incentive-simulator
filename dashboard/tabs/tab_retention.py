"""Tab 2: Retention Predictor — model comparison, SHAP, PDP."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    'tenure_months', 'monthly_contracts', 'retention_rate_13m',
    'commission_rate', 'customer_satisfaction', 'cross_sell_count',
    'complaint_count', 'training_hours', 'monthly_premium_10k',
    'commission_cost_10k',
]


@st.cache_resource
def load_model():
    return joblib.load(PROJECT_ROOT / 'models' / 'best_tree_model.joblib')


@st.cache_data
def load_test_data():
    X = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv')
    y = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv').squeeze()
    return X, y


def render():
    st.header('Retention Predictor')

    # Model comparison
    st.subheader('Model Comparison')
    comp = pd.DataFrame({
        'Model': ['LogisticReg', 'XGB', 'LGBM', 'CAT', 'Stacking'],
        'ROC-AUC': [0.9225, 0.9139, 0.9113, 0.9176, 0.9177],
        'F1': [0.7319, 0.7091, 0.7093, 0.7017, 0.7024],
        'Precision': [0.6293, 0.6389, 0.6421, 0.6192, 0.6949],
        'Recall': [0.8745, 0.7965, 0.7922, 0.8095, 0.7100],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.info('LogisticReg이 ROC-AUC 최고 — 합성 데이터의 선형 구조 때문. '
            '실데이터에서는 Stacking이 비선형 패턴을 더 잘 잡을 것으로 예상.')

    # Churn risk Top-N
    st.subheader('High-Risk Agents (Top 20)')
    model = load_model()
    X_test, y_test = load_test_data()

    probs = model.predict_proba(X_test)[:, 1]
    risk_df = X_test.copy()
    risk_df['churn_prob'] = probs
    risk_df['actual'] = y_test.values
    top_risk = risk_df.nlargest(20, 'churn_prob')
    st.dataframe(
        top_risk[['churn_prob', 'actual', 'customer_satisfaction',
                  'monthly_contracts', 'retention_rate_13m']].round(3),
        use_container_width=True,
    )

    # SHAP & PDP figures
    st.subheader('Feature Importance (SHAP)')
    c1, c2 = st.columns(2)
    with c1:
        img = PROJECT_ROOT / 'figures' / 'shap_summary.png'
        if img.exists():
            st.image(str(img), caption='SHAP Beeswarm')
    with c2:
        img = PROJECT_ROOT / 'figures' / 'shap_bar.png'
        if img.exists():
            st.image(str(img), caption='SHAP Feature Importance')

    st.subheader('Partial Dependence')
    pdp_img = PROJECT_ROOT / 'figures' / 'pdp_top4.png'
    if pdp_img.exists():
        st.image(str(pdp_img), caption='PDP — Top 4 Features')

    st.subheader('Waterfall — Highest Risk Agent')
    wf_img = PROJECT_ROOT / 'figures' / 'shap_waterfall_high_risk.png'
    if wf_img.exists():
        st.image(str(wf_img))
