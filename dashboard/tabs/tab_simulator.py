"""Tab 3: Scenario Simulator — interactive commission sliders + MC."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    'tenure_months', 'monthly_contracts', 'retention_rate_13m',
    'commission_rate', 'customer_satisfaction', 'cross_sell_count',
    'complaint_count', 'training_hours', 'monthly_premium_10k',
    'commission_cost_10k',
]
SEGMENT_ORDER = ['Star', 'Stable', 'Developing', 'At-Risk']


@st.cache_data
def load_agents():
    return pd.read_csv(PROJECT_ROOT / 'data' / 'synthetic' / 'agent_performance.csv')


@st.cache_resource
def load_model():
    return joblib.load(PROJECT_ROOT / 'models' / 'best_tree_model.joblib')


def run_quick_mc(df, model, multipliers, n_iter=1000):
    """Lightweight MC for interactive use."""
    rng = np.random.RandomState(42)
    n = len(df)
    records = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx].copy()

        for seg, mult in multipliers.items():
            mask = sample['segment'] == seg
            sample.loc[mask, 'commission_rate'] *= mult
        sample['commission_cost_10k'] = (
            sample['commission_rate'] / 100 * sample['monthly_premium_10k']
        )

        X = sample[FEATURES]
        churn_prob = model.predict_proba(X)[:, 1]
        retain = 1 - churn_prob

        records.append({
            'retention_rate': retain.mean(),
            'total_cost': sample['commission_cost_10k'].sum(),
            'productivity': (retain * sample['monthly_contracts']).sum(),
        })

    return pd.DataFrame(records)


def render():
    st.header('Scenario Simulator')
    st.caption('Commission rate multipliers by segment → Monte Carlo 1,000 runs')

    df = load_agents()
    model = load_model()

    # Sliders
    cols = st.columns(4)
    multipliers = {}
    for col, seg in zip(cols, SEGMENT_ORDER):
        with col:
            val = st.slider(
                f'{seg} (%)',
                min_value=-20, max_value=30, value=0, step=1,
                key=f'slider_{seg}',
            )
            multipliers[seg] = 1 + val / 100

    if st.button('Run Simulation', type='primary'):
        with st.spinner('Running 1,000 MC iterations...'):
            dist = run_quick_mc(df, model, multipliers)

        c1, c2, c3 = st.columns(3)
        c1.metric('Retention (median)',
                  f"{dist['retention_rate'].median():.1%}",
                  f"{(dist['retention_rate'].median() - 0.680) * 100:+.1f}pp vs baseline")
        c2.metric('Cost (median)',
                  f"{dist['total_cost'].median():,.0f}",
                  f"{(dist['total_cost'].median() / 2_368_303 - 1) * 100:+.1f}% vs baseline")
        c3.metric('Productivity (median)',
                  f"{dist['productivity'].median():,.0f}",
                  f"{(dist['productivity'].median() / 19_512 - 1) * 100:+.1f}% vs baseline")

        # Distribution plots
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dist['retention_rate'], nbinsx=40,
            name='Retention Rate', marker_color='#2E86C1',
        ))
        med = dist['retention_rate'].median()
        lo = dist['retention_rate'].quantile(0.025)
        hi = dist['retention_rate'].quantile(0.975)
        fig.add_vline(x=med, line_dash='dash', line_color='red',
                      annotation_text=f'Median: {med:.3f}')
        fig.add_vrect(x0=lo, x1=hi, fillcolor='rgba(46,134,193,0.1)',
                      line_width=0, annotation_text='95% CI')
        fig.update_layout(
            title='MC Distribution — Retention Rate',
            xaxis_title='Retention Rate', yaxis_title='Count',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info('슬라이더를 조정한 후 "Run Simulation"을 누르세요.')
