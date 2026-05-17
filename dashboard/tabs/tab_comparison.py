"""Tab 4: Scenario Comparison — pre-computed A/B/C + AI summary + PDF."""

import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_summary():
    return pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'scenario_summary.csv')


def render():
    st.header('Scenario Comparison')

    summary = load_summary()

    # Summary table
    st.subheader('All Scenarios — Median [95% CI]')
    display = summary.copy()
    display['Retention'] = display.apply(
        lambda r: f"{r['retention_rate_med']:.1%} [{r['retention_rate_lo']:.1%}, {r['retention_rate_hi']:.1%}]",
        axis=1,
    )
    display['Cost'] = display.apply(
        lambda r: f"{r['total_cost_med']:,.0f} [{r['total_cost_lo']:,.0f}, {r['total_cost_hi']:,.0f}]",
        axis=1,
    )
    display['Productivity'] = display.apply(
        lambda r: f"{r['productivity_med']:,.0f} [{r['productivity_lo']:,.0f}, {r['productivity_hi']:,.0f}]",
        axis=1,
    )
    st.dataframe(
        display[['scenario', 'Retention', 'Cost', 'Productivity']],
        use_container_width=True, hide_index=True,
    )

    # Pareto + comparison charts
    c1, c2 = st.columns(2)
    with c1:
        img = PROJECT_ROOT / 'figures' / 'pareto_cost_retention.png'
        if img.exists():
            st.image(str(img), caption='Cost vs. Retention Pareto')
    with c2:
        img = PROJECT_ROOT / 'figures' / 'scenario_comparison.png'
        if img.exists():
            st.image(str(img), caption='Impact vs. Baseline')

    # Violin
    violin = PROJECT_ROOT / 'figures' / 'mc_retention_violin.png'
    if violin.exists():
        st.image(str(violin), caption='MC Retention Distribution')

    # AI analysis
    st.subheader('AI Trade-off Summary')
    analysis_path = PROJECT_ROOT / 'data' / 'processed' / 'report_analysis.txt'
    if analysis_path.exists():
        st.markdown(analysis_path.read_text(encoding='utf-8'))
    else:
        st.warning('Run `python -m core.report.llm_backend` to generate analysis.')

    # PDF download
    pdf_path = PROJECT_ROOT / 'data' / 'processed' / 'executive_report.pdf'
    if pdf_path.exists():
        st.download_button(
            'Download Executive Report (PDF)',
            data=pdf_path.read_bytes(),
            file_name='executive_report.pdf',
            mime='application/pdf',
        )
