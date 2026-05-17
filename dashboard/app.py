"""Agent Incentive Scenario Simulator — Streamlit Dashboard

Run: .streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title='Agent Incentive Simulator',
    page_icon='📊',
    layout='wide',
)

st.title('Agent Incentive Scenario Simulator')
st.caption('Insurance agent performance profiling, churn prediction, '
           'and incentive scenario trade-off analysis')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Agent Profile',
    'Retention Predictor',
    'Scenario Simulator',
    'Scenario Comparison',
    'Causal Exploration',
    'Market Context',
])

with tab1:
    from dashboard.tabs.tab_agent_profile import render
    render()

with tab2:
    from dashboard.tabs.tab_retention import render
    render()

with tab3:
    from dashboard.tabs.tab_simulator import render
    render()

with tab4:
    from dashboard.tabs.tab_comparison import render
    render()

with tab5:
    from dashboard.tabs.tab_causal import render
    render()

with tab6:
    from dashboard.tabs.tab_market import render
    render()
