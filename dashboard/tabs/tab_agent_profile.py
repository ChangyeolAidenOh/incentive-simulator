"""Tab 1: Agent Profile — UMAP clusters + segment KPI summary."""

import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'agent_clustered.csv')
    return df


def render():
    st.header('Agent Profile')

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    for col, seg in zip([col1, col2, col3, col4],
                        ['Star', 'Stable', 'Developing', 'At-Risk']):
        count = len(df[df['segment'] == seg])
        col.metric(seg, f'{count:,}', f'{count/len(df):.1%}')

    # UMAP plots
    umap_true = PROJECT_ROOT / 'figures' / 'umap_true_segments.html'
    umap_cluster = PROJECT_ROOT / 'figures' / 'umap_kmeans_clusters.html'

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('True Segments')
        if umap_true.exists():
            st.components.v1.html(umap_true.read_text(), height=500, scrolling=True)
        else:
            img = PROJECT_ROOT / 'figures' / 'umap_true_segments.png'
            if img.exists():
                st.image(str(img))
    with c2:
        st.subheader('KMeans Clusters')
        if umap_cluster.exists():
            st.components.v1.html(umap_cluster.read_text(), height=500, scrolling=True)
        else:
            img = PROJECT_ROOT / 'figures' / 'umap_kmeans_clusters.png'
            if img.exists():
                st.image(str(img))

    # KPI table
    st.subheader('Segment KPI Summary')
    kpi = df.groupby('segment').agg(
        count=('agent_id', 'count'),
        avg_contracts=('monthly_contracts', 'mean'),
        avg_retention=('retention_rate_13m', 'mean'),
        avg_satisfaction=('customer_satisfaction', 'mean'),
        churn_rate=('churned_12m', 'mean'),
    ).round(3)
    kpi = kpi.reindex(['Star', 'Stable', 'Developing', 'At-Risk'])
    st.dataframe(kpi, use_container_width=True)

    # Radar + churn charts
    c3, c4 = st.columns(2)
    with c3:
        radar = PROJECT_ROOT / 'figures' / 'segment_kpi_radar.png'
        if radar.exists():
            st.image(str(radar))
    with c4:
        churn = PROJECT_ROOT / 'figures' / 'churn_by_segment.png'
        if churn.exists():
            st.image(str(churn))
