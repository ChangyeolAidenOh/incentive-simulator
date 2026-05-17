"""Module 1: Agent Profiler - UMAP Visualization

UMAP 2D projection colored by true segments and KMeans clusters.
Includes method comparison bar, radar KPI, and churn bar chart.
"""

import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENT_ORDER = ['Star', 'Stable', 'Developing', 'At-Risk']

SEGMENT_COLORS = {
    'Star': '#FFD700',
    'Stable': '#2E86C1',
    'Developing': '#27AE60',
    'At-Risk': '#E74C3C',
    'Noise': '#95A5A6',
}


# ------------------------------------------------------------------
# UMAP
# ------------------------------------------------------------------

def compute_umap(
    X_scaled: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce scaled features to 2D via UMAP."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean',
    )
    return reducer.fit_transform(X_scaled)


def plot_umap(
    df: pd.DataFrame,
    embedding: np.ndarray,
    color_col: str = 'segment',
    title: str = 'UMAP - Agent Segments',
) -> go.Figure:
    """Scatter plot of UMAP embedding."""
    df_plot = df.copy()
    df_plot['umap_1'] = embedding[:, 0]
    df_plot['umap_2'] = embedding[:, 1]

    fig = px.scatter(
        df_plot,
        x='umap_1',
        y='umap_2',
        color=color_col,
        color_discrete_map=SEGMENT_COLORS,
        category_orders={color_col: SEGMENT_ORDER + ['Noise']},
        hover_data=['agent_id', 'tenure_months', 'monthly_contracts',
                    'retention_rate_13m'],
        title=title,
        width=700,
        height=560,
        opacity=0.6,
    )
    fig.update_layout(
        xaxis_title='UMAP-1',
        yaxis_title='UMAP-2',
        legend_title=color_col.replace('_', ' ').title(),
    )
    return fig


# ------------------------------------------------------------------
# Method comparison
# ------------------------------------------------------------------

def plot_method_comparison(comp: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of ARI and silhouette by method."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp['method'], y=comp['ari'],
        name='ARI', marker_color='#2E86C1',
    ))
    fig.add_trace(go.Bar(
        x=comp['method'], y=comp['silhouette'],
        name='Silhouette', marker_color='#E67E22',
    ))
    fig.update_layout(
        barmode='group',
        title='Clustering Method Comparison',
        yaxis_title='Score',
        width=700, height=560,
    )
    return fig


# ------------------------------------------------------------------
# Segment KPI charts
# ------------------------------------------------------------------

def plot_kpi_radar(
    kpi: pd.DataFrame,
    title: str = 'Segment KPI Radar',
) -> go.Figure:
    """Radar chart comparing segments across normalized KPIs."""
    metrics = [
        'avg_contracts', 'avg_retention', 'avg_satisfaction',
        'avg_commission', 'avg_premium',
    ]
    labels = ['Contracts', 'Retention', 'Satisfaction', 'Commission', 'Premium']

    kpi_norm = kpi[metrics].copy()
    for col in metrics:
        cmin, cmax = kpi_norm[col].min(), kpi_norm[col].max()
        if cmax > cmin:
            kpi_norm[col] = (kpi_norm[col] - cmin) / (cmax - cmin)
        else:
            kpi_norm[col] = 0.5

    fig = go.Figure()
    for seg in SEGMENT_ORDER:
        if seg not in kpi_norm.index:
            continue
        vals = kpi_norm.loc[seg].tolist()
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels + [labels[0]],
            name=seg,
            line=dict(color=SEGMENT_COLORS[seg]),
            fill='toself',
            opacity=0.3,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        width=700, height=560,
    )
    return fig


def plot_churn_by_segment(
    kpi: pd.DataFrame,
    title: str = 'Churn Rate by Segment',
) -> go.Figure:
    """Bar chart of churn rates per segment."""
    kpi_reset = kpi.reset_index()
    fig = px.bar(
        kpi_reset,
        x='segment',
        y='churn_rate',
        color='segment',
        color_discrete_map=SEGMENT_COLORS,
        category_orders={'segment': SEGMENT_ORDER},
        title=title,
        width=700,
        height=560,
        text_auto='.1%',
    )
    fig.update_layout(
        yaxis_title='Churn Rate',
        xaxis_title='Segment',
        showlegend=False,
    )
    return fig


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

def save_figures(figures: dict, output_dir: Path = None):
    """Save Plotly figures as PNG and HTML."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig.write_image(str(output_dir / f'{name}.png'), scale=2)
        fig.write_html(str(output_dir / f'{name}.html'))
    print(f"Saved {len(figures)} figures to {output_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from core.profiler.clustering import (
        load_agent_data, scale_features, fit_kmeans,
        map_clusters_to_segments, compare_methods, segment_kpi_summary,
    )

    df = load_agent_data()
    scaler, X_scaled = scale_features(df)

    # Method comparison chart
    comp = compare_methods(df, X_scaled)
    fig_comp = plot_method_comparison(comp)

    # Best method (KMeans) for UMAP coloring
    model, labels = fit_kmeans(X_scaled)
    cluster_map = map_clusters_to_segments(df['segment'], labels)
    df['cluster'] = labels
    df['cluster_segment'] = df['cluster'].map(cluster_map).fillna('Noise')

    kpi = segment_kpi_summary(df)

    print("Computing UMAP...")
    embedding = compute_umap(X_scaled)

    fig_true = plot_umap(df, embedding, 'segment', 'UMAP - True Segments')
    fig_cluster = plot_umap(df, embedding, 'cluster_segment', 'UMAP - KMeans Clusters')
    fig_radar = plot_kpi_radar(kpi)
    fig_churn = plot_churn_by_segment(kpi)

    fig_comp.show()
    fig_true.show()
    fig_cluster.show()
    fig_radar.show()
    fig_churn.show()

    save_figures({
        'method_comparison': fig_comp,
        'umap_true_segments': fig_true,
        'umap_kmeans_clusters': fig_cluster,
        'segment_kpi_radar': fig_radar,
        'churn_by_segment': fig_churn,
    })
