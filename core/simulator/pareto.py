"""Module 3: Incentive Scenario Simulator - Pareto Frontier

Plots cost vs. retention trade-off across scenarios,
identifies Pareto-optimal points, and generates comparison charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_COLORS = {
    'Baseline': '#95A5A6',
    'Uniform +10%': '#3498DB',
    'Target At-Risk': '#E74C3C',
    'Reward Stars': '#F1C40F',
    'Balanced Growth': '#27AE60',
    'Cost Reduction': '#8E44AD',
}


def find_pareto_front(costs: np.ndarray, retentions: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal points (minimize cost, maximize retention)."""
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower/equal cost AND higher/equal retention
            # with at least one strict inequality
            if (costs[j] <= costs[i] and retentions[j] >= retentions[i] and
                    (costs[j] < costs[i] or retentions[j] > retentions[i])):
                is_pareto[i] = False
                break

    return np.where(is_pareto)[0]


def plot_pareto(summary_df: pd.DataFrame) -> go.Figure:
    """Scatter plot with error bars and Pareto frontier line."""
    fig = go.Figure()

    costs = summary_df['total_cost_med'].values
    rets = summary_df['retention_rate_med'].values
    names = summary_df['scenario'].values

    # Error bars
    for _, row in summary_df.iterrows():
        name = row['scenario']
        color = SCENARIO_COLORS.get(name, '#7F8C8D')
        fig.add_trace(go.Scatter(
            x=[row['total_cost_med']],
            y=[row['retention_rate_med']],
            error_x=dict(
                type='data',
                symmetric=False,
                array=[row['total_cost_hi'] - row['total_cost_med']],
                arrayminus=[row['total_cost_med'] - row['total_cost_lo']],
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[row['retention_rate_hi'] - row['retention_rate_med']],
                arrayminus=[row['retention_rate_med'] - row['retention_rate_lo']],
            ),
            mode='markers+text',
            marker=dict(size=14, color=color),
            text=[name],
            textposition='top center',
            name=name,
            showlegend=True,
        ))

    # Pareto frontier
    pareto_idx = find_pareto_front(costs, rets)
    if len(pareto_idx) > 1:
        pareto_sorted = pareto_idx[np.argsort(costs[pareto_idx])]
        fig.add_trace(go.Scatter(
            x=costs[pareto_sorted],
            y=rets[pareto_sorted],
            mode='lines',
            line=dict(dash='dash', color='rgba(0,0,0,0.3)', width=2),
            name='Pareto Frontier',
            showlegend=True,
        ))

    fig.update_layout(
        title='Cost vs. Retention — Scenario Trade-off',
        xaxis_title='Total Commission Cost (10K KRW)',
        yaxis_title='Expected Retention Rate',
        width=700, height=560,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_scenario_comparison(summary_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of retention and cost change vs. baseline."""
    baseline = summary_df[summary_df['scenario'] == 'Baseline'].iloc[0]
    rows = []
    for _, row in summary_df.iterrows():
        if row['scenario'] == 'Baseline':
            continue
        rows.append({
            'scenario': row['scenario'],
            'retention_delta': (row['retention_rate_med'] - baseline['retention_rate_med']) * 100,
            'cost_delta_pct': (row['total_cost_med'] - baseline['total_cost_med'])
                              / baseline['total_cost_med'] * 100,
        })

    delta_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=delta_df['scenario'], y=delta_df['retention_delta'],
        name='Retention Change (pp)', marker_color='#27AE60',
    ))
    fig.add_trace(go.Bar(
        x=delta_df['scenario'], y=delta_df['cost_delta_pct'],
        name='Cost Change (%)', marker_color='#E74C3C',
    ))
    fig.update_layout(
        barmode='group',
        title='Scenario Impact vs. Baseline',
        yaxis_title='Change',
        width=700, height=560,
    )
    return fig


def plot_distribution_ridge(all_dist: dict, metric: str = 'retention_rate') -> go.Figure:
    """Violin plot showing MC distribution for each scenario."""
    records = []
    for name, dist in all_dist.items():
        for val in dist[metric].values:
            records.append({'scenario': name, metric: val})
    plot_df = pd.DataFrame(records)

    fig = px.violin(
        plot_df, x='scenario', y=metric, color='scenario',
        color_discrete_map=SCENARIO_COLORS,
        title=f'Monte Carlo Distribution — {metric.replace("_", " ").title()}',
        width=700, height=560,
    )
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title=metric)
    return fig


def save_figures(figures: dict, output_dir: Path = None):
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig.write_image(str(output_dir / f'{name}.png'), scale=2)
        fig.write_html(str(output_dir / f'{name}.html'))
    print(f"Saved {len(figures)} figures to {output_dir}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from core.simulator.monte_carlo import load_assets, run_all_scenarios

    df, model = load_assets()
    summary_df, all_dist = run_all_scenarios(df, model)

    fig_pareto = plot_pareto(summary_df)
    fig_comp = plot_scenario_comparison(summary_df)
    fig_violin = plot_distribution_ridge(all_dist, 'retention_rate')

    fig_pareto.show()
    fig_comp.show()
    fig_violin.show()

    save_figures({
        'pareto_cost_retention': fig_pareto,
        'scenario_comparison': fig_comp,
        'mc_retention_violin': fig_violin,
    })
