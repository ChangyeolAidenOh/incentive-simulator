"""Exploration B: Market Context — ECOS CSI + Insurance Stats

Fetches macro indicators (CSI, interest rate) from ECOS API,
combines with insurance market data, and estimates
CSI-insurance new contract elasticity.

CSV fallback is first-class (sportswear pattern) for cases
where ECOS API key is not available.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


# ------------------------------------------------------------------
# Sample data (CSV fallback)
# ------------------------------------------------------------------

def generate_sample_market_data() -> pd.DataFrame:
    """Generate representative market context data for development.

    Sources modeled after:
      - Bank of Korea ECOS: CSI (Consumer Sentiment Index)
      - Life Insurance Association of Korea: new contracts, lapse rates
    """
    dates = pd.date_range('2022-01', '2025-12', freq='MS')
    np.random.seed(42)

    n = len(dates)
    # CSI: oscillates around 100, with post-COVID recovery
    csi_base = 95 + np.linspace(0, 10, n) + np.random.normal(0, 3, n)
    csi = np.clip(csi_base, 80, 120).round(1)

    # Base rate (Bank of Korea): gradual rise then plateau
    base_rate = np.concatenate([
        np.linspace(1.25, 3.50, n // 2),
        np.linspace(3.50, 3.00, n - n // 2),
    ]).round(2)

    # Insurance new contracts (10K): loosely correlated with CSI
    new_contracts = (csi * 50 + np.random.normal(0, 200, n)).round(0).astype(int)

    # 13-month retention rate: industry average ~75-82%
    retention = (78 + np.random.normal(0, 1.5, n)).round(1)

    df = pd.DataFrame({
        'date': dates,
        'csi': csi,
        'base_rate': base_rate,
        'new_contracts': new_contracts,
        'industry_retention_13m': retention,
    })
    return df


# ------------------------------------------------------------------
# ECOS API (optional — requires API key)
# ------------------------------------------------------------------

def fetch_ecos(stat_code: str, item_code: str, start: str, end: str,
               api_key: str = None, freq: str = 'M') -> pd.DataFrame:
    """Fetch data from Bank of Korea ECOS API.

    Args:
        stat_code: e.g. '512Y014' for CSI
        item_code: e.g. 'FME' for overall CSI
        start/end: 'YYYYMM' format
        api_key: ECOS API key (register at ecos.bok.or.kr)

    Returns empty DataFrame if API key is not set.
    """
    if not api_key:
        print("ECOS API key not set, using CSV fallback.")
        return pd.DataFrame()

    import requests
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{api_key}/json/kr/1/100/{stat_code}/{freq}/{start}/{end}/"
        f"{item_code}/A/"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get('StatisticSearch', {}).get('row', [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.rename(columns={'TIME': 'date', 'DATA_VALUE': 'value'})
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
        return df[['date', 'value']]
    except Exception as e:
        print(f"ECOS fetch failed: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def estimate_elasticity(df: pd.DataFrame) -> dict:
    """Estimate CSI-insurance new contract elasticity.

    Elasticity = (% change in new_contracts) / (% change in CSI)
    Estimated via log-log OLS regression.
    """
    df_clean = df.dropna(subset=['csi', 'new_contracts'])
    df_clean = df_clean[df_clean['csi'] > 0]
    df_clean = df_clean[df_clean['new_contracts'] > 0]

    log_csi = np.log(df_clean['csi'].values)
    log_nc = np.log(df_clean['new_contracts'].values)

    # OLS: log(new_contracts) = alpha + beta * log(csi)
    X = np.column_stack([np.ones(len(log_csi)), log_csi])
    beta = np.linalg.lstsq(X, log_nc, rcond=None)[0]

    # R-squared
    y_hat = X @ beta
    ss_res = np.sum((log_nc - y_hat) ** 2)
    ss_tot = np.sum((log_nc - log_nc.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Correlation
    corr = np.corrcoef(df_clean['csi'], df_clean['new_contracts'])[0, 1]

    return {
        'elasticity': round(beta[1], 4),
        'intercept': round(beta[0], 4),
        'r_squared': round(r2, 4),
        'correlation': round(corr, 4),
        'n_obs': len(df_clean),
    }


def plot_market_overview(df: pd.DataFrame) -> go.Figure:
    """Dual-axis time series: CSI + new contracts."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['csi'],
        name='CSI', line=dict(color='#2E86C1', width=2),
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['new_contracts'],
        name='New Contracts', marker_color='rgba(231,76,60,0.4)',
    ), secondary_y=True)

    fig.update_layout(
        title='Market Context: CSI vs. Insurance New Contracts',
        width=700, height=560,
    )
    fig.update_yaxes(title_text='CSI', secondary_y=False)
    fig.update_yaxes(title_text='New Contracts', secondary_y=True)
    return fig


def plot_elasticity_scatter(df: pd.DataFrame, stats: dict) -> go.Figure:
    """Scatter plot: CSI vs new contracts with regression line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['csi'], y=df['new_contracts'],
        mode='markers', name='Monthly data',
        marker=dict(size=6, color='#2E86C1', opacity=0.6),
    ))

    # Regression line
    x_range = np.linspace(df['csi'].min(), df['csi'].max(), 50)
    y_pred = np.exp(stats['intercept'] + stats['elasticity'] * np.log(x_range))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_pred,
        mode='lines', name=f"Elasticity={stats['elasticity']:.2f}",
        line=dict(color='#E74C3C', dash='dash'),
    ))

    fig.update_layout(
        title=f"CSI-Insurance Elasticity (R²={stats['r_squared']:.3f})",
        xaxis_title='Consumer Sentiment Index',
        yaxis_title='New Contracts',
        width=700, height=560,
    )
    return fig


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    # Load or generate data
    csv_path = DATA_DIR / 'market_context.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"Loaded: {csv_path}")
    else:
        print("Generating sample market data (CSV fallback)...")
        df = generate_sample_market_data()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print(f"Period: {df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}")
    print(f"Observations: {len(df)}\n")

    # Elasticity estimation
    stats = estimate_elasticity(df)
    print("CSI-Insurance Elasticity:")
    print(f"  Elasticity: {stats['elasticity']:.4f}")
    print(f"  R-squared:  {stats['r_squared']:.4f}")
    print(f"  Correlation: {stats['correlation']:.4f}")

    # Plots
    fig_overview = plot_market_overview(df)
    fig_scatter = plot_elasticity_scatter(df, stats)

    fig_overview.show()
    fig_scatter.show()

    fig_dir = PROJECT_ROOT / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_overview.write_image(str(fig_dir / 'market_csi_contracts.png'), scale=2)
    fig_scatter.write_image(str(fig_dir / 'csi_elasticity.png'), scale=2)
    fig_overview.write_html(str(fig_dir / 'market_csi_contracts.html'))
    fig_scatter.write_html(str(fig_dir / 'csi_elasticity.html'))
    print(f"\nSaved 2 figures to {fig_dir}")
