"""Module 1: Agent Profiler - Clustering

Compares three clustering approaches on agent performance features:
  - HDBSCAN (density-based, no k required)
  - KMeans  (centroid-based, k=4)
  - GMM     (probabilistic, k=4)

Selects the best method by ARI against true segments,
then saves clustered data and model artifacts.

Insight: agent performance is a continuum — density-based methods
struggle because segments overlap without sharp density boundaries.
Centroid/probabilistic methods with known k outperform here.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_FEATURES = [
    'tenure_months', 'monthly_contracts', 'retention_rate_13m',
    'commission_rate', 'customer_satisfaction', 'cross_sell_count',
    'complaint_count', 'training_hours', 'monthly_premium_10k',
]

SEGMENT_ORDER = ['Star', 'Stable', 'Developing', 'At-Risk']
N_SEGMENTS = len(SEGMENT_ORDER)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_agent_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = PROJECT_ROOT / 'data' / 'synthetic' / 'agent_performance.csv'
    return pd.read_csv(path)


def scale_features(df: pd.DataFrame) -> tuple:
    """StandardScaler on cluster features. Returns (scaler, X_scaled)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES].values)
    return scaler, X_scaled


# ------------------------------------------------------------------
# Clustering methods
# ------------------------------------------------------------------

def fit_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 30,
                min_samples: int = 3) -> tuple:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(X_scaled)
    return clusterer, labels


def fit_kmeans(X_scaled: np.ndarray, k: int = N_SEGMENTS,
               random_state: int = 42) -> tuple:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    return model, labels


def fit_gmm(X_scaled: np.ndarray, k: int = N_SEGMENTS,
            random_state: int = 42) -> tuple:
    model = GaussianMixture(
        n_components=k, covariance_type='full',
        n_init=5, random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def map_clusters_to_segments(true_segments: pd.Series,
                             labels: np.ndarray) -> dict:
    """Map clusters to segments via Hungarian algorithm (1-to-1)."""
    from scipy.optimize import linear_sum_assignment

    temp = pd.DataFrame({'segment': true_segments.values, 'cluster': labels})
    valid = temp[temp['cluster'] != -1]

    cluster_ids = sorted(valid['cluster'].unique())
    segments = SEGMENT_ORDER

    # Build cost matrix: -count(cluster_i ∩ segment_j)
    cost = np.zeros((len(cluster_ids), len(segments)))
    for i, cid in enumerate(cluster_ids):
        cluster_mask = valid['cluster'] == cid
        for j, seg in enumerate(segments):
            cost[i, j] = -(cluster_mask & (valid['segment'] == seg)).sum()

    row_idx, col_idx = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_idx, col_idx):
        mapping[cluster_ids[r]] = segments[c]
    return mapping

def evaluate(true_segments: pd.Series, X_scaled: np.ndarray,
             labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    valid = labels != -1
    k = len(set(labels) - {-1})
    metrics = {
        'n_clusters': k,
        'noise_ratio': round(float((~valid).mean()), 4),
    }
    if valid.sum() > 1 and k > 1:
        metrics['silhouette'] = round(
            silhouette_score(X_scaled[valid], labels[valid]), 4)
        metrics['ari'] = round(
            adjusted_rand_score(true_segments[valid], labels[valid]), 4)
    else:
        metrics['silhouette'] = -1.0
        metrics['ari'] = -1.0
    return metrics


def compare_methods(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """Run all three methods and return comparison table."""
    results = []

    for name, fit_fn in [
        ('HDBSCAN', lambda X: fit_hdbscan(X)),
        ('KMeans',  lambda X: fit_kmeans(X)),
        ('GMM',     lambda X: fit_gmm(X)),
    ]:
        _, labels = fit_fn(X_scaled)
        m = evaluate(df['segment'], X_scaled, labels)
        m['method'] = name
        results.append(m)

    comp = pd.DataFrame(results)[['method', 'n_clusters', 'noise_ratio',
                                   'silhouette', 'ari']]
    return comp


# ------------------------------------------------------------------
# Segment KPI
# ------------------------------------------------------------------

def segment_kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    kpi = df.groupby('segment').agg(
        count=('agent_id', 'count'),
        avg_contracts=('monthly_contracts', 'mean'),
        avg_retention=('retention_rate_13m', 'mean'),
        avg_satisfaction=('customer_satisfaction', 'mean'),
        churn_rate=('churned_12m', 'mean'),
        avg_commission=('commission_rate', 'mean'),
        avg_premium=('monthly_premium_10k', 'mean'),
    ).round(3)
    return kpi.reindex(SEGMENT_ORDER)


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

def save_results(df: pd.DataFrame, labels: np.ndarray, cluster_map: dict,
                 scaler, model, method_name: str) -> pd.DataFrame:
    df_out = df.copy()
    df_out['cluster'] = labels
    df_out['cluster_segment'] = (
        df_out['cluster'].map(cluster_map).fillna('Noise')
    )

    out_path = PROJECT_ROOT / 'data' / 'processed' / 'agent_clustered.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    model_dir = PROJECT_ROOT / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'profiler_clusterer.joblib')
    joblib.dump(scaler, model_dir / 'profiler_scaler.joblib')

    return df_out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    df = load_agent_data()
    print(f"Loaded {len(df)} agents\n")

    scaler, X_scaled = scale_features(df)

    # --- Method comparison ---
    comp = compare_methods(df, X_scaled)
    print("Method comparison:")
    print(comp.to_string(index=False))

    best_method = comp.loc[comp['ari'].idxmax(), 'method']
    print(f"\nBest by ARI: {best_method}")

    # --- Fit best method ---
    if best_method == 'HDBSCAN':
        model, labels = fit_hdbscan(X_scaled)
    elif best_method == 'KMeans':
        model, labels = fit_kmeans(X_scaled)
    else:
        model, labels = fit_gmm(X_scaled)

    cluster_map = map_clusters_to_segments(df['segment'], labels)
    metrics = evaluate(df['segment'], X_scaled, labels)

    print(f"\n{best_method} result:")
    print(f"  Clusters: {metrics['n_clusters']}")
    print(f"  Noise: {metrics['noise_ratio']:.1%}")
    print(f"  Silhouette: {metrics['silhouette']:.3f}")
    print(f"  ARI: {metrics['ari']:.3f}")
    print(f"  Cluster->Segment: {cluster_map}")

    kpi = segment_kpi_summary(df)
    print(f"\nSegment KPI:\n{kpi}")

    df_out = save_results(df, labels, cluster_map, scaler, model, best_method)
    print(f"\nSaved: data/processed/agent_clustered.csv ({len(df_out)} rows)")
    print(f"Saved: models/profiler_clusterer.joblib ({best_method})")
