"""Module 3: Incentive Scenario Simulator - Monte Carlo Engine

For each scenario, bootstrap-samples agents, applies commission changes,
predicts churn probabilities, and aggregates cost/retention/productivity.
Repeats N times to produce result distributions with median + 95% CI.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor.baseline import FEATURES, TARGET
from core.simulator.scenarios import SCENARIOS, apply_scenario, SEGMENT_ORDER


N_ITERATIONS = 10_000
RANDOM_SEED = 42


def load_assets() -> tuple:
    """Load agent data and best prediction model."""
    df = pd.read_csv(
        PROJECT_ROOT / 'data' / 'synthetic' / 'agent_performance.csv',
    )
    model = joblib.load(PROJECT_ROOT / 'models' / 'best_tree_model.joblib')
    return df, model


def simulate_scenario(
    df: pd.DataFrame,
    model,
    scenario: dict,
    n_iter: int = N_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run Monte Carlo for a single scenario.

    Returns DataFrame with one row per iteration:
        retention_rate, total_cost, productivity, net_value
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    records = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx].reset_index(drop=True)

        sample_mod = apply_scenario(sample, scenario)

        X = sample_mod[FEATURES]
        churn_prob = model.predict_proba(X)[:, 1]
        retain_prob = 1 - churn_prob

        retention_rate = retain_prob.mean()
        total_cost = sample_mod['commission_cost_10k'].sum()
        productivity = (retain_prob * sample_mod['monthly_contracts']).sum()
        net_value = productivity - total_cost / 1000  # normalize scale

        records.append({
            'retention_rate': retention_rate,
            'total_cost': total_cost,
            'productivity': productivity,
            'net_value': net_value,
        })

    return pd.DataFrame(records)


def summarize(dist: pd.DataFrame) -> dict:
    """Compute median and 95% CI for each metric."""
    summary = {}
    for col in dist.columns:
        vals = dist[col].values
        summary[col] = {
            'median': np.median(vals),
            'ci_lower': np.percentile(vals, 2.5),
            'ci_upper': np.percentile(vals, 97.5),
        }
    return summary


def run_all_scenarios(
    df: pd.DataFrame,
    model,
    scenarios: dict = None,
    n_iter: int = N_ITERATIONS,
) -> tuple:
    """Run MC for all scenarios. Returns (summary_table, raw_distributions)."""
    if scenarios is None:
        scenarios = SCENARIOS

    all_dist = {}
    rows = []

    for name, sc in scenarios.items():
        print(f"  Simulating: {name} ({n_iter:,} iterations)...")
        dist = simulate_scenario(df, model, sc, n_iter=n_iter)
        all_dist[name] = dist

        s = summarize(dist)
        row = {'scenario': name}
        for metric, vals in s.items():
            row[f'{metric}_med'] = round(vals['median'], 4)
            row[f'{metric}_lo'] = round(vals['ci_lower'], 4)
            row[f'{metric}_hi'] = round(vals['ci_upper'], 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df, all_dist


def save_results(summary_df: pd.DataFrame, all_dist: dict):
    """Save summary table and raw distributions."""
    out_dir = PROJECT_ROOT / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / 'scenario_summary.csv', index=False)

    for name, dist in all_dist.items():
        safe_name = name.lower().replace(' ', '_').replace('+', 'plus')
        dist.to_csv(out_dir / f'mc_dist_{safe_name}.csv', index=False)


if __name__ == '__main__':
    df, model = load_assets()
    print(f"Agents: {len(df)} | Model: {type(model).__name__}")
    print(f"Scenarios: {len(SCENARIOS)} | Iterations: {N_ITERATIONS:,}\n")

    summary_df, all_dist = run_all_scenarios(df, model)

    print("\nScenario summary (median [95% CI]):")
    for _, row in summary_df.iterrows():
        print(f"\n  {row['scenario']}:")
        print(f"    Retention: {row['retention_rate_med']:.3f} "
              f"[{row['retention_rate_lo']:.3f}, {row['retention_rate_hi']:.3f}]")
        print(f"    Cost:      {row['total_cost_med']:,.0f} "
              f"[{row['total_cost_lo']:,.0f}, {row['total_cost_hi']:,.0f}]")
        print(f"    Product:   {row['productivity_med']:,.1f} "
              f"[{row['productivity_lo']:,.1f}, {row['productivity_hi']:,.1f}]")

    save_results(summary_df, all_dist)
    print(f"\nSaved: data/processed/scenario_summary.csv")
    print(f"Saved: data/processed/mc_dist_*.csv ({len(all_dist)} files)")
