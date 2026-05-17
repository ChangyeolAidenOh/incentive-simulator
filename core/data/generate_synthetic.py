"""
Module 0: Synthetic Agent Performance Data Generator.

Generates realistic insurance agent performance data with domain-driven
correlation structures. Key design principles:
  - Segment assignment drives base distributions
  - Tenure shapes performance trajectory (ramp-up, plateau, decline)
  - Churn signal: 3-month performance decline before exit
  - Commission elasticity varies by segment
  - Variable correlations are intentionally structured for downstream
    causal exploration (DoWhy DAG validation)
"""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SYNTHETIC_DIR, N_AGENTS, RANDOM_SEED, SEGMENT_DIST


def assign_segments(n: int, rng: np.random.Generator) -> np.ndarray:
    """Assign agent segments based on target distribution."""
    segments = list(SEGMENT_DIST.keys())
    probs = list(SEGMENT_DIST.values())
    return rng.choice(segments, size=n, p=probs)


def generate_tenure(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate tenure in months (0-120), skewed toward early career."""
    raw = rng.exponential(scale=30, size=n)
    return np.clip(raw, 0, 120).astype(int)


def generate_monthly_contracts(
    tenure: np.ndarray,
    segments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monthly contract count shaped by tenure trajectory and segment.
    - New agents: rapid ramp-up then plateau
    - Experienced: stable with segment-level variation
    - Star: high base, At-Risk: low base
    """
    n = len(tenure)
    base = np.zeros(n, dtype=float)

    segment_base = {"Star": 12, "Stable": 7, "Developing": 4, "At-Risk": 2}
    for seg, val in segment_base.items():
        mask = segments == seg
        base[mask] = val

    # Tenure effect: ramp-up curve (log-shaped) peaking around 36 months
    tenure_factor = np.log1p(tenure) / np.log1p(36)
    tenure_factor = np.clip(tenure_factor, 0.3, 1.5)

    # Experienced agents (>60 months) slight decline unless Star
    decline_mask = (tenure > 60) & (segments != "Star")
    tenure_factor[decline_mask] *= 0.85

    contracts = base * tenure_factor + rng.normal(0, 1.5, n)
    return np.clip(contracts, 0, None).round(1)


def generate_retention_rate(
    segments: np.ndarray,
    tenure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    13-month retention rate (%).
    Star: 85%+, Stable: 70-85%, Developing: 60-75%, At-Risk: 40-65%.
    Higher tenure generally improves retention.
    """
    n = len(segments)
    base = np.zeros(n, dtype=float)

    segment_params = {
        "Star": (88, 4),
        "Stable": (77, 5),
        "Developing": (67, 6),
        "At-Risk": (52, 8),
    }
    for seg, (mu, sigma) in segment_params.items():
        mask = segments == seg
        count = mask.sum()
        base[mask] = rng.normal(mu, sigma, count)

    # Tenure bonus: experienced agents retain better (diminishing)
    tenure_bonus = np.log1p(tenure) * 0.8
    base += tenure_bonus

    return np.clip(base, 20, 99).round(1)


def generate_commission_rate(
    segments: np.ndarray,
    tenure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Commission rate (%) — tiered by segment with tenure adjustment.
    Represents first-year commission as % of premium.
    """
    n = len(segments)
    rates = np.zeros(n, dtype=float)

    segment_rates = {
        "Star": (45, 3),
        "Stable": (38, 3),
        "Developing": (32, 3),
        "At-Risk": (28, 4),
    }
    for seg, (mu, sigma) in segment_rates.items():
        mask = segments == seg
        count = mask.sum()
        rates[mask] = rng.normal(mu, sigma, count)

    # Tenure adjustment: gradual increase with experience
    tenure_adj = np.minimum(tenure / 120 * 5, 5)
    rates += tenure_adj

    return np.clip(rates, 15, 60).round(1)


def generate_customer_satisfaction(
    segments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Customer satisfaction score (1-10 scale, normal distribution)."""
    n = len(segments)
    scores = np.zeros(n, dtype=float)

    segment_params = {
        "Star": (8.5, 0.8),
        "Stable": (7.2, 1.0),
        "Developing": (6.5, 1.2),
        "At-Risk": (5.5, 1.5),
    }
    for seg, (mu, sigma) in segment_params.items():
        mask = segments == seg
        count = mask.sum()
        scores[mask] = rng.normal(mu, sigma, count)

    return np.clip(scores, 1, 10).round(1)


def generate_cross_sell_count(
    segments: np.ndarray,
    tenure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cross-sell count (Poisson). Star agents cross-sell more."""
    n = len(segments)
    counts = np.zeros(n, dtype=int)

    segment_lambda = {"Star": 3.5, "Stable": 1.8, "Developing": 0.8, "At-Risk": 0.3}
    for seg, lam in segment_lambda.items():
        mask = segments == seg
        count = mask.sum()
        # Tenure boost for cross-selling skill
        tenure_mult = np.clip(tenure[mask] / 48, 0.5, 2.0)
        adj_lam = lam * tenure_mult
        counts[mask] = rng.poisson(adj_lam)

    return counts


def generate_complaint_count(
    segments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Complaint count (Poisson). At-Risk agents have higher complaints."""
    n = len(segments)
    counts = np.zeros(n, dtype=int)

    segment_lambda = {"Star": 0.2, "Stable": 0.5, "Developing": 1.0, "At-Risk": 2.0}
    for seg, lam in segment_lambda.items():
        mask = segments == seg
        count = mask.sum()
        counts[mask] = rng.poisson(lam, count)

    return counts


def generate_training_hours(
    tenure: np.ndarray,
    segments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monthly training hours.
    New agents: high (onboarding), experienced: lower.
    Star agents maintain moderate training (self-development).
    """
    n = len(tenure)
    hours = np.zeros(n, dtype=float)

    # Base: inversely related to tenure
    base = 20 * np.exp(-tenure / 24) + 2
    hours = base + rng.normal(0, 2, n)

    # Star agents maintain higher training
    star_mask = segments == "Star"
    hours[star_mask] += 3

    return np.clip(hours, 0, 40).round(1)


def generate_churn_labels(
    segments: np.ndarray,
    tenure: np.ndarray,
    retention_rate: np.ndarray,
    satisfaction: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    12-month churn label (binary).
    Base probability by segment, modified by retention rate and satisfaction.
    Domain pattern: At-Risk with low retention and low satisfaction churn most.
    """
    n = len(segments)
    base_prob = np.zeros(n, dtype=float)

    segment_churn = {"Star": 0.03, "Stable": 0.08, "Developing": 0.18, "At-Risk": 0.35}
    for seg, prob in segment_churn.items():
        mask = segments == seg
        base_prob[mask] = prob

    # Retention rate effect: lower retention -> higher churn
    retention_effect = (80 - retention_rate) / 100
    retention_effect = np.clip(retention_effect, -0.1, 0.2)

    # Satisfaction effect: lower satisfaction -> higher churn
    satisfaction_effect = (7 - satisfaction) / 20
    satisfaction_effect = np.clip(satisfaction_effect, -0.05, 0.15)

    # Early tenure churn risk (first 12 months)
    early_risk = np.where(tenure < 12, 0.08, 0)

    final_prob = base_prob + retention_effect + satisfaction_effect + early_risk
    final_prob = np.clip(final_prob, 0.01, 0.70)

    return (rng.random(n) < final_prob).astype(int)


def inject_pre_churn_signal(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Domain pattern: churned agents show 3-month performance decline.
    Reduce monthly_contracts and satisfaction for churned agents
    to simulate the pre-churn decline signal.
    """
    churned = df["churned_12m"] == 1
    n_churned = churned.sum()

    # Performance decline: 20-50% drop
    decline_factor = rng.uniform(0.5, 0.8, n_churned)
    df.loc[churned, "monthly_contracts"] = (
        df.loc[churned, "monthly_contracts"] * decline_factor
    ).round(1)

    # Satisfaction decline: 0.5-2.0 point drop
    sat_drop = rng.uniform(0.5, 2.0, n_churned)
    df.loc[churned, "customer_satisfaction"] = np.clip(
        df.loc[churned, "customer_satisfaction"] - sat_drop, 1, 10
    ).round(1)

    # Cross-sell decline
    df.loc[churned, "cross_sell_count"] = np.maximum(
        0, df.loc[churned, "cross_sell_count"] - rng.integers(0, 2, n_churned)
    )

    return df


def generate_monthly_premium(
    monthly_contracts: np.ndarray,
    segments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monthly premium volume (KRW, 10K units).
    Derived from contract count x average premium per contract.
    Star agents sell higher-value products.
    """
    avg_premium = {"Star": 350, "Stable": 250, "Developing": 180, "At-Risk": 120}
    n = len(segments)
    premium = np.zeros(n, dtype=float)

    for seg, avg in avg_premium.items():
        mask = segments == seg
        noise = rng.normal(1.0, 0.15, mask.sum())
        premium[mask] = monthly_contracts[mask] * avg * noise

    return np.clip(premium, 0, None).round(0)


def generate_dataset(n: int = N_AGENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic agent performance dataset."""
    rng = np.random.default_rng(seed)

    segments = assign_segments(n, rng)
    tenure = generate_tenure(n, rng)
    contracts = generate_monthly_contracts(tenure, segments, rng)
    retention = generate_retention_rate(segments, tenure, rng)
    commission = generate_commission_rate(segments, tenure, rng)
    satisfaction = generate_customer_satisfaction(segments, rng)
    cross_sell = generate_cross_sell_count(segments, tenure, rng)
    complaints = generate_complaint_count(segments, rng)
    training = generate_training_hours(tenure, segments, rng)

    churned = generate_churn_labels(
        segments, tenure, retention, satisfaction, rng
    )

    df = pd.DataFrame({
        "agent_id": [f"AG{i:05d}" for i in range(n)],
        "segment": segments,
        "tenure_months": tenure,
        "monthly_contracts": contracts,
        "retention_rate_13m": retention,
        "commission_rate": commission,
        "customer_satisfaction": satisfaction,
        "cross_sell_count": cross_sell,
        "complaint_count": complaints,
        "training_hours": training,
        "churned_12m": churned,
    })

    # Inject pre-churn performance decline signal
    df = inject_pre_churn_signal(df, rng)

    # Derived: monthly premium volume
    df["monthly_premium_10k"] = generate_monthly_premium(
        df["monthly_contracts"].values, df["segment"].values, rng
    )

    # Derived: estimated commission cost (premium x commission rate)
    df["commission_cost_10k"] = (
        df["monthly_premium_10k"] * df["commission_rate"] / 100
    ).round(0)

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run sanity checks on generated dataset."""
    checks = {}

    # Segment distribution
    actual_dist = df["segment"].value_counts(normalize=True).to_dict()
    checks["segment_distribution"] = actual_dist

    # Churn rate by segment
    churn_by_seg = df.groupby("segment")["churned_12m"].mean().to_dict()
    checks["churn_rate_by_segment"] = churn_by_seg

    # Correlation: satisfaction vs churn (should be negative)
    corr = df["customer_satisfaction"].corr(df["churned_12m"])
    checks["satisfaction_churn_corr"] = round(corr, 3)

    # Mean contracts by segment (Star should be highest)
    contracts_by_seg = (
        df.groupby("segment")["monthly_contracts"].mean().round(1).to_dict()
    )
    checks["mean_contracts_by_segment"] = contracts_by_seg

    # Retention by segment
    retention_by_seg = (
        df.groupby("segment")["retention_rate_13m"].mean().round(1).to_dict()
    )
    checks["mean_retention_by_segment"] = retention_by_seg

    return checks


def main():
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic agent performance data...")
    df = generate_dataset()
    print(f"  Agents: {len(df)}")

    out_path = SYNTHETIC_DIR / "agent_performance.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print("\nValidation:")
    checks = validate_dataset(df)
    for key, val in checks.items():
        print(f"  {key}: {val}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nChurn rate: {df['churned_12m'].mean():.1%}")


if __name__ == "__main__":
    main()
