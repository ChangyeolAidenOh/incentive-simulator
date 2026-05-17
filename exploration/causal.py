"""Exploration A: Causal Structure Analysis (DoWhy)

DAG construction + Backdoor ATE estimation + Refutation tests.

IMPORTANT: ATE values from synthetic data have no empirical validity
(circular reasoning risk). The value of this module is:
  1. DAG structuring methodology
  2. Confounder identification
  3. Refutation test design for real-data transition
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from dowhy import CausalModel

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# DAG definition (GML format for DoWhy)
# ------------------------------------------------------------------

GML_GRAPH = """
graph [
  directed 1

  node [ id "tenure_months"        label "tenure_months" ]
  node [ id "commission_rate"      label "commission_rate" ]
  node [ id "customer_satisfaction" label "customer_satisfaction" ]
  node [ id "monthly_contracts"    label "monthly_contracts" ]
  node [ id "training_hours"       label "training_hours" ]
  node [ id "complaint_count"      label "complaint_count" ]
  node [ id "cross_sell_count"     label "cross_sell_count" ]
  node [ id "churned_12m"          label "churned_12m" ]

  edge [ source "tenure_months"        target "commission_rate" ]
  edge [ source "tenure_months"        target "monthly_contracts" ]
  edge [ source "tenure_months"        target "churned_12m" ]
  edge [ source "commission_rate"      target "churned_12m" ]
  edge [ source "customer_satisfaction" target "churned_12m" ]
  edge [ source "customer_satisfaction" target "monthly_contracts" ]
  edge [ source "monthly_contracts"    target "churned_12m" ]
  edge [ source "training_hours"       target "customer_satisfaction" ]
  edge [ source "complaint_count"      target "churned_12m" ]
  edge [ source "cross_sell_count"     target "churned_12m" ]
]
"""

TREATMENT = 'commission_rate'
OUTCOME = 'churned_12m'


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    path = PROJECT_ROOT / 'data' / 'synthetic' / 'agent_performance.csv'
    return pd.read_csv(path)


def build_causal_model(df: pd.DataFrame) -> CausalModel:
    """Build DoWhy CausalModel with domain-knowledge DAG."""
    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=GML_GRAPH,
    )
    return model


def identify_and_estimate(model: CausalModel) -> tuple:
    """Identify effect via backdoor criterion, estimate via linear regression."""
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"Identified estimand:\n{identified}\n")

    estimate = model.estimate_effect(
        identified,
        method_name='backdoor.linear_regression',
    )
    print(f"ATE estimate: {estimate.value:.6f}")
    print(f"  (commission_rate -> churned_12m)")
    return identified, estimate


def run_refutation_tests(model: CausalModel, identified, estimate) -> dict:
    """Run 3 refutation tests to assess estimate robustness."""
    results = {}

    # 1. Placebo treatment
    print("\n[Refutation 1] Placebo treatment...")
    ref_placebo = model.refute_estimate(
        identified, estimate,
        method_name='placebo_treatment_refuter',
        placebo_type='permute',
        num_simulations=100,
    )
    results['placebo'] = ref_placebo
    print(f"  {ref_placebo}")

    # 2. Random common cause
    print("[Refutation 2] Random common cause...")
    ref_random = model.refute_estimate(
        identified, estimate,
        method_name='random_common_cause',
        num_simulations=100,
    )
    results['random_common_cause'] = ref_random
    print(f"  {ref_random}")

    # 3. Data subset
    print("[Refutation 3] Data subset...")
    ref_subset = model.refute_estimate(
        identified, estimate,
        method_name='data_subset_refuter',
        subset_fraction=0.8,
        num_simulations=100,
    )
    results['data_subset'] = ref_subset
    print(f"  {ref_subset}")

    return results


def save_dag_plot(model: CausalModel, output_path: Path = None):
    """Save DAG visualization."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'figures' / 'causal_dag.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model.view_model()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"DAG plot skipped ({e}). Install graphviz if needed.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    df = load_data()
    print(f"Loaded {len(df)} agents\n")

    print("=" * 50)
    print("CAUSAL STRUCTURE ANALYSIS")
    print("=" * 50)
    print("NOTE: Synthetic data -- ATE values are NOT empirically valid.")
    print("      Focus: DAG structure + confounder identification.\n")

    model = build_causal_model(df)
    identified, estimate = identify_and_estimate(model)
    refutation_results = run_refutation_tests(model, identified, estimate)

    save_dag_plot(model)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Treatment: {TREATMENT} -> Outcome: {OUTCOME}")
    print(f"Backdoor ATE: {estimate.value:.6f}")
    print(f"\nLIMITATION: Synthetic-data artifact.")
    print(f"  Real-data transition requires:")
    print(f"    1. Domain expert DAG validation")
    print(f"    2. Sensitivity analysis for unobserved confounders")
    print(f"    3. IV identification (if exclusion restriction holds)")
