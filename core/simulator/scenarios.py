"""Module 3: Incentive Scenario Simulator - Scenario Definitions

Defines commission-rate adjustment scenarios per segment
and a function to apply them to agent data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENT_ORDER = ['Star', 'Stable', 'Developing', 'At-Risk']

# Each scenario maps segment -> commission_rate multiplier
# 1.0 = no change, 1.10 = +10%, 0.90 = -10%
SCENARIOS = {
    'Baseline': {
        'Star': 1.00, 'Stable': 1.00, 'Developing': 1.00, 'At-Risk': 1.00,
    },
    'Uniform +10%': {
        'Star': 1.10, 'Stable': 1.10, 'Developing': 1.10, 'At-Risk': 1.10,
    },
    'Target At-Risk': {
        'Star': 1.00, 'Stable': 1.00, 'Developing': 1.05, 'At-Risk': 1.20,
    },
    'Reward Stars': {
        'Star': 1.15, 'Stable': 1.05, 'Developing': 1.00, 'At-Risk': 1.00,
    },
    'Balanced Growth': {
        'Star': 1.05, 'Stable': 1.03, 'Developing': 1.10, 'At-Risk': 1.15,
    },
    'Cost Reduction': {
        'Star': 0.95, 'Stable': 0.95, 'Developing': 0.90, 'At-Risk': 0.85,
    },
}


def apply_scenario(df: pd.DataFrame, scenario: dict,
                   segment_col: str = 'segment') -> pd.DataFrame:
    """Apply commission multipliers and recalculate derived features.

    Args:
        df: Agent data with segment, commission_rate, monthly_premium_10k.
        scenario: Dict mapping segment names to multipliers.
        segment_col: Column used for segment lookup.

    Returns:
        Modified copy of df with updated commission_rate and commission_cost_10k.
    """
    df_mod = df.copy()
    for seg, mult in scenario.items():
        mask = df_mod[segment_col] == seg
        df_mod.loc[mask, 'commission_rate'] *= mult

    # Recalculate derived cost
    df_mod['commission_cost_10k'] = (
        df_mod['commission_rate'] / 100 * df_mod['monthly_premium_10k']
    )
    return df_mod


def describe_scenario(name: str, scenario: dict) -> str:
    """One-line description for display."""
    changes = []
    for seg in SEGMENT_ORDER:
        mult = scenario.get(seg, 1.0)
        if mult != 1.0:
            pct = (mult - 1) * 100
            changes.append(f"{seg} {pct:+.0f}%")
    return f"{name}: {', '.join(changes) if changes else 'no change'}"


if __name__ == '__main__':
    print("Defined scenarios:")
    for name, sc in SCENARIOS.items():
        print(f"  {describe_scenario(name, sc)}")
