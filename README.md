# Agent Incentive Scenario Simulator

> **Positioning:** Prototype decision-support tool that validates sensitivity and trade-offs of incentive policy changes in a synthetic operational environment

---

## Project Overview

**Starting Question:** "Can we compare productivity, retention, and cost trade-offs of different incentive policy changes on a single screen?"

This project builds an end-to-end pipeline from agent performance profiling -> churn prediction -> incentive scenario simulation -> AI executive report, using synthetic agent data modeled on insurance industry domain knowledge.

**Boundary:** This is a sensitivity analysis framework, not a causal claims engine. It does not propose an optimal incentive plan - it shows "under this scenario, these trade-offs emerge" and designs the validation process for real-data transition.

---

## Architecture

```
[Synthetic Agent Data (5,000 agents)]
         |
         v
  Module 1: Agent Profiler
  HDBSCAN vs KMeans vs GMM --> KMeans selected (ARI 0.337)
  Hungarian algorithm for 1:1 cluster-segment mapping
  UMAP 2D visualization
         |
         v
  Module 2: Retention & Productivity Predictor
  Baseline: Logistic Regression (ROC-AUC 0.9225)
  Advanced: XGBoost + LightGBM + CatBoost Stacking (0.9177)
  TreeSHAP + PDP --> commission_rate effect is flat
         |
         v
  Module 3: Incentive Scenario Simulator
  6 scenarios x Monte Carlo 10,000 runs
  Median + 95% CI for retention, cost, productivity
  Cost-performance Pareto frontier
  Key finding: retention varies only ~3pp across all scenarios
         |
         v
  Module 4: AI Report Generator
  Mock (dev) / Ollama Qwen 2.5 (test) / Anthropic Haiku (prod)
  1-page Korean PDF executive report via ReportLab
         |
         v
  [Streamlit Dashboard — 6 tabs]

  [Exploration Modules — separate tabs]
  Exp-A: DoWhy causal DAG + Backdoor ATE + Refutation tests
  Exp-B: ECOS CSI + insurance new contract elasticity
```

---

## Directory Structure

```
Incentive_simulator/
|-- config.py
|-- requirements.txt
|-- .gitignore
|-- README.md
|
|-- core/
|   |-- data/
|   |   |-- generate_synthetic.py
|   |
|   |-- profiler/                     # Module 1
|   |   |-- clustering.py             # HDBSCAN/KMeans/GMM comparison
|   |   |-- visualization.py          # UMAP + KPI charts
|   |
|   |-- predictor/                    # Module 2
|   |   |-- baseline.py               # Logistic Regression
|   |   |-- stacking.py               # XGB + LGBM + CatBoost stacking
|   |   |-- explainer.py              # TreeSHAP + PDP
|   |
|   |-- simulator/                    # Module 3
|   |   |-- scenarios.py              # Scenario definitions
|   |   |-- monte_carlo.py            # MC 10,000 simulation engine
|   |   |-- pareto.py                 # Pareto frontier visualization
|   |
|   |-- report/                       # Module 4
|       |-- llm_backend.py            # Mock/Ollama/Haiku abstraction
|       |-- pdf_generator.py          # ReportLab Korean PDF
|
|-- exploration/
|   |-- causal.py                     # DoWhy DAG + ATE + Refutation
|   |-- market_context.py             # ECOS CSI elasticity
|
|-- dashboard/
|   |-- app.py                        # Streamlit entrypoint
|   |-- tabs/
|       |-- tab_agent_profile.py
|       |-- tab_retention.py
|       |-- tab_simulator.py
|       |-- tab_comparison.py
|       |-- tab_causal.py
|       |-- tab_market.py
|
|-- data/
|   |-- raw/                          # ECOS market data
|   |-- synthetic/                    # Generated agent data
|   |-- processed/                    # Pipeline outputs
|
|-- models/                           # Saved model artifacts
|-- figures/                          # Exported visualizations
|-- notebooks/
|-- docs/
|-- tests/
```

---

## Implementation Process & Key Decisions

### Phase 1: Synthetic Data Generation

Generated 5,000 synthetic insurance agents with domain-informed distributions:

| Variable | Design Logic |
|---|---|
| `segment` | Star(10%) / Stable(50%) / Developing(25%) / At-Risk(15%) |
| `churned_12m` | Segment-based probability: Star 1%, Stable 10.8%, Developing 34.9%, At-Risk 61.2% |
| `customer_satisfaction` | Normal distribution, negatively correlated with churn (r = -0.585) |
| `monthly_contracts` | Star ~9.7, Stable ~5.4, Developing ~2.8, At-Risk ~1.3 |

Overall churn rate: 23.1% - realistic for insurance agent turnover.

---

### Phase 2: Agent Profiling: Clustering Method Selection

#### Problem: HDBSCAN Failed Catastrophically

Initial HDBSCAN run: **68.5% noise, 2 clusters, both mapped to Stable, ARI 0.127**.

A full grid search over `min_cluster_size` (30–120), `min_samples` (3–10), and `cluster_selection_method` (eom/leaf), 24 combinations total, confirmed this was not a hyperparameter issue:

| Best HDBSCAN (size=30, samp=10, eom) | Result |
|---|---|
| Clusters | 3 |
| Noise | 67.6% |
| ARI | 0.167 (best across all 24 combinations) |

#### Diagnosis

UMAP visualization revealed the root cause: agent performance features form a **continuous spectrum** across segments, not density-isolated clusters. Star separates clearly, but Stable -> Developing -> At-Risk blend into each other with no sharp density boundaries. This is realistic -> insurance agent performance IS a continuum.

#### Decision: 3-Method Comparison

Replaced HDBSCAN-only with a comparison framework:

| Method | Clusters | Noise | Silhouette | ARI |
|---|---|---|---|---|
| HDBSCAN | 4 | 39.1% | 0.039 | 0.148 |
| **KMeans** | **4** | **0%** | **0.224** | **0.337** |
| GMM | 4 | 0% | 0.088 | 0.118 |

**Selected: KMeans** - highest ARI, zero noise, known k=4 from domain knowledge.

#### Problem: Majority Vote Mapping Collision

KMeans clusters 1 and 2 both mapped to "Stable" via majority vote, leaving Developing unmapped.

**Solution:** Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for 1-to-1 optimal mapping. Result: Star, Stable, Developing, At-Risk each uniquely assigned.

**Portfolio insight:** The HDBSCAN failure itself is a valuable result: It demonstrates that "density-based methods struggle when the data generating process produces overlapping continuous distributions" and shows analytical maturity in method selection.

---

### Phase 3: Churn Prediction

#### Result: Baseline Beat the Ensemble

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.9225** | **0.7319** | 0.6293 | **0.8745** |
| Stacking (XGB+LGBM+CAT) | 0.9177 | 0.7024 | **0.6949** | 0.7100 |
| CatBoost | 0.9176 | 0.7017 | 0.6192 | 0.8095 |
| XGBoost | 0.9139 | 0.7091 | 0.6389 | 0.7965 |
| LightGBM | 0.9113 | 0.7093 | 0.6421 | 0.7922 |

Stacking CV ROC-AUC: 0.9197 ± 0.0107

#### Interpretation

LR outperforming Stacking is **expected and informative**, not a failure. The synthetic data's churn signal is largely linear (satisfaction and contracts dominate), so LR captures it efficiently. Stacking adds complexity without benefit here, but would outperform on real data with non-linear interaction patterns.

#### SHAP Analysis - Key Finding

Top features by mean |SHAP|:

| Feature | mean |SHAP| |
|---|---|
| customer_satisfaction | 1.45 |
| monthly_contracts | 1.37 |
| retention_rate_13m | 0.29 |
| commission_rate | 0.26 |

**Critical insight:** `customer_satisfaction` and `monthly_contracts` together account for ~80% of the model's predictive power. `commission_rate` ranks 4th with 5x less impact.

#### PDP Confirmation

PDP plots show:
- `customer_satisfaction`: strong monotonic decrease (higher → less churn)
- `monthly_contracts`: sharp drop at 2–3 contracts
- `commission_rate`: **nearly flat** - changing commission barely moves churn probability

This PDP finding directly motivates Module 3's key question: **if commission changes don't move churn, what do incentive scenarios actually achieve?**

---

### Phase 4: Scenario Simulation: The Core Insight

#### 6 Scenarios × 10,000 MC Iterations

| Scenario | Retention (median) | Cost (median) | vs Baseline |
|---|---|---|---|
| Baseline | 68.0% | 2,368,303 | — |
| Uniform +10% | 66.3% | 2,605,133 | -1.7pp / +10.0% cost |
| Target At-Risk | 67.2% | 2,385,485 | -0.8pp / +0.7% cost |
| Reward Stars | 67.6% | 2,552,323 | -0.4pp / +7.8% cost |
| Balanced Growth | 66.7% | 2,473,292 | -1.3pp / +4.4% cost |
| Cost Reduction | 69.3% | 2,236,236 | +1.3pp / -5.6% cost |

#### Key Finding

**Retention varies only ~3pp across all 6 scenarios** (66.3%–69.3%), while **cost varies from -5.6% to +10%**. Commission changes primarily move cost, not retention.

The Pareto frontier shows Cost Reduction dominating (lowest cost, highest retention): a synthetic data artifact where the model learned a weak inverse correlation. In real data, commission cuts would likely increase churn through causal pathways not captured here.

#### Executive Message

> "A 10% across-the-board commission increase adds ~10% to cost but improves retention by less than 2 percentage points. Commission structure changes alone are insufficient for retention management; they must be paired with satisfaction improvement programs and performance support systems."

---

### Phase 5: AI Report Generation

Three-tier LLM backend following the multi-backend abstraction pattern:

| Backend | Use Case | Cost |
|---|---|---|
| MockBackend | Development, CI/CD | Free |
| OllamaBackend (Qwen 2.5) | Testing, iteration | Free (local) |
| HaikuBackend (Claude Haiku) | Production executive report | ~$0.01/report |

Output: 1-page Korean PDF with scenario table, LLM trade-off analysis, and embedded Pareto chart.

---

### Phase 6: Exploration Modules

#### Exploration A - Causal Structure (DoWhy)

| Item | Result |
|---|---|
| Backdoor variable | `tenure_months` (confounder) |
| ATE | -0.00196 (near zero, consistent with PDP) |
| Placebo test | New effect ≈ 0, p = 0.82 — passed |
| Random common cause | Effect unchanged, p = 0.84 — passed |
| Data subset (80%) | Effect unchanged, p = 0.92 — passed |

**Limitation explicitly stated:** ATE from synthetic data is circular (the data was generated with these relationships). The value is the DAG structuring methodology and refutation test design for real-data transition.

**IV decision:** `tenure_months` as IV was considered and rejected - exclusion restriction not satisfied (tenure directly affects churn, not only through commission).

#### Exploration B - Market Context

CSI-Insurance new contract elasticity: **1.00** (unit elastic), R² = 0.495, correlation = 0.70.

Estimated independently from ECOS + insurance market data (not transferred from sportswear project's 3.98 elasticity).

---

## Dashboard

Streamlit 6-tab dashboard:

| Tab | Content |
|---|---|
| **Agent Profile** | UMAP (interactive Plotly), segment metrics, KPI radar |
| **Retention Predictor** | Model comparison, Top-20 risk agents, SHAP, PDP |
| **Scenario Simulator** | 4 commission sliders → MC 1,000 runs → real-time results |
| **Scenario Comparison** | Pre-computed 6 scenarios, Pareto, AI summary, PDF download |
| **Causal Exploration** | DAG visualization, ATE, refutation table, transition roadmap |
| **Market Context** | CSI time series, elasticity scatter, data preview |

```bash
streamlit run dashboard/app.py
```

---

## How to Run

### Setup

```bash
cd ~/PycharmProjects/Incentive_simulator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pipeline (sequential)

```bash
# 1. Generate synthetic data
python core/data/generate_synthetic.py

# 2. Agent profiling
python -m core.profiler.clustering
python -m core.profiler.visualization

# 3. Churn prediction
python -m core.predictor.baseline
python -m core.predictor.stacking
python -m core.predictor.explainer

# 4. Scenario simulation
python -m core.simulator.monte_carlo
python -m core.simulator.pareto

# 5. AI report
python -m core.report.llm_backend --backend mock
python -m core.report.pdf_generator

# 6. Exploration
python -m exploration.causal
python -m exploration.market_context

# 7. Dashboard
streamlit run dashboard/app.py
```

---

## Tech Stack

```
Python 3.10
scikit-learn | XGBoost | LightGBM | CatBoost
HDBSCAN | UMAP | SHAP
DoWhy (causal inference)
NumPy | SciPy (Monte Carlo)
Plotly | Matplotlib | Kaleido
Streamlit
ReportLab (PDF)
Ollama + Qwen 2.5 (local LLM) | Anthropic Haiku (production LLM)
```

---

## Portfolio Asset Transfer

| Asset | Origin Project | Application Here |
|---|---|---|
| Stacking ensemble | Binary Classification Ensemble | Churn prediction |
| SHAP + PDP | sensor-governance-platform | Feature importance analysis |
| CSI analysis | sportswear-brand-monitor Stage 7 | Capability reference (separate estimation) |
| Ollama multi-backend | consumer-signal-agentic-platform | LLM abstraction pattern |
| CSV fallback pattern | sportswear-brand-monitor | Market context deployment |

---