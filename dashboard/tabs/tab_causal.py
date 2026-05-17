"""Tab 5: Causal Exploration — DoWhy DAG + ATE + Refutation results."""

import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render():
    st.header('Causal Exploration')

    st.warning(
        '**합성 데이터 한계:** ATE 수치에 실증적 의미를 부여하지 않습니다. '
        '이 모듈의 가치는 DAG 구조화 + confounder 식별 + 실데이터 전환 프로세스 설계에 있습니다.'
    )

    # DAG
    st.subheader('Causal DAG')
    dag_img = PROJECT_ROOT / 'figures' / 'causal_dag.png'
    if dag_img.exists():
        st.image(str(dag_img), caption='Domain-knowledge DAG (DoWhy)')
    else:
        st.info('Run `python -m exploration.causal` to generate DAG.')

    # DAG explanation
    st.markdown("""
    **Edge 설계 근거:**
    - `tenure → commission_rate`: 경력에 따른 수수료 등급 차등
    - `tenure → churned_12m`: 신인 설계사 이탈률이 높음 (confounder)
    - `customer_satisfaction → churned_12m`: 만족도가 이탈의 핵심 요인 (SHAP 1위)
    - `commission_rate → churned_12m`: 추정 대상 (treatment → outcome)
    """)

    # Results
    st.subheader('Backdoor ATE Estimation')
    col1, col2 = st.columns(2)
    col1.metric('ATE (commission → churn)', '-0.00196')
    col2.metric('Interpretation', '1%p 수수료 인상 → 이탈률 0.2%p 감소')

    st.caption('Backdoor adjustment variable: tenure_months')

    # Refutation
    st.subheader('Refutation Tests')
    ref_data = {
        'Test': ['Placebo Treatment', 'Random Common Cause', 'Data Subset (80%)'],
        'New ATE': ['~0.0001', '-0.00196', '-0.00200'],
        'p-value': ['0.82', '0.84', '0.92'],
        'Result': ['Passed', 'Passed', 'Passed'],
    }
    st.dataframe(ref_data, use_container_width=True, hide_index=True)

    st.success(
        '3개 refutation test 통과 — 추정치가 robust하나, '
        '이는 합성 데이터 구조의 일관성을 확인한 것이며 인과효과의 실증 근거는 아닙니다.'
    )

    # Transition plan
    st.subheader('실데이터 전환 프로세스')
    st.markdown("""
    1. **도메인 전문가 DAG 검증** — 보험업 전문가와 edge 구조 검토
    2. **Unobserved confounder 민감도 분석** — E-value, Sensitivity analysis
    3. **IV 탐색** — 외부 정책 변경 등 exclusion restriction 충족하는 도구변수 식별
    4. **Refutation 재실행** — 실데이터에서 3개 test 재검증
    """)
