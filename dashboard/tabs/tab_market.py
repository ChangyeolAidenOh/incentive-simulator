"""Tab 6: Market Context — ECOS CSI + insurance market elasticity."""

import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_market_data():
    path = PROJECT_ROOT / 'data' / 'raw' / 'market_context.csv'
    if path.exists():
        return pd.read_csv(path, parse_dates=['date'])
    return None


def render():
    st.header('Market Context')
    st.caption('ECOS 소비자심리지수(CSI) + 보험 신계약 탄력성 분석')

    df = load_market_data()
    if df is None:
        st.warning('Run `python -m exploration.market_context` first.')
        return

    # Key metrics
    c1, c2, c3 = st.columns(3)
    c1.metric('CSI (latest)', f"{df['csi'].iloc[-1]:.1f}",
              f"{df['csi'].iloc[-1] - df['csi'].iloc[-2]:+.1f} vs prev month")
    c2.metric('Elasticity', '1.00',
              'Unit elastic: 1% CSI → ~1% contracts')
    c3.metric('Correlation', '0.70', 'CSI ↔ New Contracts')

    # Time series
    st.subheader('CSI vs. Insurance New Contracts')
    img_ts = PROJECT_ROOT / 'figures' / 'market_csi_contracts.png'
    if img_ts.exists():
        st.image(str(img_ts))

    # Elasticity scatter
    st.subheader('CSI-Insurance Elasticity')
    img_sc = PROJECT_ROOT / 'figures' / 'csi_elasticity.png'
    if img_sc.exists():
        st.image(str(img_sc))

    # Context note
    st.info(
        '**참고:** CSI 탄력성 1.00은 sample 데이터 기반 추정이며, '
        'sportswear 프로젝트의 3.98 탄력성과는 별도로 보험 도메인에서 독립 추정한 값입니다. '
        'ECOS API 키 등록 후 실제 데이터로 교체 가능합니다.'
    )

    # Data preview
    with st.expander('Raw Data Preview'):
        st.dataframe(df.tail(12), use_container_width=True)
