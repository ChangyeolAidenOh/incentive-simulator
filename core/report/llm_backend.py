"""Module 4: AI Report Generator - LLM Backend

Multi-backend abstraction for report generation:
  - MockBackend:  template-based, no LLM needed (dev/testing)
  - OllamaBackend: Ollama + Qwen 2.5 (local, free)
  - HaikuBackend:  Anthropic Claude Haiku (production quality)
"""

import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a management consultant specializing in insurance agent "
    "incentive structures. Write concise, data-driven analysis in Korean "
    "suitable for C-level executives. Use formal business Korean (합니다체)."
)


def build_prompt(summary_df: pd.DataFrame) -> str:
    """Build analysis prompt from scenario summary."""
    table_str = summary_df.to_string(index=False)

    # Extract baseline for context
    baseline = summary_df[summary_df['scenario'] == 'Baseline'].iloc[0]

    return f"""아래는 보험 설계사 인센티브 시나리오별 Monte Carlo 시뮬레이션(10,000회) 결과입니다.

시나리오 요약:
{table_str}

추가 컨텍스트:
- Baseline 유지율: {baseline['retention_rate_med']:.1%}
- SHAP 분석 결과: customer_satisfaction과 monthly_contracts가 이탈의 핵심 요인
- PDP 분석 결과: commission_rate 변경은 이탈 확률에 미미한 영향

다음 구조로 경영진 보고서의 핵심 분석을 작성하세요 (총 4단락, 각 2-3문장):

1. 핵심 발견: 수수료 변경이 유지율에 미치는 영향의 크기
2. 비용-성과 Trade-off: 어떤 시나리오가 가장 효율적인지
3. 한계 및 주의사항: 합성 데이터 기반 민감도 분석의 경계
4. 제언: 수수료 외 보완 전략 방향"""


# ------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        pass


class MockBackend(LLMBackend):
    """Template-based fallback — no LLM required."""

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        return (
            "1. 핵심 발견\n"
            "6개 시나리오의 Monte Carlo 시뮬레이션 결과, 수수료율 변경에 따른 "
            "유지율 변동 폭은 약 3%p 이내로 나타났습니다. 이는 수수료 구조 변경만으로는 "
            "설계사 이탈을 효과적으로 방지하기 어렵다는 것을 시사합니다.\n\n"
            "2. 비용-성과 Trade-off\n"
            "전 세그먼트 일괄 10% 인상 시 비용은 약 10% 증가하나 유지율 개선은 미미합니다. "
            "반면, At-Risk 세그먼트 집중 투자 시나리오는 상대적으로 낮은 비용 증가 대비 "
            "유지율 하락을 최소화할 수 있습니다.\n\n"
            "3. 한계 및 주의사항\n"
            "본 분석은 합성 데이터 기반의 민감도 분석 프레임워크입니다. 수수료 변경의 "
            "실제 인과효과를 주장하지 않으며, 실데이터 적용 시 도메인 전문가 검증이 "
            "필수적입니다.\n\n"
            "4. 제언\n"
            "수수료 조정과 함께 고객 만족도 제고 프로그램, 실적 향상 지원 체계 등 "
            "비금전적 보상 전략의 병행이 필요합니다. 특히 SHAP 분석에서 확인된 "
            "고객 만족도와 월간 계약 건수의 높은 영향력을 고려한 종합적 접근이 권장됩니다."
        )


class OllamaBackend(LLMBackend):
    """Local Ollama server (qwen2.5:7b recommended)."""

    def __init__(self, model: str = 'qwen2.5:7b', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        import requests
        resp = requests.post(
            f'{self.base_url}/api/generate',
            json={
                'model': self.model,
                'system': system,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.3, 'num_predict': 1024},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()['response']


class HaikuBackend(LLMBackend):
    """Anthropic Claude Haiku API."""

    def __init__(self, api_key: str = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        message = self.client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            system=system,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return message.content[0].text


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_backend(name: str = 'mock') -> LLMBackend:
    """Create LLM backend by name: mock, ollama, haiku."""
    if name == 'mock':
        return MockBackend()
    elif name == 'ollama':
        return OllamaBackend()
    elif name == 'haiku':
        return HaikuBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")


def generate_report_text(summary_df: pd.DataFrame,
                         backend_name: str = 'mock') -> str:
    """End-to-end: build prompt -> call LLM -> return analysis text."""
    backend = get_backend(backend_name)
    prompt = build_prompt(summary_df)
    return backend.generate(prompt)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='mock',
                        choices=['mock', 'ollama', 'haiku'])
    args = parser.parse_args()

    summary_df = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'scenario_summary.csv',
    )
    print(f"Backend: {args.backend}")
    print(f"Scenarios: {len(summary_df)}\n")

    text = generate_report_text(summary_df, args.backend)
    print(text)

    # Save for PDF generator
    out_path = PROJECT_ROOT / 'data' / 'processed' / 'report_analysis.txt'
    out_path.write_text(text, encoding='utf-8')
    print(f"\nSaved: {out_path}")
