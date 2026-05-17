"""Module 4: AI Report Generator - PDF Executive Report

Generates a 1-page PDF combining scenario summary table,
LLM-generated trade-off analysis, and Pareto chart.
Uses Korean font with fallback for macOS.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRAND_RED = HexColor('#C8102E')
BRAND_GRAY = HexColor('#4A4A4A')


# ------------------------------------------------------------------
# Font registration
# ------------------------------------------------------------------

def register_korean_font() -> str:
    """Try to register a Korean font. Returns font name."""
    candidates = [
        ('AppleGothic', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', None),
        ('AppleSDGothic', '/System/Library/Fonts/AppleSDGothicNeo.ttc', 0),
        ('NanumGothic', '/Library/Fonts/NanumGothic.ttf', None),
        ('NanumGothic', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf', None),
    ]
    for name, path, subfont in candidates:
        if Path(path).exists():
            try:
                if subfont is not None:
                    pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subfont))
                else:
                    pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue

    print("Warning: no Korean font found, falling back to Helvetica")
    return 'Helvetica'


# ------------------------------------------------------------------
# Styles
# ------------------------------------------------------------------

def build_styles(font_name: str) -> dict:
    """Create paragraph styles for the report."""
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'ReportTitle', parent=base['Title'],
            fontName=font_name, fontSize=16, textColor=BRAND_RED,
            spaceAfter=4 * mm,
        ),
        'subtitle': ParagraphStyle(
            'ReportSubtitle', parent=base['Normal'],
            fontName=font_name, fontSize=9, textColor=BRAND_GRAY,
            spaceAfter=6 * mm,
        ),
        'heading': ParagraphStyle(
            'SectionHeading', parent=base['Heading2'],
            fontName=font_name, fontSize=11, textColor=BRAND_RED,
            spaceBefore=4 * mm, spaceAfter=2 * mm,
        ),
        'body': ParagraphStyle(
            'ReportBody', parent=base['Normal'],
            fontName=font_name, fontSize=9, leading=14,
            textColor=BRAND_GRAY, spaceAfter=2 * mm,
        ),
    }


# ------------------------------------------------------------------
# Table builder
# ------------------------------------------------------------------

def build_scenario_table(summary_df: pd.DataFrame, font_name: str) -> Table:
    """Create a formatted scenario comparison table."""
    headers = ['Scenario', 'Retention', 'Cost (10K)', 'Productivity']
    data = [headers]

    for _, row in summary_df.iterrows():
        data.append([
            row['scenario'],
            f"{row['retention_rate_med']:.1%}",
            f"{row['total_cost_med']:,.0f}",
            f"{row['productivity_med']:,.0f}",
        ])

    table = Table(data, colWidths=[40 * mm, 30 * mm, 35 * mm, 35 * mm])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), font_name),
        ('FONTNAME', (0, 1), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_RED),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F8F8F8')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return table


# ------------------------------------------------------------------
# PDF builder
# ------------------------------------------------------------------

def generate_pdf(
    summary_df: pd.DataFrame,
    analysis_text: str,
    chart_path: Path = None,
    output_path: Path = None,
):
    """Generate 1-page executive PDF report."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'processed' / 'executive_report.pdf'
    if chart_path is None:
        chart_path = PROJECT_ROOT / 'figures' / 'pareto_cost_retention.png'

    font_name = register_korean_font()
    styles = build_styles(font_name)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    story = []

    # Title
    story.append(Paragraph(
        '설계사 인센티브 시나리오 분석 보고서', styles['title'],
    ))
    story.append(Paragraph(
        f'Agent Incentive Scenario Analysis | {datetime.now().strftime("%Y-%m-%d")}',
        styles['subtitle'],
    ))

    # Scenario table
    story.append(Paragraph('시나리오 비교', styles['heading']))
    story.append(build_scenario_table(summary_df, font_name))
    story.append(Spacer(1, 4 * mm))

    # LLM analysis
    story.append(Paragraph('분석 요약', styles['heading']))
    for paragraph in analysis_text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        story.append(Paragraph(paragraph.replace('\n', '<br/>'), styles['body']))

    # Chart
    if chart_path.exists():
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph('비용-유지율 Trade-off', styles['heading']))
        img = Image(str(chart_path), width=130 * mm, height=79 * mm)
        story.append(img)

    # Footer note
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        '※ 본 보고서는 합성 데이터 기반 민감도 분석 결과이며, '
        '실제 정책 결정에는 실데이터 검증이 필요합니다.',
        ParagraphStyle('Footer', fontName=font_name, fontSize=7,
                       textColor=colors.grey),
    ))

    doc.build(story)
    return output_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from core.report.llm_backend import generate_report_text

    summary_df = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'scenario_summary.csv',
    )

    # Check for saved analysis or generate fresh
    analysis_path = PROJECT_ROOT / 'data' / 'processed' / 'report_analysis.txt'
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding='utf-8')
        print("Loaded existing analysis text")
    else:
        print("Generating analysis (mock backend)...")
        analysis_text = generate_report_text(summary_df, 'mock')

    out = generate_pdf(summary_df, analysis_text)
    print(f"Generated: {out}")
