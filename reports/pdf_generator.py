"""
Clinical-Grade PDF Case Summary Report Generator.
Utilizes ReportLab to generate a multi-page / structured PDF document
summarizing AI detection, severity, SB-RS risk score, uncertainty metrics,
Grad-CAM overlay visualizations, and risk factor decomposition.
"""

import os
import cv2
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from config.settings import REPORTS_DIR, DISCLAIMER_TEXT

def generate_pdf_case_report(
    case_id: str,
    detection_data: Dict[str, Any],
    severity_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    uncertainty_data: Dict[str, Any],
    explanation_data: Dict[str, Any],
    decision_data: Dict[str, Any],
    original_img_rgb: np.ndarray,
    overlay_img_rgb: np.ndarray,
    gestational_week: Optional[int] = 20,
    output_filename: Optional[str] = None
) -> str:
    """
    Generates a formatted PDF report and returns the absolute file path.
    """
    if output_filename is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"SpinaBifida_Report_{case_id}_{timestamp_str}.pdf"
        
    pdf_path = str(REPORTS_DIR / output_filename)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E293B"),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )
    section_heading = ParagraphStyle(
        "SecHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_text = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#64748B")
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("AI-ASSISTED SPINA BIFIDA RISK STRATIFICATION REPORT", title_style))
    story.append(Paragraph(f"Research Decision-Support Prototype | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=8, spaceBefore=4))
    
    # Case Overview Metadata Table
    case_table_data = [
        [
            Paragraph(f"<b>Case ID:</b> {case_id}", body_text),
            Paragraph(f"<b>Gestational Age:</b> {gestational_week} Weeks", body_text),
            Paragraph(f"<b>Scan Modality:</b> Fetal Ultrasound (Transabdominal)", body_text)
        ],
        [
            Paragraph(f"<b>Model Architecture:</b> ResNet-18 (MC Dropout)", body_text),
            Paragraph(f"<b>Analysis Status:</b> Evaluated (Research Prototype)", body_text),
            Paragraph(f"<b>Security / Privacy:</b> De-identified Data", body_text)
        ]
    ]
    t_case = Table(case_table_data, colWidths=[180, 180, 180])
    t_case.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_case)
    story.append(Spacer(1, 8))
    
    # Key Diagnostic Findings Grid
    story.append(Paragraph("1. Primary Diagnostic & Risk Stratification Summary", section_heading))
    
    pred_label = detection_data.get("prediction", "Spina Bifida")
    pred_prob = detection_data.get("probability", 0.90) * 100
    sev_grade = severity_data.get("grade", "Moderate")
    risk_score = risk_data.get("sb_rs_score", 75.0)
    risk_cat = risk_data.get("risk_category", "HIGH")
    unc_level = uncertainty_data.get("uncertainty_level", "LOW")
    action_protocol = decision_data.get("action_protocol", "Specialist evaluation advised.")
    
    summary_data = [
        [
            Paragraph("<b>Diagnostic Prediction</b>", body_text),
            Paragraph("<b>Severity Assessment</b>", body_text),
            Paragraph("<b>SB-RS Risk Score</b>", body_text),
            Paragraph("<b>Epistemic Uncertainty</b>", body_text)
        ],
        [
            Paragraph(f"<font color='#0284C7' size='11'><b>{pred_label}</b></font><br/>Prob: {pred_prob:.1f}%", body_text),
            Paragraph(f"<font color='#D97706' size='11'><b>{sev_grade}</b></font><br/>Index: {severity_data.get('severity_score', 0.5):.2f}", body_text),
            Paragraph(f"<font color='#DC2626' size='11'><b>{risk_score}/100 ({risk_cat})</b></font><br/>Category: {risk_cat}", body_text),
            Paragraph(f"<font color='#059669' size='11'><b>{unc_level}</b></font><br/>Variance: {uncertainty_data.get('variance', 0.01):.4f}", body_text)
        ]
    ]
    t_summary = Table(summary_data, colWidths=[135, 135, 135, 135])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))
    
    # Visual Explainability (Images Side by Side)
    story.append(Paragraph("2. Multimodal Explainability & Visual AI Attention (Grad-CAM)", section_heading))
    
    # Save temporary PNGs for ReportLab embedding
    temp_dir = tempfile.mkdtemp()
    orig_path = os.path.join(temp_dir, "orig.png")
    overlay_path = os.path.join(temp_dir, "overlay.png")
    
    cv2.imwrite(orig_path, cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img_rgb, cv2.COLOR_RGB2BGR))
    
    img_orig_rl = RLImage(orig_path, width=200, height=200)
    img_overlay_rl = RLImage(overlay_path, width=200, height=200)
    
    img_table_data = [
        [img_orig_rl, img_overlay_rl],
        [
            Paragraph("<b>Original Ultrasound Scan</b>", body_text),
            Paragraph(f"<b>Grad-CAM Overlay</b> (Focus: {explanation_data.get('focus_data', {}).get('anatomical_zone', 'Frontal Cranium')})", body_text)
        ]
    ]
    t_imgs = Table(img_table_data, colWidths=[270, 270])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 8))
    
    # Risk Factor Breakdown Table
    story.append(Paragraph("3. Risk Factor Contribution Attribution (Why this Risk?)", section_heading))
    
    factors = explanation_data.get("contributions", {}).get("factors", [])
    factor_table_rows = [
        [Paragraph("<b>Risk Contributing Factor</b>", body_text),
         Paragraph("<b>Observed Indicator</b>", body_text),
         Paragraph("<b>Config Weight</b>", body_text),
         Paragraph("<b>SB-RS Contribution</b>", body_text)]
    ]
    for f in factors:
        factor_table_rows.append([
            Paragraph(f.get("factor", ""), body_text),
            Paragraph(str(f.get("value", "")), body_text),
            Paragraph(str(f.get("weight", "")), body_text),
            Paragraph(f"<b>+{f.get('contribution', 0.0):.1f} pts</b>", body_text)
        ])
        
    t_factors = Table(factor_table_rows, colWidths=[200, 110, 100, 130])
    t_factors.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_factors)
    story.append(Spacer(1, 8))
    
    # Narrative Clinical Interpretation & Recommendation
    story.append(Paragraph("4. Clinical Decision-Support Interpretation & Recommendation", section_heading))
    narrative_text = explanation_data.get("narrative", "Case evaluation completed.")
    story.append(Paragraph(f"<b>Summary Narrative:</b> {narrative_text}", body_text))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Recommended Action Protocol:</b> {action_protocol}", body_text))
    story.append(Spacer(1, 10))
    
    # Mandatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"), spaceAfter=6, spaceBefore=4))
    story.append(Paragraph(f"<b>RESEARCH DISCLAIMER:</b> {DISCLAIMER_TEXT}", disclaimer_style))
    
    doc.build(story)
    
    # Clean temporary image files
    try:
        os.remove(orig_path)
        os.remove(overlay_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    return pdf_path
