# report_generator.py

from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io
from PIL import Image
import numpy as np


# =============================
#  PDF REPORT GENERATOR
# =============================
def generate_report_pdf(
    result,
    probability,
    risk_level,
    benign_prob,
    malignant_prob,
    stats,
    image_size,
    file_format,
    original_image,
    overlay_image,
    heatmap_only,
    bbox_image,
    confidence,
):
    """
    Full professional PDF report generator — FASTAPI READY VERSION
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    story = []
    styles = getSampleStyleSheet()

    # -------------------------
    # CUSTOM STYLES
    # -------------------------
    cover_title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1A237E'),
        alignment=TA_CENTER,
        spaceAfter=30,
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#424242'),
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1A237E'),
        spaceBefore=15,
        spaceAfter=10,
    )

    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#303F9F'),
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=4,
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#C62828'),
        alignment=TA_JUSTIFY,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=6,
    )

    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    # ----------------------------------------
    # Helper: Convert PIL → ReportLab Image
    # ----------------------------------------
    def pil_to_rl_image(img, max_w=5.5 * inch, max_h=4.0 * inch):
        if img is None:
            return None
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        rl_img = RLImage(buf)

        # maintain aspect ratio
        w, h = img.size
        aspect = w / h

        if aspect > (max_w / max_h):  # width-dominant
            rl_img.drawWidth = max_w
            rl_img.drawHeight = max_w / aspect
        else:
            rl_img.drawHeight = max_h
            rl_img.drawWidth = max_h * aspect

        return rl_img

    # ============================
    #  COVER PAGE
    # ============================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("BREAST CANCER DETECTION<br/>ANALYSIS REPORT", cover_title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("AI-Powered Medical Imaging Report", subtitle_style))
    story.append(Spacer(1, 0.4 * inch))

    # COVER Summary Box
    cover_table_data = [
        [f"Classification: {result}"],
        [f"Risk Level: {risk_level}"],
        [f"Confidence: {probability:.2f}%"],
    ]

    cover_table = Table(cover_table_data, colWidths=[5.5 * inch])
    cover_table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#E8EAF6")),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor("#1A237E")),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 14),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]
        )
    )

    story.append(cover_table)
    story.append(Spacer(1, 1 * inch))
    story.append(
        Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}",
            normal_style,
        )
    )
    story.append(PageBreak())

    # ============================
    # EXECUTIVE SUMMARY
    # ============================
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))

    exec_text = f"""
    The AI-based deep learning model analyzed the uploaded breast tissue image and classified it as
    <b>{result}</b> with a confidence score of <b>{probability:.2f}%</b>.
    This report includes detailed visualization insights using Grad-CAM, technical image statistics,
    risk assessment, and clinical recommendations.
    """

    story.append(Paragraph(exec_text, normal_style))
    story.append(Spacer(1, 0.3 * inch))

    # ============================
    # PREDICTION TABLE
    # ============================
    story.append(Paragraph("PREDICTION RESULTS", heading_style))

    prediction_data = [
        ['Parameter', 'Value'],
        ['Classification', result],
        ['Risk Assessment', risk_level],
        ['Confidence Score', f"{probability:.2f}%"],
        ['Benign Probability', f"{benign_prob:.2f}%"],
        ['Malignant Probability', f"{malignant_prob:.2f}%"],
        ['Image Format', file_format],
        ['Dimensions', f"{image_size[0]} × {image_size[1]} pixels"],
        ['Raw Model Output', f"{confidence:.6f}"],
    ]

    prediction_table = Table(prediction_data, colWidths=[2.2 * inch, 4 * inch])
    prediction_table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E8EAF6")),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )

    story.append(prediction_table)
    story.append(PageBreak())

    # =============================
    # VISUAL ANALYSIS (4 IMAGES)
    # =============================

    story.append(Paragraph("VISUAL ANALYSIS", heading_style))

    # 1. Original Image
    story.append(Paragraph("1. Original Medical Image", subheading_style))
    story.append(pil_to_rl_image(original_image))
    story.append(Spacer(1, 0.2 * inch))

    # 2. Heatmap Overlay
    story.append(Paragraph("2. Grad-CAM Heatmap Overlay", subheading_style))
    if overlay_image:
        story.append(pil_to_rl_image(overlay_image))
    story.append(Spacer(1, 0.2 * inch))

    # 3. Heatmap Only
    story.append(Paragraph("3. Heatmap (Standalone Activation Map)", subheading_style))
    if heatmap_only:
        story.append(pil_to_rl_image(heatmap_only))
    story.append(Spacer(1, 0.2 * inch))

    # 4. BBox regions
    story.append(Paragraph("4. Suspicious Regions (Bounding Boxes)", subheading_style))
    if bbox_image:
        story.append(pil_to_rl_image(bbox_image))
    else:
        story.append(Paragraph("No high-activation regions detected above threshold.", normal_style))

    story.append(PageBreak())

    # ============================
    # IMAGE STATISTICS
    # ============================
    story.append(Paragraph("IMAGE STATISTICS", heading_style))

    stats_data = [
        ['Property', 'Value'],
        ['Mean Intensity', f"{stats['mean_intensity']:.2f}"],
        ['Std. Deviation', f"{stats['std_intensity']:.2f}"],
        ['Minimum Pixel', f"{stats['min_intensity']:.0f}"],
        ['Maximum Pixel', f"{stats['max_intensity']:.0f}"],
        ['Median Pixel', f"{stats['median_intensity']:.2f}"],
        ['Brightness Index', f"{stats['brightness']:.2f}%"],
        ['Contrast Index', f"{stats['contrast']:.2f}%"],
    ]

    stats_table = Table(stats_data, colWidths=[2.5 * inch, 3.7 * inch])
    stats_table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E8EAF6")),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ]
        )
    )

    story.append(stats_table)
    story.append(PageBreak())

    # ============================
    # CLINICAL RECOMMENDATIONS
    # ============================
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))

    if confidence > 0.5:
        recs = [
            "Consult an oncologist or radiologist immediately.",
            "Request additional diagnostic imaging and biopsy.",
            "Schedule urgent follow-up appointments.",
            "Share this report with medical professionals.",
            "Discuss potential treatment options.",
        ]
    else:
        recs = [
            "Continue routine breast cancer screening.",
            "Maintain follow-ups with healthcare providers.",
            "Monitor for any changes in symptoms.",
            "Keep previous medical imaging for comparison.",
        ]

    for r in recs:
        story.append(Paragraph("• " + r, bullet_style))

    story.append(PageBreak())

    # ============================
    # DISCLAIMER
    # ============================
    disclaimer_text = """
    ⚠ <b>IMPORTANT MEDICAL DISCLAIMER</b><br/><br/>
    This AI system is for <b>educational and research use only</b>.
    It is NOT clinically validated and must NOT be used as a substitute for real medical diagnosis,
    treatment decisions, or professional healthcare evaluation. Always consult licensed medical
    professionals for diagnosis, imaging interpretation, and treatment.
    """

    disclaimer_box = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[6.5 * inch],
    )
    disclaimer_box.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFEBEE")),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor("#C62828")),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ]
        )
    )

    story.append(disclaimer_box)
    story.append(Spacer(1, 0.4 * inch))

    # ============================
    # FOOTER
    # ============================
    story.append(
        Paragraph(
            "Generated by AI-Powered Breast Cancer Detection System",
            footer_style,
        )
    )
    story.append(
        Paragraph(
            f"© {datetime.now().year} — Educational Use Only",
            footer_style,
        )
    )

    # ============================
    # FINAL BUILD
    # ============================
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
