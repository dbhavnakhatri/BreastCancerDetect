import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from grad_cam import create_gradcam_visualization

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = keras.models.load_model("models/breast_cancer_model.keras")
    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_image_statistics(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    stats = {
        'mean_intensity': np.mean(img_array),
        'std_intensity': np.std(img_array),
        'min_intensity': np.min(img_array),
        'max_intensity': np.max(img_array),
        'median_intensity': np.median(img_array),
        'brightness': np.mean(img_array) / 255.0 * 100,
        'contrast': np.std(img_array) / 255.0 * 100
    }
    
    return stats

def get_risk_level(confidence):
    if confidence > 0.5:
        malignant_prob = confidence * 100
        if malignant_prob >= 90:
            return "Very High Risk", "🔴", "#ff0000"
        elif malignant_prob >= 75:
            return "High Risk", "🟠", "#ff6600"
        elif malignant_prob >= 60:
            return "Moderate-High Risk", "🟡", "#ffaa00"
        else:
            return "Moderate Risk", "🟡", "#ffcc00"
    else:
        benign_prob = (1 - confidence) * 100
        if benign_prob >= 90:
            return "Very Low Risk", "🟢", "#00cc00"
        elif benign_prob >= 75:
            return "Low Risk", "🟢", "#33cc33"
        elif benign_prob >= 60:
            return "Low-Moderate Risk", "🟡", "#99cc00"
        else:
            return "Moderate Risk", "🟡", "#cccc00"

def create_probability_chart(benign_prob, malignant_prob):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['Benign\n(Non-Cancerous)', 'Malignant\n(Cancerous)']
    probabilities = [benign_prob, malignant_prob]
    colors = ['#4CAF50', '#F44336']
    
    bars = ax.barh(categories, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}%', ha='left', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='50% Threshold')
    ax.legend()
    
    plt.tight_layout()
    return fig

def generate_report_pdf(result, probability, risk_level, benign_prob, malignant_prob, stats, 
                        image_size, file_format, original_image, overlay_image, heatmap_only, bbox_image, confidence):
    """Generate a detailed professional PDF report with images"""
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import io
    from PIL import Image
    import numpy as np
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Define custom styles
    cover_title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1A237E'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#424242'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1A237E'),
        spaceAfter=12,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#303F9F'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        leftIndent=20,
        bulletIndent=10
    )
    
    # === COVER PAGE ===
    story.append(Spacer(1, 1.5*inch))
    
    cover_title = Paragraph("BREAST CANCER DETECTION<br/>ANALYSIS REPORT", cover_title_style)
    story.append(cover_title)
    story.append(Spacer(1, 0.3*inch))
    
    subtitle = Paragraph("AI-Powered Medical Image Analysis System", subtitle_style)
    story.append(subtitle)
    story.append(Spacer(1, 0.5*inch))
    
    # Risk box on cover
    if "Malignant" in result:
        risk_bg_color = colors.HexColor('#FFEBEE')
        risk_border_color = colors.HexColor('#F44336')
    else:
        risk_bg_color = colors.HexColor('#E8F5E9')
        risk_border_color = colors.HexColor('#4CAF50')
    
    cover_summary_data = [[f"Classification: {result}"], 
                         [f"Risk Level: {risk_level}"],
                         [f"Confidence: {probability:.2f}%"]]
    
    cover_table = Table(cover_summary_data, colWidths=[5*inch])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), risk_bg_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 3, risk_border_color),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 1*inch))
    
    timestamp_para = Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", 
                               normal_style)
    story.append(timestamp_para)
    
    story.append(PageBreak())
    
    # === EXECUTIVE SUMMARY ===
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    exec_summary_text = f"""
    This report presents the results of an AI-powered breast cancer detection analysis performed on a medical image. 
    The analysis was conducted using a Convolutional Neural Network (CNN) trained for binary classification of breast 
    tissue as either benign (non-cancerous) or malignant (cancerous). The system has classified this image as 
    <b>{result}</b> with a confidence score of <b>{probability:.2f}%</b>, indicating a <b>{risk_level}</b>.
    """
    story.append(Paragraph(exec_summary_text, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # === PREDICTION RESULTS ===
    story.append(Paragraph("PREDICTION RESULTS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    prediction_data = [
        ['Parameter', 'Value'],
        ['Classification', result],
        ['Risk Assessment', risk_level],
        ['Confidence Score', f'{probability:.2f}%'],
        ['Benign Probability', f'{benign_prob:.2f}%'],
        ['Malignant Probability', f'{malignant_prob:.2f}%'],
        ['Decision Threshold', '50.0%'],
        ['Raw Model Output', f'{confidence:.6f}']
    ]
    
    prediction_table = Table(prediction_data, colWidths=[3*inch, 3.5*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1A237E')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#E8EAF6')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(prediction_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === VISUAL ANALYSIS ===
    story.append(PageBreak())
    story.append(Paragraph("VISUAL ANALYSIS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    visual_intro = """
    The following section presents multiple visualizations of the analyzed medical image. These include the original 
    image, AI attention heatmaps showing regions of interest, and automated region detection highlighting suspicious areas.
    """
    story.append(Paragraph(visual_intro, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Helper function to convert PIL image to ReportLab Image
    def pil_to_rl_image(pil_img, max_width=5.5*inch, max_height=3.5*inch):
        if pil_img is None:
            return None
        img_buffer = io.BytesIO()
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(pil_img.astype('uint8'))
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        rl_img = RLImage(img_buffer)
        aspect = pil_img.size[0] / pil_img.size[1]
        
        if aspect > max_width / max_height:
            rl_img.drawWidth = max_width
            rl_img.drawHeight = max_width / aspect
        else:
            rl_img.drawHeight = max_height
            rl_img.drawWidth = max_height * aspect
        
        return rl_img
    
    # 1. Original Image
    story.append(Paragraph("1. Original Medical Image", subheading_style))
    story.append(Paragraph("Unmodified uploaded medical image", normal_style))
    story.append(Spacer(1, 0.1*inch))
    if original_image:
        orig_img = pil_to_rl_image(original_image, max_width=4.5*inch, max_height=3*inch)
        if orig_img:
            story.append(orig_img)
    story.append(Spacer(1, 0.2*inch))
    
    # 2. Heatmap Overlay
    story.append(Paragraph("2. AI Attention Map (Heatmap Overlay)", subheading_style))
    story.append(Paragraph("Red/yellow regions indicate areas where the AI model focused its attention during analysis. "
                          "Higher intensity colors represent stronger activation.", normal_style))
    story.append(Spacer(1, 0.1*inch))
    if overlay_image:
        overlay_img = pil_to_rl_image(overlay_image, max_width=4.5*inch, max_height=3*inch)
        if overlay_img:
            story.append(overlay_img)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # 3. Heatmap Only
    story.append(Paragraph("3. Detection Heatmap (Standalone)", subheading_style))
    story.append(Paragraph("Pure activation map showing confidence levels across different regions of the image. "
                          "This visualization isolates the model's attention without the original image overlay.", 
                          normal_style))
    story.append(Spacer(1, 0.1*inch))
    if heatmap_only:
        heatmap_img = pil_to_rl_image(heatmap_only, max_width=4.5*inch, max_height=3*inch)
        if heatmap_img:
            story.append(heatmap_img)
    story.append(Spacer(1, 0.2*inch))
    
    # 4. Bounding Boxes
    story.append(Paragraph("4. Automated Region Detection", subheading_style))
    story.append(Paragraph("Red bounding boxes highlight detected regions with concentrated suspicious patterns. "
                          "Each box represents an area where the model detected significant activation patterns.", 
                          normal_style))
    story.append(Spacer(1, 0.1*inch))
    if bbox_image:
        bbox_img = pil_to_rl_image(bbox_image, max_width=4.5*inch, max_height=3*inch)
        if bbox_img:
            story.append(bbox_img)
    else:
        story.append(Paragraph("<i>No distinct regions detected above the detection threshold.</i>", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # === IMAGE STATISTICS ===
    story.append(PageBreak())
    story.append(Paragraph("IMAGE TECHNICAL SPECIFICATIONS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    image_data = [
        ['Property', 'Value'],
        ['File Format', file_format],
        ['Image Dimensions', f'{image_size[0]} × {image_size[1]} pixels'],
        ['Mean Intensity', f"{stats['mean_intensity']:.2f}"],
        ['Standard Deviation', f"{stats['std_intensity']:.2f}"],
        ['Minimum Pixel Value', f"{stats['min_intensity']:.0f}"],
        ['Maximum Pixel Value', f"{stats['max_intensity']:.0f}"],
        ['Brightness Index', f"{stats['brightness']:.2f}%"],
        ['Contrast Index', f"{stats['contrast']:.2f}%"],
    ]
    
    image_table = Table(image_data, colWidths=[3*inch, 3.5*inch])
    image_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1A237E')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#E8EAF6')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === CLINICAL RECOMMENDATIONS ===
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    if confidence > 0.5:
        story.append(Paragraph("<b>ELEVATED RISK DETECTED</b>", subheading_style))
        rec_text = f"""
        The AI model has detected patterns suggesting a {malignant_prob:.2f}% probability of malignancy. 
        Immediate medical consultation is strongly recommended.
        """
        story.append(Paragraph(rec_text, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Recommended Actions:</b>", normal_style))
        story.append(Spacer(1, 0.05*inch))
        recs = [
            "Consult with an oncologist or specialized radiologist immediately",
            "Request additional diagnostic tests including biopsy and advanced imaging",
            "Schedule follow-up appointments promptly",
            "Bring this analysis report to your medical appointment for reference",
            "Discuss comprehensive treatment options with qualified healthcare professionals",
            "Consider seeking a second medical opinion from a specialist"
        ]
        for rec in recs:
            story.append(Paragraph(f"• {rec}", bullet_style))
    else:
        story.append(Paragraph("<b>LOWER RISK INDICATED</b>", subheading_style))
        rec_text = f"""
        The AI model suggests a {benign_prob:.2f}% probability of benign tissue. While this indicates lower risk, 
        regular medical monitoring remains essential.
        """
        story.append(Paragraph(rec_text, normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Recommended Actions:</b>", normal_style))
        story.append(Spacer(1, 0.05*inch))
        recs = [
            "Continue regular screening schedules as recommended by your healthcare provider",
            "Maintain routine check-ups and follow-up appointments",
            "Keep comprehensive records of all imaging studies for future reference",
            "Monitor for any changes in symptoms or physical examination findings",
            "Follow established preventive health guidelines and lifestyle recommendations",
            "Report any new or changing symptoms to your healthcare provider immediately"
        ]
        for rec in recs:
            story.append(Paragraph(f"• {rec}", bullet_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # === MODEL INFORMATION ===
    story.append(PageBreak())
    story.append(Paragraph("MODEL TECHNICAL DETAILS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    model_info = """
    <b>Architecture:</b> Convolutional Neural Network (CNN) for binary classification<br/>
    <b>Input Size:</b> 224 × 224 × 3 (RGB color images)<br/>
    <b>Model Type:</b> Sequential deep learning model<br/>
    <b>Training Framework:</b> TensorFlow/Keras<br/>
    <b>Activation Function:</b> Sigmoid (for binary output)<br/>
    <b>Optimization Algorithm:</b> Adam optimizer<br/>
    <b>Loss Function:</b> Binary cross-entropy<br/>
    <b>Visualization Method:</b> Grad-CAM (Gradient-weighted Class Activation Mapping)<br/>
    <b>Classification Threshold:</b> 50% probability cutoff
    """
    story.append(Paragraph(model_info, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Model Classification Criteria:", subheading_style))
    criteria_text = """
    The model outputs a probability score between 0 and 1. Classifications are determined as follows:<br/>
    • <b>Benign (Non-Cancerous):</b> Prediction score &lt; 0.5 (50%)<br/>
    • <b>Malignant (Cancerous):</b> Prediction score ≥ 0.5 (50%)
    """
    story.append(Paragraph(criteria_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # === RISK ASSESSMENT GUIDE ===
    story.append(Paragraph("RISK ASSESSMENT CLASSIFICATION GUIDE", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    risk_data = [
        ['Risk Level', 'Probability Range', 'Recommendation'],
        ['Very Low Risk', '< 10%', 'Routine screening'],
        ['Low Risk', '10-25%', 'Regular monitoring'],
        ['Low-Moderate Risk', '25-40%', 'Enhanced surveillance'],
        ['Moderate Risk', '40-60%', 'Additional testing recommended'],
        ['Moderate-High Risk', '60-75%', 'Urgent medical consultation'],
        ['High Risk', '75-90%', 'Immediate specialist referral'],
        ['Very High Risk', '> 90%', 'Emergency medical attention'],
    ]
    
    risk_table = Table(risk_data, colWidths=[1.8*inch, 1.8*inch, 3*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#1A237E')),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
        ('BACKGROUND', (0, 1), (2, -1), colors.HexColor('#F5F5F5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.2*inch))
    
    current_risk = Paragraph(f"<b>Current Assessment: {risk_level} ({malignant_prob:.2f}% Malignant Probability)</b>", 
                            normal_style)
    story.append(current_risk)
    story.append(Spacer(1, 0.4*inch))
    
    # === MEDICAL DISCLAIMER ===
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#C62828'),
        spaceAfter=6,
        leftIndent=15,
        rightIndent=15,
        alignment=TA_JUSTIFY
    )
    
    disclaimer_box_data = [[
        """⚠ WARNING: This analysis is for EDUCATIONAL and RESEARCH purposes ONLY. 
        
This AI-powered system should NOT be used as the sole basis for medical diagnosis or treatment decisions. 
The model was trained on synthetic data and may not accurately reflect real-world medical conditions.

CRITICAL NOTICES:
• This tool is NOT a substitute for professional medical evaluation
• Always consult qualified healthcare professionals for medical advice and diagnosis
• Do not delay seeking medical attention based on these results
• Professional medical imaging interpretation by certified radiologists is essential
• The accuracy and reliability of this system have not been validated for clinical use
• Results should be interpreted only by licensed medical professionals

This system is designed as an educational demonstration of AI technology in medical imaging 
and should not influence medical treatment decisions."""
    ]]
    
    disclaimer_table = Table(disclaimer_box_data, colWidths=[6*inch])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFEBEE')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#C62828')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#C62828')),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === FOOTER ===
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Spacer(1, 0.3*inch))
    footer = Paragraph(f"Generated by Breast Cancer Detection System (AI-Powered) | {datetime.now().strftime('%Y')}", 
                      footer_style)
    story.append(footer)
    footer2 = Paragraph("This report is confidential and intended for educational purposes only.", footer_style)
    story.append(footer2)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def show_upload_page():
    """Display the file upload page"""
    # Custom CSS for modern design
    st.markdown("""
    <style>
    .main-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-card">
        <h1 style="margin:0; font-size: 2.5em;">🔬 Breast Cancer Detection System</h1>
        <p style="margin:10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
            Advanced AI-Powered Medical Image Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. 
    It should NOT be used for actual medical diagnosis. Always consult qualified healthcare 
    professionals for medical advice and diagnosis.
    """)
    
    st.markdown("### 📤 Upload Medical Image")
    st.info("💡 Upload a mammogram or histopathology image for comprehensive AI analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.success("✅ Image uploaded successfully!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Medical Image", use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔬 Analyzing image with AI models... Please wait..."):
                try:
                    model = load_model()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.stop()
                
                try:
                    preprocessed_img = preprocess_image(image)
                except Exception as e:
                    st.error(f"Error preprocessing image: {str(e)}")
                    st.stop()
                
                prediction = model.predict(preprocessed_img, verbose=0)[0][0]
                confidence = float(prediction)
                stats = get_image_statistics(image)
                
                benign_prob = (1 - confidence) * 100
                malignant_prob = confidence * 100
                
                heatmap, overlay_image, heatmap_only, bbox_image, heatmap_error = create_gradcam_visualization(
                    image, preprocessed_img, model, confidence
                )
                
                if confidence > 0.5:
                    result = "Malignant (Cancerous)"
                    probability = malignant_prob
                    risk_icon = "🔴"
                    risk_color = "#F44336"
                else:
                    result = "Benign (Non-Cancerous)"
                    probability = benign_prob
                    risk_icon = "🟢"
                    risk_color = "#4CAF50"
                
                if malignant_prob > 90:
                    risk_level = "Very High Risk"
                elif malignant_prob > 75:
                    risk_level = "High Risk"
                elif malignant_prob > 60:
                    risk_level = "Moderate-High Risk"
                elif malignant_prob > 40:
                    risk_level = "Moderate Risk"
                elif malignant_prob > 25:
                    risk_level = "Low-Moderate Risk"
                elif malignant_prob > 10:
                    risk_level = "Low Risk"
                else:
                    risk_level = "Very Low Risk"
                
                st.session_state.analysis_complete = True
                st.session_state.analysis_data = {
                    'image': image,
                    'result': result,
                    'probability': probability,
                    'confidence': confidence,
                    'benign_prob': benign_prob,
                    'malignant_prob': malignant_prob,
                    'risk_level': risk_level,
                    'risk_icon': risk_icon,
                    'risk_color': risk_color,
                    'stats': stats,
                    'overlay_image': overlay_image,
                    'heatmap_only': heatmap_only,
                    'bbox_image': bbox_image,
                    'heatmap_error': heatmap_error
                }
                st.session_state.current_page = 'results'
                st.rerun()
    
    else:
        st.info("👆 Please upload a medical image to begin analysis")
    
    st.markdown("---")
    st.markdown("""
    ### About This Tool
    
    This breast cancer detection system uses a Convolutional Neural Network (CNN) to analyze 
    medical images and predict the likelihood of malignancy. The model analyzes patterns in 
    the uploaded image to classify tissue as benign or malignant.
    
    **How it works:**
    1. Upload a medical image (mammogram or histopathology slide)
    2. The image is preprocessed and normalized
    3. A deep learning model analyzes the image features
    4. Results are displayed with confidence scores and visualizations
    
    **Important Notes:**
    - This is a demonstration model for educational purposes
    - Actual medical diagnosis requires professional evaluation
    - The model has not been clinically validated
    - Always seek professional medical advice
    """)

def show_results_page():
    """Display the comprehensive analysis results page"""
    if 'analysis_data' not in st.session_state:
        st.error("No analysis data found. Please upload and analyze an image first.")
        if st.button("← Back to Upload"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    data = st.session_state.analysis_data
    image = data['image']
    result = data['result']
    probability = data['probability']
    confidence = data['confidence']
    benign_prob = data['benign_prob']
    malignant_prob = data['malignant_prob']
    risk_level = data['risk_level']
    risk_icon = data['risk_icon']
    risk_color = data['risk_color']
    stats = data['stats']
    overlay_image = data['overlay_image']
    heatmap_only = data['heatmap_only']
    bbox_image = data['bbox_image']
    heatmap_error = data['heatmap_error']
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Back button at top
    if st.button("← Back to Upload", key="back_top"):
        st.session_state.current_page = 'upload'
        st.session_state.analysis_complete = False
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div class="main-card">
        <h1 style="margin:0; font-size: 2em;">📊 Comprehensive Analysis Report</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Main results with modern card design
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {risk_color}15 0%, {risk_color}30 100%);
        padding: 15px 20px;
        border-radius: 10px;
        border: 2px solid {risk_color};
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    ">
        <h2 style="text-align: center; margin: 0; color: {risk_color}; font-size: 1.5em;">
            <span style="font-size: 0.7em;">{risk_icon}</span> {result}
        </h2>
        <h3 style="text-align: center; color: {risk_color}; margin: 8px 0; font-weight: normal; font-size: 1.1em;">
            {risk_level}
        </h3>
        <div style="
            background: white;
            padding: 12px;
            border-radius: 8px;
            margin-top: 12px;
            text-align: center;
        ">
            <h4 style="margin: 0; color: #333; font-size: 1em;">Confidence: {probability:.2f}%</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics in modern cards
    st.markdown("### 📊 Prediction Metrics")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            label="🎯 Confidence",
            value=f"{probability:.1f}%",
            delta=None
        )
    
    with metrics_col2:
        st.metric(
            label="✅ Benign",
            value=f"{benign_prob:.1f}%",
            delta=f"{benign_prob-50:.1f}%" if benign_prob > 50 else None,
            delta_color="inverse"
        )
    
    with metrics_col3:
        st.metric(
            label="⚠️ Malignant",
            value=f"{malignant_prob:.1f}%",
            delta=f"{malignant_prob-50:.1f}%" if malignant_prob > 50 else None,
            delta_color="normal"
        )
    
    with metrics_col4:
        st.metric(
            label="📈 Risk Level",
            value=risk_level.split()[0],
            delta=None
        )
    
    # Progress bar with gradient
    st.markdown("#### Confidence Indicator")
    st.progress(probability / 100)
    
    st.markdown("---")
    
    # Visual Analysis Section - Clean and Professional
    st.markdown("## 🔍 Visual Analysis")
    
    # All tabs are always visible
    tabs = st.tabs(["Heatmap Overlay", "Heatmap Only", "Region Detection", "Original Image"])
    
    # Heatmap Overlay Tab
    with tabs[0]:
        st.markdown("#### AI Attention Map")
        st.caption("Red/yellow areas show where the AI model focused its attention during analysis")
        
        if overlay_image is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(overlay_image, use_container_width=True)
        else:
            st.error("Unable to generate heatmap")
    
    # Heatmap Only Tab
    with tabs[1]:
        st.markdown("#### Detection Heatmap")
        st.caption("Pure activation map showing confidence levels across the image")
        
        if heatmap_only is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(heatmap_only, use_container_width=True)
        else:
            st.error("Unable to generate heatmap")
    
    # Bounding Boxes Tab
    with tabs[2]:
        st.markdown("#### Detected Regions")
        st.caption("Red boxes highlight areas with concentrated suspicious patterns")
        
        if bbox_image is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(bbox_image, use_container_width=True)
        else:
            st.info("No distinct regions detected above threshold")
    
    # Original Image Tab
    with tabs[3]:
        st.markdown("#### Original Medical Image")
        st.caption("Unmodified uploaded image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # File info
        with st.expander("File Details"):
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.text(f"Format: {image.format if image.format else 'N/A'}")
                st.text(f"Dimensions: {image.size[0]} × {image.size[1]} px")
            with info_col2:
                st.text(f"Color Mode: {image.mode}")
                import sys
                img_size = sys.getsizeof(image)
                st.text(f"Estimated Size: {img_size / 1024:.1f} KB")
    
    st.markdown("---")
    
    # Recommendations Section - 3 column layout for better space utilization
    st.markdown("## 💡 Analysis Summary")
    
    rec_col1, rec_col2, rec_col3 = st.columns([2, 1, 1])
    
    with rec_col1:
        st.markdown("### Clinical Recommendations")
        if confidence > 0.5:
            st.error(f"""
            **⚠️ Elevated Risk Detected**
            
            The AI model has detected patterns suggesting a **{malignant_prob:.2f}% probability of malignancy**.
            
            **Recommended Actions:**
            - 🏥 Consult with an oncologist or specialized radiologist immediately
            - 📋 Request additional diagnostic tests (biopsy, advanced imaging)
            - 📅 Schedule follow-up appointments promptly
            - 📝 Bring this analysis to your medical appointment
            - 👨‍⚕️ Discuss treatment options with healthcare professionals
            
            **Important:** This is an AI screening tool only. Professional medical evaluation is essential.
            """)
        else:
            st.success(f"""
            **✅ Lower Risk Indicated**
            
            The AI model suggests a **{benign_prob:.2f}% probability of benign tissue**.
            
            **Recommended Actions:**
            - 📅 Continue regular screening schedules as recommended
            - 🏥 Maintain routine check-ups with your healthcare provider
            - 📋 Keep records of all imaging studies
            - 👁️ Monitor for any changes in symptoms
            - 💪 Follow preventive health guidelines
            
            **Important:** Regular medical monitoring is still essential for early detection.
            """)
    
    with rec_col2:
        st.markdown("### ⚡ Quick Actions")
        
        # Generate PDF report for download
        report_pdf = generate_report_pdf(
            result, probability, risk_level, 
            benign_prob, malignant_prob, stats,
            image.size, image.format,
            image, overlay_image, heatmap_only, bbox_image, confidence
        )
        
        st.download_button(
            label="📥 Download Report",
            data=report_pdf,
            file_name=f"breast_cancer_analysis_{probability:.0f}pct.pdf",
            mime="application/pdf",
            help="Download detailed analysis report as PDF",
            use_container_width=True
        )
        
        if st.button("🔄 Analyze Another", help="Upload new image", use_container_width=True):
            st.session_state.current_page = 'upload'
            st.session_state.analysis_complete = False
            st.rerun()
    
    with rec_col3:
        st.markdown("### 📊 Risk Breakdown")
        st.markdown(f"""
        <div style="background: {risk_color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {risk_color};">
            <p style="margin: 5px 0;"><strong>Classification:</strong> {result}</p>
            <p style="margin: 5px 0;"><strong>Risk Level:</strong> {risk_level}</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> {probability:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability Chart - always visible
    st.markdown("### 📈 Probability Distribution Chart")
    fig = create_probability_chart(benign_prob, malignant_prob)
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Model Information", "📊 Risk Assessment Guide", "🎯 Heatmap Explanation", "🩺 Clinical Context"])
    
    with tab1:
        st.markdown(f"""
        #### Deep Learning Model Details
        
        **Architecture:**
        - **Type:** Convolutional Neural Network (CNN)
        - **Input Size:** 224 × 224 × 3 (RGB)
        - **Layers:** Multiple convolutional and pooling layers
        - **Output:** Binary classification (Sigmoid activation)
        - **Training:** Synthetic dataset (demonstration purposes)
        
        **Classification Criteria:**
        - **Threshold:** 50% probability
        - **Benign:** Prediction < 0.5
        - **Malignant:** Prediction ≥ 0.5
        
        **Current Prediction Scores:**
        - **Raw Score:** {confidence:.6f}
        - **Benign Probability:** {benign_prob:.4f}%
        - **Malignant Probability:** {malignant_prob:.4f}%
        
        **Visualization Technology:**
        - **Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)
        - **Purpose:** Shows which regions of the image influenced the model's decision
        - **Layer Used:** Final convolutional layer activations
        """)
    
    with tab2:
        st.markdown(f"""
        #### Risk Level Classification Guide
        
        | Risk Level | Malignant Probability | Recommendation |
        |------------|----------------------|----------------|
        | 🟢 Very Low Risk | < 10% | Routine screening |
        | 🟢 Low Risk | 10-25% | Regular monitoring |
        | 🟡 Low-Moderate Risk | 25-40% | Enhanced surveillance |
        | 🟡 Moderate Risk | 40-60% | Additional testing recommended |
        | 🟡 Moderate-High Risk | 60-75% | Urgent medical consultation |
        | 🟠 High Risk | 75-90% | Immediate specialist referral |
        | 🔴 Very High Risk | > 90% | Emergency medical attention |
        
        **Your Current Assessment:** {risk_icon} **{risk_level}**
        """)
    
    with tab3:
        st.markdown("""
        #### 🎯 Understanding the Detection Heatmap
        
        **What is Grad-CAM?**
        
        Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique that shows exactly which parts of the image the AI model focused on when making its prediction. Think of it as highlighting where the model "looked" to make its decision.
        
        **How to Read the Heatmap:**
        
        - **🔴 Red/Hot Colors:** Areas the model considers MOST important for its prediction
        - **🟡 Yellow/Warm Colors:** Moderately important regions
        - **🟢 Green/Cool Colors:** Less important areas
        - **🔵 Blue/Cold Colors:** Regions the model largely ignored
        
        **What the Heatmap Shows:**
        
        1. **Suspicious Regions:** Red/yellow areas indicate where the model detected patterns associated with malignancy or benign features
        2. **Focus Areas:** The model examines texture, density, and structural patterns in these highlighted regions
        3. **Spatial Distribution:** Shows whether concerning features are localized or spread across the tissue
        
        **Important Limitations:**
        
        ⚠️ **The heatmap is NOT a medical diagnosis**
        - It shows where the AI looked, not a definitive cancer location
        - This demonstration model uses synthetic training data
        - Real diagnostic imaging requires professional radiologist interpretation
        - Heatmaps can be diffuse or imprecise with limited training data
        - The model may focus on artifacts or image quality issues
        
        **Clinical Perspective:**
        
        In real medical imaging:
        - Radiologists use multiple views and imaging modalities
        - Biopsy is required for definitive diagnosis
        - AI heatmaps serve as decision support tools, not replacements
        - Context from patient history is essential
        
        **How Medical Professionals Use This:**
        
        1. **Screening Aid:** Highlights areas for closer human examination
        2. **Second Opinion:** Provides additional data point alongside radiologist review
        3. **Documentation:** Visual record of AI analysis for medical records
        4. **Patient Communication:** Helps explain findings in consultations
        """)
    
    with tab4:
        st.markdown("""
        #### Understanding Breast Cancer Detection
        
        **Types of Medical Images:**
        - **Mammograms:** X-ray images of breast tissue
        - **Histopathology:** Microscopic tissue samples
        - **Ultrasound:** Sound wave imaging
        - **MRI:** Magnetic resonance imaging
        
        **What the AI Looks For:**
        - Irregular tissue patterns
        - Abnormal cell structures
        - Density variations
        - Suspicious calcifications
        - Tissue architecture changes
        
        **Limitations of AI Analysis:**
        - ⚠️ This is a demonstration model using synthetic training data
        - ⚠️ Not validated for clinical use
        - ⚠️ Cannot replace professional radiologist interpretation
        - ⚠️ Many factors affect accuracy (image quality, positioning, etc.)
        - ⚠️ False positives and false negatives can occur
        
        **Next Steps:**
        - Share results with your healthcare provider
        - Undergo professional diagnostic evaluation
        - Follow recommended screening protocols
        - Maintain open communication with medical team
        """)
    
    # Back button at bottom
    if st.button("← Back to Upload", key="back_bottom", use_container_width=True):
        st.session_state.current_page = 'upload'
        st.session_state.analysis_complete = False
        st.rerun()

def main():
    """Main function to route between pages"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Route to appropriate page
    if st.session_state.current_page == 'upload':
        show_upload_page()
    elif st.session_state.current_page == 'results':
        show_results_page()

if __name__ == "__main__":
    main()



