"""
Hugging Face Spaces - Gradio Interface
This is the main app file for Hugging Face deployment
"""
import gradio as gr
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path

# Import existing backend functions
from main import (
    load_model,
    preprocess_image,
    get_risk_level,
    calculate_image_stats
)
from grad_cam import create_gradcam_visualization
from report_generator import generate_report_pdf

# Load model at startup
print("Loading model...")
model = load_model()
print("Model loaded successfully!")

def analyze_image(image):
    """
    Analyze uploaded image and return results
    """
    if image is None:
        return None, None, None, None, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Preprocess
        img_array = preprocess_image(image)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)
        malignant_prob = float(prediction[0][0]) * 100
        benign_prob = 100 - malignant_prob
        
        # Determine result
        threshold = 0.5
        result = "Malignant" if prediction[0][0] > threshold else "Benign"
        confidence = max(malignant_prob, benign_prob)
        
        # Get risk level
        risk_level, risk_icon, risk_color = get_risk_level(malignant_prob / 100)
        
        # Calculate stats
        stats = calculate_image_stats(image)
        
        # Generate Grad-CAM visualizations
        try:
            gradcam_results = create_gradcam_visualization(model, img_array, image)
            overlay_img = gradcam_results.get("overlay")
            heatmap_img = gradcam_results.get("heatmap_only")
            bbox_img = gradcam_results.get("bbox")
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            overlay_img = image
            heatmap_img = image
            bbox_img = image
        
        # Format results
        result_text = f"""
## 🔬 Analysis Results

**Prediction:** {result}  
**Confidence:** {confidence:.2f}%  
**Risk Level:** {risk_icon} {risk_level}

### Probabilities:
- **Benign:** {benign_prob:.2f}%
- **Malignant:** {malignant_prob:.2f}%

### Image Statistics:
- **Mean Intensity:** {stats['mean']:.2f}
- **Std Deviation:** {stats['std']:.2f}
- **Min Intensity:** {stats['min']:.2f}
- **Max Intensity:** {stats['max']:.2f}

---
⚠️ **Disclaimer:** This is an educational demo. Always consult healthcare professionals for medical decisions.
        """
        
        return overlay_img, heatmap_img, bbox_img, result_text
        
    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        return None, None, None, error_msg


def generate_pdf_report(image):
    """
    Generate PDF report for the analyzed image
    """
    if image is None:
        return None
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Preprocess
        img_array = preprocess_image(image)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)
        malignant_prob = float(prediction[0][0])
        
        # Get risk level
        risk_level, _, _ = get_risk_level(malignant_prob)
        
        # Calculate stats
        stats = calculate_image_stats(image)
        
        # Generate Grad-CAM
        try:
            gradcam_results = create_gradcam_visualization(model, img_array, image)
        except:
            gradcam_results = {}
        
        # Generate PDF
        pdf_bytes = generate_report_pdf(
            original_image=image,
            prediction_result="Malignant" if malignant_prob > 0.5 else "Benign",
            confidence=max(malignant_prob, 1 - malignant_prob),
            malignant_prob=malignant_prob,
            benign_prob=1 - malignant_prob,
            risk_level=risk_level,
            stats=stats,
            gradcam_images=gradcam_results
        )
        
        # Save to temporary file
        pdf_path = "/tmp/breast_cancer_report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        
        return pdf_path
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Breast Cancer Detection") as demo:
    
    gr.Markdown("""
    # 🩺 Breast Cancer Detection AI
    
    Upload a medical image (X-ray/Mammogram) for AI-powered analysis using deep learning.
    
    **Features:**
    - Binary classification (Benign/Malignant)
    - Grad-CAM visualization showing model attention
    - Risk level assessment
    - Downloadable PDF report
    
    ⚠️ **Educational Use Only** - Not for clinical diagnosis
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Medical Image",
                type="pil",
                height=400
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                report_btn = gr.Button("📄 Download PDF Report", variant="secondary")
            
            gr.Markdown("""
            ### Supported Formats:
            - JPG, JPEG, PNG
            - Max size: 200MB
            """)
        
        with gr.Column(scale=1):
            result_text = gr.Markdown(label="Analysis Results")
    
    gr.Markdown("## 📊 Grad-CAM Visualizations")
    gr.Markdown("These heatmaps show which regions of the image influenced the model's decision.")
    
    with gr.Row():
        overlay_output = gr.Image(label="Heatmap Overlay", type="pil")
        heatmap_output = gr.Image(label="Heatmap Only", type="pil")
        bbox_output = gr.Image(label="Region Detection", type="pil")
    
    pdf_output = gr.File(label="PDF Report")
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[overlay_output, heatmap_output, bbox_output, result_text]
    )
    
    report_btn.click(
        fn=generate_pdf_report,
        inputs=[input_image],
        outputs=[pdf_output]
    )
    
    gr.Markdown("""
    ---
    ### 📖 About This Model
    
    This AI model uses a Convolutional Neural Network (CNN) trained on medical imaging data to classify breast tissue as benign or malignant.
    
    **Technology Stack:**
    - TensorFlow/Keras for deep learning
    - Grad-CAM for explainable AI
    - FastAPI backend
    - React frontend (deployed separately on Vercel)
    
    **Disclaimer:** This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    ---
    Built with ❤️ using Gradio and Hugging Face Spaces
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()

