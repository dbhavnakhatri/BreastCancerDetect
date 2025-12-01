# Overview

This is a Breast Cancer Detection System built with Streamlit and TensorFlow. The application provides a comprehensive web interface for uploading medical images (mammograms or histopathology images) and uses a deep learning model to perform binary classification (benign vs. malignant). The system features detailed analysis results including risk assessment, probability visualizations, image statistics, and clinical context. Designed for educational and research purposes only, with clear medical disclaimers prominently displayed.

# Recent Changes (November 19, 2025)

**2-Page Application Structure (Latest):**
- Complete reorganization into separate upload and results pages
- **Page 1 - Upload Page:** Drag-and-drop file uploader with preview, medical disclaimer, and "About This Tool" section
- **Page 2 - Comprehensive Analysis:** Complete analysis results displayed on dedicated page
- Session state management for navigation between pages and data persistence
- "Back to Upload" buttons on results page (top and bottom) for easy navigation
- "Analyze Image" button triggers AI analysis and automatic page transition
- All analysis features (heatmaps, bounding boxes, statistics) remain mandatory and always visible
- Clean separation improves user experience and workflow clarity

**Professional PDF Report Generation:**
- Replaced text download with comprehensive multi-page PDF report
- 8-page professional format with cover page, executive summary, and detailed sections
- All 4 visualization images embedded (Original, Heatmap Overlay, Heatmap Only, Bounding Boxes)
- Technical specifications table with image statistics
- Risk assessment guide with 7-tier classification system
- Clinical recommendations based on malignancy probability
- Model technical details and Grad-CAM explanation
- Medical disclaimer prominently displayed
- Professional styling with tables, headers, and color-coded risk indicators

**Mandatory Visualization Features:**
- Removed all toggles - heatmap, bounding boxes, and statistics always displayed
- Streamlined 3-column layout (Clinical Recommendations, Quick Actions, Risk Breakdown)
- All features permanently visible for comprehensive analysis
- Interactive controls section removed in favor of mandatory display

**Bounding Box Detection:**
- Added automatic bounding box detection around high-activation regions from the heatmap
- Implemented region detection using thresholding and connected component analysis (scipy.ndimage)
- Each detected region shows: red bounding box + confidence label ("Region N: XX.X%")
- Smart label positioning to avoid clipping at image edges
- UI displays bounding boxes with informative message when no distinct regions are detected
- Provides YOLO-like visual localization without requiring labeled training data

**Grad-CAM Heatmap Visualization:**
- Implemented Grad-CAM (Gradient-weighted Class Activation Mapping) to show exact regions where the model detects potential cancer
- Created grad_cam.py module with heatmap generation using layer indices (compatible with loaded Sequential models)
- Added color-coded heatmap overlay (red/yellow for high attention areas, green/blue for low attention)
- Integrated heatmap visualization in the UI showing both standalone heatmap and overlay on original image
- Added "Heatmap Explanation" tab with detailed interpretation guide and color legend
- Includes prominent medical disclaimer that heatmap is for educational purposes only

**Enhanced Detailed Results Display:**
- Added preprocessed image visualization showing the 224x224 normalized input to the model
- Implemented multi-tier risk assessment levels (Very Low, Low, Low-Moderate, Moderate, Moderate-High, High, Very High)
- Added comprehensive image statistics (mean intensity, standard deviation, brightness, contrast, min/max values)
- Created interactive probability distribution chart using matplotlib
- Implemented expandable technical details section for file and image information
- Added four informational tabs: Model Information, Risk Assessment Guide, Clinical Context, and Heatmap Explanation
- Enhanced recommendation section with specific actionable steps based on risk level
- Improved visual layout with color-coded risk indicators and progress bars

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Framework**: Streamlit web application framework
- **Rationale**: Streamlit provides rapid development of ML/AI applications with minimal code, ideal for proof-of-concept medical image analysis tools
- **Page Configuration**: Wide layout with custom page title and medical-themed icon (🔬)
- **User Interface Components**: File uploader supporting multiple medical image formats (PNG, JPG, JPEG, BMP, TIFF) with inline help text

**2-Page Navigation Pattern**:
- **Page 1 (Upload)**: `show_upload_page()` - File uploader, image preview, "Analyze Image" button, medical disclaimer, and tool information
- **Page 2 (Results)**: `show_results_page()` - Comprehensive analysis with all visualizations, metrics, recommendations, and PDF download
- **Session State Management**: `st.session_state.current_page` tracks current page ('upload' or 'results')
- **Data Persistence**: `st.session_state.analysis_data` stores all analysis results for display on results page
- **Navigation Flow**: Upload → Analyze → Results → Back to Upload (via buttons on results page)

## Backend Architecture

**Machine Learning Framework**: TensorFlow/Keras
- **Model Type**: Convolutional Neural Network (CNN) for binary classification
- **Architecture Pattern**: Sequential model with standard CNN layers
  - Input layer: 224x224x3 (standard ImageNet size for transfer learning compatibility)
  - Convolutional layers: 2 blocks with increasing filters (32→64) and ReLU activation
  - Pooling layers: MaxPooling for dimensionality reduction
  - Dense layers: Fully connected layer (128 units) with dropout (0.5) for regularization
  - Output layer: Single sigmoid neuron for binary classification
- **Training Configuration**: Adam optimizer with binary cross-entropy loss

**Image Processing Pipeline**:
1. Resize to 224x224 pixels
2. Handle grayscale conversion (duplicate channels) and alpha channel removal
3. Normalize pixel values to [0, 1] range
4. Add batch dimension for model inference

**Model Persistence**: 
- Pre-trained model loaded from `models/breast_cancer_model.keras`
- Streamlit resource caching (`@st.cache_resource`) to avoid reloading on every interaction

## Data Architecture

**Training Data**: Synthetic data generation for demonstration
- **Problem Addressed**: No real medical dataset included (privacy/licensing concerns)
- **Solution**: Procedurally generated random images with different statistical properties for benign/malignant classes
- **Limitations**: Synthetic data won't generalize to real medical images; serves as placeholder for proof-of-concept

**Image Format Support**: Multi-format compatibility to handle various medical imaging standards

## Design Patterns

**Separation of Concerns**:
- `app.py`: User interface and application logic
- `train_model.py`: Model architecture definition and training utilities
- `grad_cam.py`: Gradient-weighted Class Activation Mapping visualization module
- `main.py`: Entry point placeholder (likely unused in current configuration)

**Caching Strategy**: Model loaded once and cached using Streamlit's resource caching to optimize performance across user sessions

**Grad-CAM Implementation**:
- Uses layer indices instead of layer names for compatibility with loaded Sequential models
- Creates functional model from original model's inputs and specified layer outputs
- Generates normalized heatmap using gradient-weighted activations
- Applies jet colormap and overlays on original image with configurable transparency

## Ethical Considerations

**Medical Disclaimer**: Prominently displayed warning that the tool is not for actual medical diagnosis, addressing liability and ethical concerns around AI in healthcare

# External Dependencies

## Python Libraries

**Core ML/AI**:
- TensorFlow (with Keras API): Deep learning framework for model training and inference
- NumPy: Numerical computing for array operations and data manipulation

**Web Application**:
- Streamlit: Web framework for creating the interactive UI

**Image Processing**:
- PIL (Pillow): Python Imaging Library for image loading and preprocessing
- io: Standard library for handling byte streams

## File System Dependencies

**Model Storage**:
- Path: `models/breast_cancer_model.keras`
- Format: Keras native format (.keras)
- Purpose: Stores pre-trained model weights and architecture

## Notes

- No external APIs or cloud services integrated
- No database system currently implemented
- No authentication/authorization system (single-user local application)
- Application designed to run locally or on Replit environment
- Model training (`train_model.py`) appears to be a separate script, not integrated into the main application flow