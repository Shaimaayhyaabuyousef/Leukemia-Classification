# B-ALL Classification Web App

A Streamlit web application for B-cell Acute Lymphoblastic Leukemia (B-ALL) classification using the trained DenseNet-121 model.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   
   Or on Windows, double-click `run_app.bat`

3. **Open your browser** to the displayed URL (usually `http://localhost:8501`)

## Features

- **Easy Upload**: Drag and drop blood cell microscopy images
- **Real-time Processing**: CLAHE preprocessing applied automatically
- **Visual Results**: See both original and enhanced images
- **Detailed Predictions**: Confidence scores for all classes
- **Educational Interface**: Information about each B-ALL subtype
- **Professional Design**: Clean, medical-grade interface

## Requirements

- Python 3.8+
- Trained model file: `leukeai_densenet121.pth` (must be in same directory)
- Dependencies listed in `requirements.txt`

## Usage

1. Upload a blood cell microscopy image (PNG, JPG, or JPEG)
2. Wait for automatic preprocessing with CLAHE enhancement
3. View prediction results with confidence scores
4. Interpret results (for research/educational purposes only)

## Important Notes

- This application is for research and educational purposes only
- Always consult healthcare professionals for medical diagnosis
- Use clear microscopy images of blood smears for best results
- The model classifies into 4 categories: Benign, Early Pre-B, Pre-B, Pro-B

**Author**: Shimaa Abu Youcef