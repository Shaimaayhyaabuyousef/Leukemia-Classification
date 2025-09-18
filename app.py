import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import io

# Set page configuration
st.set_page_config(
    page_title="LeukeAI : Leukemia Classification App",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained DenseNet-121 model"""
    try:
        class_names = ["Benign", "Early Pre-B", "Pre-B", "Pro-B"]
        
        # Initialize model architecture
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(class_names))
        
        # Load weights
        model_path = "models/leukeai_densenet121.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model inference - cached for performance"""
    try:
        # Read image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize large images first to improve performance
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Apply CLAHE preprocessing (same as training)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Define transforms
        IMG_SIZE = 224
        test_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL and apply transforms
        pil_img = Image.fromarray(processed_img)
        tensor = test_transforms(pil_img).unsqueeze(0)
        
        return tensor, processed_img, img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None, None

def predict_image(model, input_tensor, class_names):
    """Make prediction on the preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
        return predicted_class, confidence, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def display_class_info():
    """Display information about the classification classes"""
    st.sidebar.markdown("### 📋 Classification Classes")
    
    class_info = {
        "🟢 Benign": "Normal, healthy blood cells",
        "🔴 Early Pre-B": "Early stage B-cell acute lymphoblastic leukemia",
        "🔴 Pre-B": "Pre-B cell acute lymphoblastic leukemia",
        "🔴 Pro-B": "Pro-B cell acute lymphoblastic leukemia"
    }
    
    for class_name, description in class_info.items():
        st.sidebar.markdown(f"**{class_name} :** {description}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🩸 LeukeAI : Leukemia Classification System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload a blood cell microscopy image to classify B-cell Acute Lymphoblastic Leukemia subtypes
        </p>
        <p style="font-style: italic; color: #888;">
            Author: Shimaa Abu Youcef
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar information
    display_class_info()
    
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This application uses a DenseNet-121 model trained to classify blood cell images into four categories:
    - **Benign** (healthy cells)
    - **B-ALL subtypes** (malignant cells)
    
    The model uses CLAHE preprocessing for enhanced contrast and provides confidence scores for each prediction.
    """)
    
    st.sidebar.markdown("### ⚠️ Important Note")
    st.sidebar.warning("""
    This tool is for research and educational purposes only. 
    Always consult healthcare professionals for medical diagnosis and treatment decisions.
    """)
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please ensure 'leukeai_densenet121.pth' is in the same directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a blood cell microscopy image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a microscopy image of blood cells for B-ALL classification (max 10MB)"
        )
        
        # Check file size
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 10:
                st.error("⚠️ File size too large! Please upload an image smaller than 10MB.")
                return
        
        if uploaded_file is not None:
            # Get image bytes for caching
            image_bytes = uploaded_file.getvalue()
            
            # Display original image (optimized)
            st.markdown("#### Original Image")
            display_image = Image.open(uploaded_file)
            # Resize for display if too large
            if max(display_image.size) > 800:
                display_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            st.image(display_image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image (cached)
            with st.spinner("Preprocessing image..."):
                result = preprocess_image(image_bytes)
                
            if result[0] is not None:
                input_tensor, processed_img, original_array = result
                st.markdown("#### Preprocessed Image (CLAHE Enhanced)")
                # Resize processed image for display
                processed_display = Image.fromarray(processed_img)
                if max(processed_display.size) > 800:
                    processed_display.thumbnail((800, 800), Image.Resampling.LANCZOS)
                st.image(processed_display, caption="CLAHE Enhanced", use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            # Get cached preprocessing results
            image_bytes = uploaded_file.getvalue()
            result = preprocess_image(image_bytes)
            
            if result[0] is not None:
                input_tensor, processed_img, original_array = result
                st.markdown("### 🔍 Prediction Results")
                
                # Make prediction (cached)
                with st.spinner("Making prediction..."):
                    predicted_class, confidence, probabilities = predict_image(model, input_tensor, class_names)
                
                if predicted_class is not None:
                    # Display main prediction
                    prediction_text = class_names[predicted_class]
                    
                    # Color coding for results
                    if prediction_text == "Benign":
                        color = "green"
                        icon = "🟢"
                    else:
                        color = "red"
                        icon = "🔴"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: {color}; margin: 0;">{icon} Predicted Class</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{prediction_text}</h2>
                        <p style="margin: 0; font-size: 1.1rem;">Confidence: <strong>{confidence:.2%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display all class probabilities
                    st.markdown("#### 📊 All Class Probabilities")
                    
                    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                        prob_percentage = prob.item() * 100
                        
                        # Create progress bar
                        st.markdown(f"**{class_name}**")
                        st.progress(prob.item())
                        st.markdown(f"*{prob_percentage:.2f}%*")
                        st.markdown("")
                    
                    # Additional information
                    st.markdown("#### 💡 Interpretation")
                    if prediction_text == "Benign":
                        st.info("The model predicts this image shows normal, healthy blood cells.")
                    else:
                        st.warning(f"The model predicts this image shows {prediction_text}, a type of B-cell acute lymphoblastic leukemia.")
                    
                    st.markdown("#### 📈 Model Details")
                    st.markdown("""
                    - **Architecture**: DenseNet-121
                    - **Preprocessing**: CLAHE enhancement
                    - **Input Size**: 224×224 pixels
                    - **Classes**: 4 (Benign + 3 B-ALL subtypes)
                    """)
        
        else:
            st.markdown("### 📋 Instructions")
            st.markdown("""
            1. **Upload an image** using the file uploader on the left
            2. **Wait for preprocessing** - the image will be enhanced using CLAHE
            3. **View results** - see the predicted class with confidence scores
            4. **Interpret carefully** - this is for research/educational use only
            
            **Supported formats**: PNG, JPG, JPEG
            
            **Best results**: Use clear microscopy images of blood smears
            """)

if __name__ == "__main__":
    main()