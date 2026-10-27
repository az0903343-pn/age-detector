import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import time

# Page configuration
st.set_page_config(page_title="Real AI Face Analyzer", page_icon="🤖", layout="centered")

st.title("🤖 Real AI Face Age & Gender Predictor")
st.write("Google Vision Transformer (ViT) Core Model. Yeh aap ke face ke real features ko deeply scan karta hai.")

# 1. Loading Real Machine Learning Pipelines (Cached for smooth performance)
@st.cache_resource
def load_real_ai_models():
    # 100% Active and Public High-Precision Transformers Models
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
    gender_pipe = pipeline("image-classification", model="rizvandg/vit-gender-classification")
    return age_pipe, gender_pipe

try:
    age_classifier, gender_classifier = load_models = load_real_ai_models()
    st.success("🎯 Genuine AI Neural Networks Connected Successfully!")
except Exception as e:
    st.error(f"AI Server Connection Error: {e}")

# 2. Camera Input Component
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    image = Image.open(img_file)
    
    # Progress UI for better experience
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📸 Image matrix extract ho raha hai...")
    time.sleep(1.0)
    progress_bar.progress(40)
    
    status_text.text("🧠 Deep Neural Network features scan kar raha hai...")
    
    try:
        # 3. Real AI Model Inference (No Fake Math, Actual Recognition)
        age_predictions = age_classifier(image)
        gender_predictions = gender_classifier(image)
        
        progress_bar.progress(100)
        status_text.empty()
        
        # Extracting top results (highest confidence)
        top_age = age_predictions[0]
        top_gender = gender_predictions[0]
        
        age_label = top_age['label']       # Outputs ranges like: "15-20", "25-32", "38-43"
        gender_label = top_gender['label'] # Outputs: "male" or "female"
        
        # Formatting gender text
        if gender_label.lower() in ['male', 'man', 'm']:
            gender_display = "👨 Male"
        else:
            gender_display = "👩 Female"
            
        # 4. Displaying True Results
        st.success("### 📊 Live AI Analysis Results:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🎯 Predicted Age Range", value=f"{age_label} Years Old")
        with col2:
            st.metric(label="🧬 Gender Detected", value=gender_display)
            
        # Showing confidence breakdown so user knows it's real AI
        st.write("---")
        st.subheader("💡 Model Recognition Details:")
        st.write(f"Age Match Confidence: **{round(top_age['score']*100, 2)}%**")
        st.write(f"Gender Match Confidence: **{round(top_gender['score']*100, 2)}%**")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"⚠️ Model Processing Error: Please capture again under clear light! ({e})")
