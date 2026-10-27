import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Page configuration and clean title UI
st.set_page_config(page_title="Official AI Face Detector", page_icon="🧠", layout="centered")

st.title("🧠 True AI Face Age & Gender Predictor")
st.write("Real-time Hugging Face Network integration. Apna live photo click karen, system unique neural features scan karega.")

# 1. Loading Highly Stable Lightweight Models (No server crash)
@st.cache_resource
def load_stable_ai_models():
    # Standard, verified and fast models that run perfectly on Streamlit Cloud
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
    gender_pipe = pipeline("image-classification", model="ufdatastudio/vit-gender")
    return age_pipe, gender_pipe

try:
    age_classifier, gender_classifier = load_stable_ai_models()
    st.success("🎯 Genuine AI Neural Networks Connected Successfully!")
except Exception as e:
    st.error(f"AI Server Connection Error: {e}")

# 2. Native Streamlit Camera Component (Bypasses Kaggle/IPython Errors)
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    image = Image.open(img_file)
    
    with st.spinner('AI Neural Network deeply scan kar raha hai...'):
        try:
            # 3. Running Actual Model Inference (Real Brain Processing)
            age_results = age_classifier(image)
            gender_results = gender_classifier(image)
            
            # Extracting highest confidence prediction tags safely
            top_age_label = age_results[0]['label']       # Output e.g., "15-20", "38-43"
            top_gender_label = gender_results[0]['label'] # Output: "male" or "female"
            
            # Standard formatting logic
            gender_display = "👨 Male" if top_gender_label.lower() in ['male', 'man', 'm'] else "👩 Female"
            
            # 4. Results Presentation View
            st.success("### 📊 Live AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age Range", value=f"{top_age_label} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_display)
                
            # Transparency Metrics display
            st.write("---")
            st.subheader("💡 Model Match Reliability Score:")
            st.write(f"Age Class Accuracy Match: **{top_age_label}** ({round(age_results[0]['score']*100, 2)}%)")
            st.write(f"Gender Class Accuracy Match: **{top_gender_label}** ({round(gender_results[0]['score']*100, 2)}%)")
            
        except Exception as e:
            st.error(f"⚠️ Prediction Processing Error: Please make sure your full face is visible. Details: {e}")
