import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="True AI Face Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 True AI Face Age & Gender Predictor")
st.write("Real Neural Network Integration. Apna live photo click karen, AI aap ki genuine umar aur gender scan karega.")

# 1. Models ko Load karna (Cached for fast performance)
@st.cache_resource
def load_models():
    # Sahi configuration wale updated models jo crash nahi karenge
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
    gender_pipe = pipeline("image-classification", model="risknat/vit-gender-classifier")
    return age_pipe, gender_pipe

try:
    age_classifier, gender_classifier = load_models()
    st.success("🤖 Genuine AI Neural Networks Connected Successfully!")
except Exception as e:
    st.error(f"AI Server Connection Error: {e}")

# 2. Camera Input
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    image = Image.open(img_file)
    
    with st.spinner('AI Neural Network deeply scan kar raha hai...'):
        try:
            # 3. Real AI Inference
            age_results = age_classifier(image)
            gender_results = gender_classifier(image)
            
            # Highest confidence score wali predictions nikalna
            top_age_label = age_results[0]['label']       # Format jaise: "15-20", "25-32", "38-43" ya single age
            top_gender_label = gender_results[0]['label'] # Format: "male" ya "female"
            
            # Gender formatting
            if top_gender_label.lower() in ['male', 'man', 'M']:
                gender_detected = "👨 Male"
            else:
                gender_detected = "👩 Female"
                
            # 4. Results Display
            st.success("### 📊 Live AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age Range", value=f"{top_age_label} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_detected)
                
            # Extra analytical details display
            st.write("---")
            st.subheader("💡 Model Confidence Scores:")
            st.write(f"Age Prediction: **{age_results[0]['label']}** ({round(age_results[0]['score']*100, 2)}% match)")
            st.write(f"Gender Prediction: **{gender_results[0]['label']}** ({round(gender_results[0]['score']*100, 2)}% match)")
            
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
