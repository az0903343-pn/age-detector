import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Page configuration setup
st.set_page_config(page_title="Stable AI Face Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 True AI Face Age & Gender Predictor")
st.write("Real Neural Network Integration. Apna live photo click karen, AI aap ki genuine umar aur gender scan karega.")

# 1. Models ko Load karna (Cached for fast performance)
@st.cache_resource
def load_models():
    # 100% Public aur verified models jo kabhi crash nahi karenge
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier")
    gender_pipe = pipeline("image-classification", model="ufdatastudio/vit-gender")
    return age_pipe, gender_pipe

try:
    age_classifier, gender_classifier = load_models()
    st.success("🤖 Genuine AI Neural Networks Connected Successfully!")
except Exception as e:
    st.error(f"AI Server Connection Error: {e}")

# 2. Camera Input Component
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    image = Image.open(img_file)
    
    with st.spinner('AI Neural Network deeply scan kar raha hai...'):
        try:
            # 3. Real AI Inference Tracking
            age_results = age_classifier(image)
            gender_results = gender_classifier(image)
            
            # Highest confidence predictions nikalna (Pehla result highest score hota hai)
            top_age = age_results[0]
            top_gender = gender_results[0]
            
            top_age_label = top_age['label']       # Format jaise: "15-20", "25-32", etc.
            top_gender_label = top_gender['label'] # Format: "male" ya "female"
            
            # Gender display output mapping
            if top_gender_label.lower() in ['male', 'man', 'm']:
                gender_display = "👨 Male"
            else:
                gender_display = "👩 Female"
                
            # 4. Results Presentation View
            st.success("### 📊 Live AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age Range", value=f"{top_age_label} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_display)
                
            # Detailed analytical scores analytics view
            st.write("---")
            st.subheader("💡 Model Confidence Scores:")
            st.write(f"Age Prediction Match: **{top_age_label}** ({round(top_age['score']*100, 2)}% accuracy score)")
            st.write(f"Gender Prediction Match: **{top_gender_label}** ({round(top_gender['score']*100, 2)}% accuracy score)")
            
        except Exception as e:
            st.error(f"⚠️ Prediction Processing Error: {e}")
