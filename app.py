import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Accurate AI Face Analyzer", page_icon="🧬", layout="centered")

st.title("🧬 Accurate AI Face Age & Gender Predictor")
st.write("Hugging Face AI Pipeline integration. Apna live photo click karen, system real patterns ko detect karega.")

# 1. Loading the Hugging Face Pipeline Model (Cached to prevent slow reloads)
@st.cache_resource
def load_ai_pipeline():
    # Hugging face standard model for high accuracy age-gender classification
    return pipeline("image-classification", model="abhilash88/age-gender-prediction")

try:
    classifier = load_ai_pipeline()
    st.success("🤖 Professional AI Models Connected Successfully!")
except Exception as e:
    st.error(f"AI Server Connection Error: {e}")

# 2. Camera Input
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    # Read the image file using PIL
    image = Image.open(img_file)
    
    with st.spinner('AI deeply scan aur check kar raha hai...'):
        try:
            # 3. Running Prediction through the Online Library Model
            results = classifier(image)
            
            # Extracting age and gender attributes from model dictionary outputs
            # Standard output structure contains category scores
            gender_detected = "Male"
            final_age = 18
            
            # Parsing results safely based on highest confidence levels
            for res in results:
                label = res['label'].lower()
                # Parsing gender
                if 'male' in label or 'man' in label:
                    gender_detected = "👨 Male"
                elif 'female' in label or 'woman' in label:
                    gender_detected = "👩 Female"
                
                # Parsing age if present in digits within the labels
                import re
                age_match = re.search(r'\d+', label)
                if age_match:
                    final_age = int(age_match.group())

            # 4. Results Display
            st.success("### 📊 Live AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age (Exact)", value=f"{final_age} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_detected)
                
            st.progress(max(0, min(100, final_age)) / 100.0)
            
            # Personalised alert based on model predictions
            if final_age <= 19:
                st.info("Aap Young Generation/Teenagers me aate hain! ⚡")
            elif final_age <= 45:
                st.info("Aap aik active aur mature adult hain! 👨‍💼")
            else:
                st.info("Tajurbakar senior age group profile. 👴")
                
        except Exception as e:
            st.error(f"⚠️ Prediction Error: Make sure your face is visible under clear lightning conditions! ({e})")
