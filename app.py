import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import os

# Page configuration
st.set_page_config(page_title="True Deep Learning Face Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 True AI Face Age & Gender Predictor")
st.write("Real DeepFace Neural Network. Apna live photo click karen, system unique biometric signatures detect karega.")

# Camera Input
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    # Converting stream to computer vision array format safely
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Image ko temporary save karna deepface pipeline ke liye
    temp_path = "live_capture.jpg"
    cv2.imwrite(temp_path, cv2_img)
    
    with st.spinner('AI deeply scan aur neural network analyze kar raha hai...'):
        try:
            # Running True DeepFace AI Analysis for Age and Gender
            predictions = DeepFace.analyze(img_path=temp_path, actions=['age', 'gender'], enforce_detection=False)
            
            # Extracting first face detected metrics
            if isinstance(predictions, list):
                result = predictions[0]
            else:
                result = predictions
                
            final_age = int(result['age'])
            predicted_gender = result['dominant_gender']
            
            # Formatting gender labels into readable Urdu/English strings
            if predicted_gender.lower() in ['man', 'male']:
                gender_display = "👨 Male"
            else:
                gender_display = "👩 Female"
                
            # Results Presentation View
            st.success("### 📊 Live AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age (Exact)", value=f"{final_age} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_display)
                
            st.progress(max(0, min(100, final_age)) / 100.0)
            
        except Exception as e:
            st.error(f"⚠️ Prediction Error: Make sure your face is visible under clear conditions! ({e})")
            
        finally:
            # Temporary file cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
