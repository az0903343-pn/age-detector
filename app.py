import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import os

# Page Settings
st.set_page_config(page_title="AI Face Analyzer", page_icon="🤖", layout="centered")

st.title("🤖 AI Face Age & Gender Predictor")
st.write("Apna live photo click karen. AI aap ki exact umar aur gender scan karega.")

# Camera Input
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    # 1. Image Conversion
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # Image ko temporary save karna deepface ke liye
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    with st.spinner('AI Face Analyze kar raha hai...'):
        try:
            # 2. DeepFace AI Analysis
            # Yeh model age aur gender dono predict karta hai
            predictions = DeepFace.analyze(img_path=temp_path, actions=['age', 'gender'], enforce_detection=True)
            
            # Agar multiple faces hon to pehla face select karen
            result = predictions[0]
            
            predicted_age = int(result['age'])
            predicted_gender = result['dominant_gender']
            
            # Age constraints (0-100 logic build)
            final_age = max(0, min(100, predicted_age))
            
            # Urdu/English Styling for Gender
            gender_display = "👨 Male (Mardaana)" if predicted_gender.lower() == "man" else "👩 Female (Zanana)"
            
            # 3. Fancy Output Results Display
            st.success("### 📊 AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age (Exact)", value=f"{final_age} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_display)
                
            st.progress(final_age / 100.0)
            
            # Fun feedback based on exact age
            if final_age == 18:
                st.balloons()
                st.info("Congratulations! Aap bilkul perfect 18 saal ke young adult hain! 🎉")
            elif final_age < 18:
                st.info("Aap abhi teenager/bachay hain! 🧒")
            else:
                st.info("Aap aik mature adult hain! 👨‍💼")
                
        except Exception as e:
            st.error("⚠️ Face clear nahi nazar aaya ya roshni kam hai. Please dobara saaf roshni me photo khinchen!")
            
        finally:
            # Temp file ko delete karna cleanup ke liye
            if os.path.exists(temp_path):
                os.remove(temp_path)
