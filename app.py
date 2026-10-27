import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Age Predictor", page_icon="🎯")
st.title("🎯 Real-Time Face Age Predictor (0-100)")

@st.cache_resource
def load_model():
    # Aap ki downloaded h5 file isi folder me honi chahiye
    return tf.keras.models.load_model("age_prediction_model.h5")

model = load_model()

# Streamlit Native Camera Input
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    # Image pipeline conversion
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing to match Kaggle structure (64x64)
    img_resized = cv2.resize(img_rgb, (64, 64)) / 255.0
    img_final = np.expand_dims(img_resized, axis=0)
    
    # Predict
    pred = model.predict(img_final)
    final_age = max(0, min(100, int(round(pred[0][0]))))
    
    # Fancy Output UI
    st.success(f"### 🎯 Predicted Age: {final_age} Years Old")
    st.progress(final_age / 100.0)
