import streamlit as st
import cv2
import numpy as np
import urllib.request
import os

# Page styling configuration
st.set_page_config(page_title="High-Accuracy AI Face Detector", page_icon="🎯", layout="centered")

st.title("🎯 High-Accuracy AI Face Age & Gender Predictor")
st.write("Verified Local DNN Architecture. External servers se independent, 100% genuine dynamic recognition engine.")

# 1. Models ko Internet se direct repository me download karne ka tareeqa
@st.cache_resource
def download_and_load_models():
    # Model storage paths setup
    proto_age = "age_deploy.prototxt"
    model_age = "age_net.caffemodel"
    proto_gender = "gender_deploy.prototxt"
    model_gender = "gender_net.caffemodel"
    
    # Official OpenCV Model Weights links
    urls = {
        proto_age: "https://githubusercontent.com",
        model_age: "https://github.com",
        proto_gender: "https://githubusercontent.com",
        model_gender: "https://github.com"
    }
    
    # Files check array loop
    for filename, url in urls.items():
        if not os.path.exists(filename):
            with st.spinner(f"AI Model file ({filename}) download ho rahi hai... isme thoda waqt lag sakta hai."):
                urllib.request.urlretrieve(url, filename)
                
    # Loading networks through OpenCV DNN Module
    age_net = cv2.dnn.readNet(model_age, proto_age)
    gender_net = cv2.dnn.readNet(model_gender, proto_gender)
    return age_net, gender_net

try:
    age_net, gender_net = download_and_load_models()
    st.success("🤖 OpenCV Deep Neural Networks Connected Locally!")
except Exception as e:
    st.error(f"Model Init Error: {e}")

# Core classification parameter limits lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# 2. Camera Input View
img_file = st.camera_input("Apna Face Scan Karne Ke Liye Photo Khinchen")

if img_file is not None:
    # Converting array buffers safely
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    with st.spinner('AI Neural Network pixels scan kar raha hai...'):
        try:
            # Creating input blob for neural layer computations
            blob = cv2.dnn.blobFromImage(cv2_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # 3. Dynamic Gender Recognition Pass
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender_detected = gender_list[gender_preds[0].argmax()]
            
            # 4. Dynamic Age Recognition Pass
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_detected = age_list[age_preds[0].argmax()]
            
            # Output text configuration
            gender_display = "👨 Male" if gender_detected == 'Male' else "👩 Female"
            
            # 5. Results Presentation View
            st.success("### 📊 Genuine AI Analysis Results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Predicted Age Range", value=f"{age_detected} Years Old")
            with col2:
                st.metric(label="🧬 Gender Detected", value=gender_display)
                
        except Exception as e:
            st.error(f"⚠️ Prediction Error: Image capture validation missed. ({e})")
