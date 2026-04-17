import streamlit as st
import numpy as np
import cv2
import joblib
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# 🔇 Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# LOAD MODELS
# =========================
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
cnn_model = load_model("cnn_feature_extractor.h5")

# Auto detect input size
IMG_SIZE = cnn_model.input_shape[1]

# Classes
classes = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]

# =========================
# UI
# =========================
st.set_page_config(page_title="Pulmonary Disease Classifier", layout="centered")

st.title("🩺 Pulmonary Disease Classifier (DenseNet + SVM)")
st.write("Upload a chest X-ray to detect pulmonary disease")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", channels="BGR")

    # Preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_processed = preprocess_input(img_resized)
    img_processed = np.expand_dims(img_processed, axis=0)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    features = cnn_model.predict(img_processed)

    # Flatten (safety)
    features = features.reshape(features.shape[0], -1)

    st.write("🔍 Feature shape before PCA:", features.shape)  # debug

    # =========================
    # SCALING + PCA
    # =========================
    features = scaler.transform(features)
    features = pca.transform(features)

    st.write("🔍 Feature shape after PCA:", features.shape)  # debug

    # =========================
    # PREDICTION
    # =========================
    prediction = svm_model.predict(features)[0]
    probs = svm_model.predict_proba(features)[0]

    confidence = np.max(probs) * 100

    # =========================
    # DISPLAY
    # =========================
    st.success(f"Prediction: {classes[prediction]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Class probabilities
    st.subheader("📊 Class Probabilities")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[i]*100:.2f}%")

    # Progress bars
    st.subheader("📈 Confidence Visualization")
    for i, cls in enumerate(classes):
        st.progress(float(probs[i]))

    # Confidence message
    if confidence > 85:
        st.success("High Confidence Prediction ✅")
    else:
        st.warning("Low Confidence ⚠️")