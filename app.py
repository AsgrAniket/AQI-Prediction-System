import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AQI Predictor", layout="wide")

# ---------------- SAFE MODEL LOADING ----------------
MODEL_PATH = "model/aqi_model.joblib"
SCALER_PATH = "model/scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- FUNCTIONS ----------------
def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Sensitive people should limit prolonged exposure."
    elif aqi <= 200:
        return "Avoid outdoor exercise. Consider a mask."
    elif aqi <= 300:
        return "Stay indoors if possible."
    else:
        return "Hazardous air! Avoid going outside."

# ---------------- TITLE ----------------
st.title("🌍 AQI Prediction Dashboard")
st.markdown("Smart Air Quality Prediction using Machine Learning")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SAMPLE DATA ----------------
if st.button("⚡ Use Sample Data"):
    st.session_state.pm25 = 120.0
    st.session_state.pm10 = 180.0
    st.session_state.no2 = 90.0
    st.session_state.so2 = 40.0
    st.session_state.co = 50.0
    st.session_state.o3 = 60.0

# ---------------- INPUT ----------------
st.subheader("🧪 Enter Pollutant Levels")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0, key="pm25")
    pm10 = st.number_input("PM10", min_value=0.0, key="pm10")
    no2 = st.number_input("NO2", min_value=0.0, key="no2")

with col2:
    so2 = st.number_input("SO2", min_value=0.0, key="so2")
    co = st.number_input("CO", min_value=0.0, key="co")
    o3 = st.number_input("O3", min_value=0.0, key="o3")

# ---------------- VALIDATION ----------------
if pm25 == 0 and pm10 == 0 and no2 == 0:
    st.warning("⚠️ Enter realistic pollutant values")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict AQI"):

    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    input_scaled = scaler.transform(input_data)

    with st.spinner("Predicting AQI..."):
        prediction = model.predict(input_scaled)[0]

    category = get_aqi_category(prediction)

    # Save history
    st.session_state.history.append({
        "AQI": round(prediction, 2),
        "Category": category
    })

    st.markdown("---")
    st.subheader("📊 Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Predicted AQI", round(prediction, 2))

    with col4:
        st.metric("Category", category)

    # Risk Bar
    risk = min(prediction / 500, 1.0)
    st.progress(risk)

    # Color Indicator
    if prediction <= 50:
        st.success("Good 🟢")
    elif prediction <= 100:
        st.info("Satisfactory 🟡")
    elif prediction <= 200:
        st.warning("Moderate 🟠")
    else:
        st.error("Poor / Severe 🔴")

    # ---------------- CHART ----------------
    st.subheader("📈 Pollutant Breakdown")

    df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "Value": [pm25, pm10, no2, so2, co, o3]
    })

    st.bar_chart(df.set_index("Pollutant"))

    # ---------------- HEALTH ----------------
    st.subheader("🩺 Health Advice")
    st.info(health_advice(prediction))

    # ---------------- FEATURE IMPORTANCE ----------------
    if hasattr(model, "feature_importances_"):
        st.subheader("🧠 Feature Importance")

        importance = model.feature_importances_
        features = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        st.pyplot(fig)

    # ---------------- DOWNLOAD REPORT ----------------
    report = f"""
AQI Report

AQI: {round(prediction, 2)}
Category: {category}

PM2.5: {pm25}
PM10: {pm10}
NO2: {no2}
SO2: {so2}
CO: {co}
O3: {o3}
"""

    st.download_button("📄 Download Report", report)

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")
st.sidebar.info("""
AQI Prediction System

✔ Machine Learning Model  
✔ Interactive Dashboard  
✔ Health Insights  
✔ Downloadable Reports  
✔ Prediction History  

Built for real-world usability.
""")