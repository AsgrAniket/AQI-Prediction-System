import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import requests

mode = st.radio("Select Mode", ["Manual Input", "Live City Data"])

CITIES = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Ahmedabad": (23.02, 72.57),
    "Bangalore": (12.97, 77.59),
    "Chennai": (13.08, 80.27)
}

API_KEY = "555ee96835fdfc8300c15d5caf5083681af5083681"

def get_live_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("❌ API request failed")
        st.stop()

    data = response.json()

    # 🔥 DEBUG PRINT (remove later)
    # st.write(data)

    if "list" not in data:
        st.error(f"❌ API Error: {data}")
        st.stop()

    components = data["list"][0]["components"]

    return {
        "pm25": components.get("pm2_5", 0),
        "pm10": components.get("pm10", 0),
        "no2": components.get("no2", 0),
        "so2": components.get("so2", 0),
        "co": components.get("co", 0),
        "o3": components.get("o3", 0)
    }

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AQI Predictor", layout="wide")

# ---------------- SAFE MODEL LOADING ----------------
MODEL_PATH = "model/aqi_model.joblib"
SCALER_PATH = "model/scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found. Please check deployment.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------- FUNCTIONS ----------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Sensitive people should limit prolonged outdoor exposure."
    elif aqi <= 200:
        return "Avoid outdoor exercise. Wear a mask."
    elif aqi <= 300:
        return "Stay indoors. Use air purifier if possible."
    else:
        return "Hazardous air! Avoid going outside."

# ---------------- UI ----------------
st.title("🌍 Air Quality Index Prediction System")
st.markdown("Predict AQI based on pollutant levels using Machine Learning")

# ---------------- SAMPLE DATA ----------------
if st.button("⚡ Use Sample Data"):
    st.session_state.pm25 = 120.0
    st.session_state.pm10 = 180.0
    st.session_state.no2 = 90.0
    st.session_state.so2 = 40.0
    st.session_state.co = 50.0
    st.session_state.o3 = 60.0

# ---------------- INPUT ----------------
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
    st.warning("⚠️ Enter meaningful pollutant values")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict AQI"):

    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])

    input_scaled = scaler.transform(input_data)

    with st.spinner("Predicting AQI..."):
        prediction = model.predict(input_scaled)[0]

    category = get_aqi_category(prediction)

    st.markdown("---")
    st.subheader("📊 Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Predicted AQI", round(prediction, 2))

    with col4:
        st.metric("Category", category)

    # Color Output
    if prediction <= 50:
        st.success("Good 🟢")
    elif prediction <= 100:
        st.info("Satisfactory 🟡")
    elif prediction <= 200:
        st.warning("Moderate 🟠")
    else:
        st.error("Poor / Severe 🔴")

    # ---------------- CHART ----------------
    st.subheader("📈 Pollutant Levels")

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

if mode == "Live City Data":
    city = st.selectbox("Select City", list(CITIES.keys()))
    
    lat, lon = CITIES[city]
    
    live_data = get_live_aqi(lat, lon)

    pm25 = live_data["pm25"]
    pm10 = live_data["pm10"]
    no2 = live_data["no2"]
    so2 = live_data["so2"]
    co = live_data["co"]
    o3 = live_data["o3"]

    st.success(f"Using live data for {city}")

city1 = st.selectbox("City 1", list(CITIES.keys()))
city2 = st.selectbox("City 2", list(CITIES.keys()))

data = pd.DataFrame({
    city1: list(get_live_aqi(*CITIES[city1]).values()),
    city2: list(get_live_aqi(*CITIES[city2]).values())
}, index=["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])

st.write(data)
st.bar_chart(data)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")
st.sidebar.info("""
This project predicts AQI using Machine Learning.

Features:
✔ Random Forest Model  
✔ Real-time Prediction  
✔ Health Suggestions  
✔ Interactive Dashboard  
""")

 