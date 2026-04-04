import streamlit as st
import numpy as np
import pandas as pd
import joblib
from src.utils import get_aqi_category

# Page config
st.set_page_config(page_title="AQI Predictor", layout="wide")

# Load model
model = joblib.load('model/aqi_model.joblib')

# Title
st.title("🌍 Air Quality Index Prediction System")
st.markdown("Predict AQI based on pollutant levels using Machine Learning")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Input Pollutants")
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)

with col2:
    st.subheader("🧪 More Inputs")
    so2 = st.number_input("SO2", min_value=0.0)
    co = st.number_input("CO", min_value=0.0)
    o3 = st.number_input("O3", min_value=0.0)

# Predict button
if st.button("🚀 Predict AQI"):

    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]
    category = get_aqi_category(prediction)

    st.markdown("---")
    st.subheader("📊 Results")

    # Metrics
    col3, col4 = st.columns(2)

    with col3:
        st.metric("Predicted AQI", round(prediction, 2))

    with col4:
        st.metric("Category", category)

    # Color indicator
    if prediction <= 50:
        st.success("Good 🟢")
    elif prediction <= 100:
        st.info("Satisfactory 🟡")
    elif prediction <= 200:
        st.warning("Moderate 🟠")
    else:
        st.error("Poor 🔴")

    # Chart
    st.subheader("📈 Pollutant Levels")
    data = {
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "Value": [pm25, pm10, no2, so2, co, o3]
    }

    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Pollutant"))

    # Health suggestion
    st.subheader("🩺 Health Advice")

    if prediction <= 50:
        st.write("Air quality is good. Enjoy outdoor activities.")
    elif prediction <= 100:
        st.write("Air quality is acceptable. Sensitive people should take care.")
    elif prediction <= 200:
        st.write("Limit prolonged outdoor exertion.")
    else:
        st.write("Avoid outdoor activities. Wear a mask.")