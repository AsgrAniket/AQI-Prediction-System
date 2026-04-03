import streamlit as st
import numpy as np
import pickle
import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import get_aqi_category

# Fix paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'aqi_model.pkl')

# Load model
model = joblib.load('model/aqi_model.joblib')


st.title("Air Quality Index Prediction System")

st.write("Enter pollutant values to predict AQI")

# Inputs
pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no = st.number_input("NO", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
nox = st.number_input("NOx", min_value=0.0)
nh3 = st.number_input("NH3", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)
benzene = st.number_input("Benzene", min_value=0.0)
toluene = st.number_input("Toluene", min_value=0.0)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene]])
    
    prediction = model.predict(input_data)[0]
    category = get_aqi_category(prediction)

    st.subheader(f"Predicted AQI: {round(prediction, 2)}")
    st.subheader(f"Category: {category}")

    # Simple insight
    if prediction > 300:
        st.error("⚠️ Severe pollution level!")
    elif prediction > 200:
        st.warning("⚠️ Poor air quality")
    else:
        st.success("Air quality is acceptable")