import streamlit as st
import joblib
import numpy as np

# Load saved objects
model = joblib.load("models/bike_demand_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/feature_selector.pkl")

st.title("🚲 Bike Sharing Demand Prediction")

season = st.selectbox("Season (1–4)", [1, 2, 3, 4])
holiday = st.selectbox("Holiday", [0, 1])
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation (1–4)", [1, 2, 3, 4])
temp = st.slider("Temperature", 0.0, 1.0)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0)
hum = st.slider("Humidity", 0.0, 1.0)
windspeed = st.slider("Windspeed", 0.0, 1.0)
weekday = st.selectbox("Weekday (0=Mon)", [0, 1, 2, 3, 4, 5, 6])
month = st.selectbox("Month", list(range(1, 13)))

if st.button("Predict Bike Demand"):
    input_data = np.array(
        [
            [
                season,
                holiday,
                workingday,
                weathersit,
                temp,
                atemp,
                hum,
                windspeed,
                weekday,
                month,
            ]
        ]
    )

    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)

    prediction = model.predict(input_selected)

    st.success(f"🚴 Predicted Bike Demand: {int(prediction[0])}")
