import streamlit as st
import pandas as pd
import xgboost as xgb

# Load XGBoost model
model = xgb.XGBRegressor()
model.load_model("finalmodel.json")  # Make sure this file is in the same folder

st.title("🏥 Medical Insurance Charge Predictor")

# User input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])  # Not used in model
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])  # Not used in model

    submit = st.form_submit_button("Predict")

if submit:
    # Preprocess input
    smoker_encoded = 1 if smoker == "yes" else 0
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded]
    })

    # Make prediction
    prediction_usd = model.predict(input_data)[0]
    prediction_inr = prediction_usd * 83  # Convert to INR

    # Show result
    st.success(f"💰 Predicted Insurance Charges: ₹{prediction_inr:,.2f} INR")
