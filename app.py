import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Load the saved model and transformer
@st.cache_resource
def load_assets():
    with open("xgboost_car_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("column_transformer.pkl", "rb") as f:
        transformer = pickle.load(f)
    return model, transformer

try:
    model, transformer = load_assets()
except FileNotFoundError:
    st.error("Error: Please ensure 'xgboost_car_model.pkl' and 'column_transformer.pkl' are in the same folder.")
    st.stop()

# Application Header
st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the vehicle details below to estimate its market resale value.")
st.write("---")

# User Input Form
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Car Brand", [
        'Hyundai', 'Mahindra', 'Maruti', 'Ford', 'Toyota', 'Honda', 
        'Renault', 'Volkswagen', 'Nissan', 'Tata', 'Chevrolet', 
        'Datsun', 'Skoda', 'Fiat', 'Jeep', 'Audi', 'BMW', 'Mercedes', 'Volvo'
    ])
    
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'LPG'])

with col2:
    kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=5, step=1)

st.write("---")

# Prediction Trigger
if st.button("Predict Resale Price", type="primary"):
    # 1. Create a DataFrame out of user inputs matching original feature names
    input_data = pd.DataFrame([{
        'kms_driven': kms_driven,
        'car_age': car_age,
        'fuel_type': fuel_type,
        'company': company
    }])
    
    # 2. Transform the input features using the loaded ColumnTransformer
    input_encoded = transformer.transform(input_data)
    
    # 3. Generate prediction
    predicted_price = model.predict(input_encoded)[0]
    
    # 4. Display result cleanly
    if predicted_price < 0:
        st.warning("⚠️ The combination of high age/mileage suggests this vehicle has depreciated fully past baseline value.")
    else:
        st.success(f"### Estimated Resale Price: ₹ {predicted_price:,.2f}")
