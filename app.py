import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the web page title and styling
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.write("Enter the vehicle details below to estimate its market value.")

# Load the trained pipeline safely
@st.cache_resource
def load_model():
    return joblib.load('xgboost_car_pipeline.pkl')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model pipeline file: {e}")
    st.stop()

# Core engineering mappings to capture brand premium dynamically
COMPANY_TIERS = {
    'Jaguar': 2495000.0, 'Land': 2100000.0, 'Mini': 1891111.0, 'Volvo': 1850000.0, 
    'Audi': 1476909.0, 'Mitsubishi': 1298333.0, 'Mercedes': 1170667.0, 'BMW': 1049875.0, 
    'Jeep': 950000.0, 'Force': 572500.0, 'Toyota': 524443.0, 'Mahindra': 522028.0, 
    'Ford': 479799.0, 'Nissan': 460666.0, 'Renault': 460512.0, 'Volkswagen': 407105.0, 
    'Honda': 347632.0, 'Hyundai': 318402.0, 'Hindustan': 303333.0, 'Maruti': 269837.0, 
    'Datsun': 255714.0, 'Skoda': 251191.0, 'Chevrolet': 202085.0, 'Tata': 200939.0, 
    'Fiat': 109875.0
}

st.header("Vehicle Specifications")
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Car Company", sorted(list(COMPANY_TIERS.keys())))
    name = st.text_input("Enter Car Model / Trim Name", placeholder="e.g., Swift VXI")
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])

with col2:
    year = st.number_input("Year of Manufacture", min_value=1980, max_value=2019, value=2015, step=1)
    kms_driven = st.number_input("Total Kilometers Driven", min_value=0, max_value=1000000, value=30000, step=5000)

if st.button("Calculate Estimated Resale Value", type="primary"):
    if not name.strip():
        st.warning("Please enter a valid Car Model/Trim name before predicting.")
    else:
        # Match the feature engineering logic used during model training
        car_age = 2019 - year
        kms_per_year = kms_driven / (car_age + 0.1)
        company_tier = COMPANY_TIERS.get(company, 300000.0)
        
        input_data = pd.DataFrame([{
            'name': name.strip(),
            'company': company,
            'fuel_type': fuel_type,
            'kms_driven': kms_driven,
            'car_age': car_age,
            'kms_per_year': kms_per_year,
            'company_tier': company_tier
        }])
        
        # Make the log prediction and convert it back to real Rupees
        log_prediction = pipeline.predict(input_data)[0]
        final_price = np.expm1(log_prediction)
        
        st.success("### Prediction Complete!")
        st.metric(
            label=f"Estimated Resale Price for {company} {name}", 
            value=f"Rs. {final_price:,.2f}"
        )
        st.caption("Note: This estimation is powered by a fine-tuned XGBoost Regressor pipeline with a 77.1% accuracy rating.")
        
