import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, scalers, and polynomial transformer
poly_model = joblib.load('model.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')
poly_reg = joblib.load('poly_features.pkl')

# Define the column names
column_names = [
    'PPG_Signal(mV)', 'Heart_Rate(bpm)', 'Systolic_Peak(mmHg)', 'Diastolic_Peak(mmHg)',
    'Pulse_Area', 'Gender(1 for Male, 0 for Female)', 'Height(cm)', 'Weight(kg)', 'Age Range[1,2,3,4,5]'
]

# Streamlit App Title
st.title("Glucose Level Prediction App")

# Create input fields for each feature
input_values = []
for column in column_names:
    value = st.number_input(f"Enter {column}:", min_value=0.0)
    input_values.append(value)

# Predict Button
if st.button('Predict Glucose Level'):
    # Prepare the input for prediction
    input_df = pd.DataFrame([input_values], columns=column_names)
    input_scaled = scaler_features.transform(input_df)
    input_poly = poly_reg.transform(input_scaled)
    
    # Predict and inverse transform to get the original glucose level
    output_scaled = poly_model.predict(input_poly)
    output = scaler_target.inverse_transform(output_scaled.reshape(-1, 1))
    
    # Display the predicted glucose level
    st.success(f"Predicted Glucose Level: {output[0][0]:.2f} mg/dL")
