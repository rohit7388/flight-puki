import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="FraudShield", layout="wide")
st.title("💳 FraudShield: Real-Time Fraud Detection")

# Load model and scaler
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    use_scaler = True
except:
    scaler = None
    use_scaler = False

st.write("Enter transaction details:")

duration = st.number_input("Duration", min_value=0.0)
days_left = st.number_input("Days Left", min_value=0.0)
price = st.number_input("Price ($)", min_value=0.0)

features = np.array([[duration, days_left, price]])
if use_scaler:
    features = scaler.transform(features)

if st.button("Predict Fraud"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Transaction is likely safe.