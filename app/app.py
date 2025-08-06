import streamlit as st
import joblib
import pandas as pd

model = joblib.load('xgb_model.pkl')

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Example input fields
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

if st.button("Predict"):
    input_df = pd.DataFrame([[tenure, monthly_charges, contract]], columns=["tenure", "MonthlyCharges", "Contract"])
    prediction = model.predict(input_df)
    st.write("Churn" if prediction[0] == 1 else "No Churn")
