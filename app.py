import joblib
import pandas as pd
import streamlit as st

# Load the model and preprocessors using joblib
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
features = joblib.load("feature_columns.pkl")

st.title("Customer Lifetime Value Analysis")

recency = st.number_input("Recency", 0, 1000, 8)
frequency = st.number_input("Frequency", 0, 1000, 12)
monetary = st.number_input("Monetary", 0.0, 1000000.0, 4200.0)

if st.button("Predict Segment"):
    df = pd.DataFrame([[recency, frequency, monetary]], columns=features)
    pred = model.predict(df)
    seg = label_encoder.inverse_transform(pred)[0]
    st.success(f"Predicted Segment: {seg}")