import streamlit as st
import joblib
import numpy as np

# Page setup
st.set_page_config(
    page_title="Stroke Prediction AI",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 AI Stroke Prediction System")
st.markdown("### Enter patient medical information")

# Load model
model = joblib.load("stroke_model.pkl")

# UI Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

with col2:
    avg_glucose_level = st.number_input(
        "Average Glucose Level", value=100.0
    )
    bmi = st.number_input("BMI", value=25.0)

st.divider()

# Prediction button
if st.button("🔍 Predict Stroke Risk"):

    data = np.array([[age,
                      hypertension,
                      heart_disease,
                      avg_glucose_level,
                      bmi]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

st.caption("AI Medical Assistant • Machine Learning Project")
