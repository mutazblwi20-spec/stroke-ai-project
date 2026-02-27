import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Stroke Prediction AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🧠 AI Stroke Prediction System")
st.markdown("### Enter patient medical information")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("stroke_model.pkl")

model = load_model()

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

with col2:
    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Stroke Risk"):

    # IMPORTANT:
    # We recreate ALL features used during training
    data = pd.DataFrame({

        "gender": [1],              # default
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [1],
        "work_type": [2],
        "Residence_type": [1],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [1]

    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High Stroke Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Stroke Risk ({probability*100:.2f}%)")

st.caption("AI Medical Assistant • Machine Learning Project")
