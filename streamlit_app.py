from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from churn_pipeline import (
    load_model_bundle,
    parse_input_json,
    predict_churn,
)

MODEL_PATH = Path("artifacts") / "telco_churn_model.joblib"

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Telco Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn.")

# -----------------------------
# Load Model Once (Cached)
# -----------------------------
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
        st.stop()

    return load_model_bundle(MODEL_PATH)


with st.spinner("Loading model..."):
    start = time.time()
    bundle = load_bundle()
    load_time = time.time() - start

st.write(f"Model load time: {load_time:.2f} seconds")
print(f"Model load time: {load_time:.2f} seconds")

st.sidebar.success(f"✅ Model Loaded ({load_time:.2f}s)")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Details")


def ui_input():
    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.sidebar.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.sidebar.selectbox(
        "Partner",
        ["Yes", "No"],
        index=1
    )

    dependents = st.sidebar.selectbox(
        "Dependents",
        ["Yes", "No"],
        index=1
    )

    tenure = st.sidebar.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone = st.sidebar.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple = st.sidebar.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.sidebar.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.sidebar.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.sidebar.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ]
    )

    paperless = st.sidebar.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    monthly = st.sidebar.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.sidebar.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(monthly * max(tenure, 1))
    )

    return {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }


# -----------------------------
# Input Source
# -----------------------------
uploaded = st.sidebar.file_uploader(
    "Upload Customer JSON",
    type=["json"]
)

if uploaded:
    try:
        raw = uploaded.read().decode("utf-8")
        record = parse_input_json(raw)
    except Exception as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()
else:
    record = ui_input()

# -----------------------------
# Preview Data
# -----------------------------
with st.expander("📄 Customer Record", expanded=False):
    st.json(record)

# -----------------------------
# Prediction
# -----------------------------
if st.button(
    "🚀 Predict Churn",
    use_container_width=True,
):
    try:
        with st.spinner("Predicting..."):
            prediction = predict_churn(
                bundle,
                record,
            )

        st.success(
            f"Prediction: {prediction['label']}"
        )

        st.metric(
            "Churn Probability",
            f"{prediction['probability']:.2%}"
        )

        if "probabilities" in prediction:
            st.subheader("Probability Breakdown")
            st.json(
                prediction["probabilities"]
            )

    except Exception as e:
        st.error(
            f"Prediction failed: {e}"
        )