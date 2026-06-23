import streamlit as st

st.markdown("""
<style>

/* Hide Streamlit toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide top decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide footer */
footer {
    visibility: hidden !important;
}

/* Hide header */
header {
    visibility: hidden !important;
}

/* Hide floating status widget */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Hide any fixed bottom-right buttons */
button[kind="header"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)
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
page_title="Telco Customer Churn Dashboard",
page_icon="📊",
layout="wide",
)

st.markdown("""

<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 100%
    );
}

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.stButton > button{
    width:100%;
    height:60px;
    border-radius:15px;
    border:none;
    font-size:18px;
    font-weight:bold;
    color:white;
    background:linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
}

div[data-testid="stMetric"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
    border:1px solid #334155;
}

</style>

""", unsafe_allow_html=True)

st.markdown("""

<div style="
background:linear-gradient(
135deg,
#1e3c72,
#2a5298
);
padding:35px;
border-radius:20px;
text-align:center;
margin-bottom:20px;
">
<h1 style="color:white;">
📊 Telco Customer Churn Predictor
</h1>
<h4 style="color:white;">
 Customer Retention Analytics Dashboard
</h4>
</div>
""", unsafe_allow_html=True)


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

st.info(f"🚀 Model Ready ({load_time:.2f}s)")
print(f"Model load time: {load_time:.2f} seconds")

st.sidebar.success(f"✅ Model Loaded ({load_time:.2f}s)")
# -----------------------------
# Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Predict",
        "Analytics",
        "About"
    ]
)

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
# Preview Data + Prediction
# -----------------------------
if page == "Predict":

    # Make sure record exists
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

    with st.expander(
        "📄 Customer Information",
        expanded=False
    ):
        st.json(record)


# -----------------------------
# Prediction
# -----------------------------
if page == "Predict":

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

            prob = float(prediction["probability"])

            if prob >= 0.8:
                risk = "🔴 Critical"
            elif prob >= 0.6:
                risk = "🟠 High"
            elif prob >= 0.4:
                risk = "🟡 Medium"
            else:
                risk = "🟢 Low"

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction",
                    prediction["label"]
                )

            with col2:
                st.metric(
                    "Churn Risk",
                    f"{prob:.2%}"
                )

            st.metric(
                "Risk Level",
                risk
            )

            revenue_risk = (
                record["MonthlyCharges"]
                * prob
                * 12
            )

            st.metric(
                "Annual Revenue At Risk",
                f"${revenue_risk:,.2f}"
            )

            recommendations = []

            if record["Contract"] == "Month-to-month":
                recommendations.append(
                    "💰 Offer annual contract discount"
                )

            if record["MonthlyCharges"] > 80:
                recommendations.append(
                    "🎁 Offer discounted subscription plan"
                )

            if record["TechSupport"] == "No":
                recommendations.append(
                    "🛠 Offer premium technical support"
                )

            if record["OnlineSecurity"] == "No":
                recommendations.append(
                    "🔒 Recommend online security package"
                )

            st.subheader(
                "🎯 Retention Recommendations"
            )

            for item in recommendations:
                st.info(item)

            st.subheader("Risk Meter")
            st.progress(prob)

            if prob >= 0.70:
                st.error("⚠ High Churn Risk")
            elif prob >= 0.40:
                st.warning("⚠ Medium Churn Risk")
            else:
                st.success("✅ Customer Likely To Stay")

        except Exception as e:
            st.error(
                f"Prediction failed: {e}"
            )
# -----------------------------
# Analytics Page
# -----------------------------
if page == "Analytics":

    st.title("📈 Model Analytics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Size", "7043")

    with col2:
        st.metric("Churn Rate", "26.5%")

    with col3:
        st.metric("Sampling", "SMOTE")

    with col4:
        st.metric("Deployment", "Streamlit")

    st.divider()

    st.subheader("📊 Model Performance")

    st.info("Add Confusion Matrix, ROC Curve and Feature Importance images here.")

    # Later
    # st.image("artifacts/confusion_matrix.png")
    # st.image("artifacts/roc_curve.png")
# -----------------------------
# About Page
# -----------------------------
if page == "About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ## AI-Powered Customer Churn Prediction & Retention Analytics Platform

    This project predicts whether a telecom customer is likely to churn.

    ### Features
    - End-to-End Machine Learning Pipeline
    - SMOTE for Class Imbalance
    - Hyperparameter Tuning
    - Real-Time Prediction
    - Revenue Risk Estimation
    - Retention Recommendations
    - Interactive Streamlit Dashboard

    ### Tech Stack
    - Python
    - Scikit-Learn
    - Pandas
    - NumPy
    - Streamlit
    - Imbalanced-Learn (SMOTE)

    ### Developed By
    Mareti Satish
    """)