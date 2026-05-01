import streamlit as st
import pandas as pd
import joblib
import os
import warnings

# ================================
# SETTINGS
# ================================
warnings.filterwarnings("ignore")

DATA_PATH = "fraud_detection.csv"
TARGET_COL = "fraud_risk"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "XGBoost.joblib")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2E86C1;}
    h2 {color: #117A65;}
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stSuccess {color: #117A65;}
    .stError {color: #C0392B;}
    </style>
""", unsafe_allow_html=True)

# Unified title and tagline
st.title("Fraud Detection App")
st.write("An Interactive Application to Predict Fraud Detection")

# ================================
# LOAD MODEL
# ================================
if not os.path.exists(MODEL_FILE):
    st.error("⚠️ Trained model not found. Please run train_model.py first.")
else:
    pipeline = joblib.load(MODEL_FILE)
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "all_label_encoders.joblib"))

    # Load dataset just to get feature columns
    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = data.drop(columns=[TARGET_COL])

    # ================================
    # PREDICTION INTERFACE
    # ================================
    st.subheader("Fraud Detection App - Prediction")

    # Dropdowns for categorical variables
    states = ["west bengal","maharashtra","uttar pradesh","tamil nadu","karnataka",
              "delhi","kerala","gujarat","rajasthan","punjab","bihar","jharkhand","odisha","assam"]
    categories = ["groceries","utilities","transport","healthcare","education","insurance","fuel",
                  "mobile_recharge","restaurant","clothing","luxury_goods","electronics","crypto",
                  "high_value_transfer","foreign_remittance","jewellery","travel_packages","gambling"]
    devices = ["mobile_app","web_portal","pos_terminal","atm"]
    payments = ["upi","credit_card","debit_card","netbanking","wallet"]
    merchants = ["retail_store","online_marketplace","hospital","restaurant","airline","hotel"]

    # Input fields
    input_data = {}
    for col in X.columns:
        if col == "state":
            input_data[col] = st.selectbox("State", states)
        elif col == "category":
            input_data[col] = st.selectbox("Category", categories)
        elif col == "device_used":
            input_data[col] = st.selectbox("Device Used", devices)
        elif col == "payment_method":
            input_data[col] = st.selectbox("Payment Method", payments)
        elif col == "merchant_type":
            input_data[col] = st.selectbox("Merchant Type", merchants)
        else:
            input_data[col] = st.text_input(f"{col}", "")

    # Buttons
    col1, col2 = st.columns([1,1])
    with col1:
        predict_btn = st.button("Predict Fraud Risk")
    with col2:
        reset_btn = st.button("Reset")

    if predict_btn:
        try:
            input_df = pd.DataFrame([input_data])

            # Apply same label encoding
            for col in input_df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    input_df[col] = le.transform(input_df[col].astype(str))

            prediction = pipeline.predict(input_df)[0]
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected")
            else:
                st.success("✅ Legitimate Transaction")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    if reset_btn:
        st.experimental_rerun()
