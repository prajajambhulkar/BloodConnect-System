import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BloodConnect AI",
    page_icon="🩸",
    layout="wide"
)

# --- CUSTOM EMERGENCY RED THEME CSS ---
st.markdown("""
    <style>
        .main-header { color: #991B1B; font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem; }
        .sub-header { color: #4B5563; font-size: 1.1rem; text-align: center; margin-bottom: 2rem; }
        .section-title { color: #1E3A8A; font-size: 1.5rem; font-weight: 700; border-bottom: 2px solid #F3F4F6; padding-bottom: 0.5rem; margin-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD RE-TRAINED ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# --- TITLE HEADERS ---
st.markdown("<div class='main-header'>🩸 BloodConnect AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Smart Blood Donor & Emergency Assistance System</div>", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("Error: model.pkl or scaler.pkl file missing from the directory.")
    st.stop()

# --- USER VITALS FORM ---
st.markdown("<div class='section-title'>📋 Enter Donor Vitals Matrix</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input("Recency (Months since last donation)", min_value=0, max_value=100, value=2)
    frequency = st.number_input("Frequency (Total number of times donated)", min_value=1, max_value=200, value=5)

with col2:
    time = st.number_input("Time (Months since first donation)", min_value=1, max_value=200, value=12)

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("Predict Future Donation Probability", type="primary"):
    # Align structural columns exactly with training data shape: [Recency, Frequency, Time]
    input_features = np.array([[recency, frequency, time]])
    scaled_features = scaler.transform(input_features)
    
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    st.markdown("<div class='section-title'>📊 System Evaluation & Output</div>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    
    if prediction == 1:
        m1.metric(label="Predicted Behavior", value="Likely to Return", delta="High Predictability")
        m2.metric(label="Donor Retaining Confidence", value=f"{probability:.2%}")
        st.success("🎉 Patient Profile Match: This donor exhibits standard high-affinity recurrence patterns. Great candidate for emergency standby scheduling rosters!")
    else:
        m1.metric(label="Predicted Behavior", value="Unlikely to Return", delta="- Critical Interval Gap", delta_color="inverse")
        m2.metric(label="Donor Retaining Confidence", value=f"{probability:.2%}")
        st.warning("⚠️ Action Required: Deferral intervals or timing gaps suggest this user is unlikely to spontaneously return for the target cycle. Consider triggering targeted system outreach notifications.")
