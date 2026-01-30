import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('loan_model_pipeline.pkl')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model assets: {e}")
        st.stop()

model = load_assets()

st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #4b1a7a 0%, #1a0b2e 100%);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            height: 100%;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* FIX: Adding space between Header and Success/Text elements */
        div[data-testid="stNotification"] {
            margin-top: 400px !important;
        }
        
        /* Typography */
        .main-title {
            font-size: 68px !important;
            font-weight: 800;
            color: white;
            text-align: center;
            margin-bottom: 50px;
            text-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        }

        .approved-card {
            background: linear-gradient(135deg, rgba(132, 250, 176, 0.2) 0%, rgba(143, 211, 244, 0.2) 100%);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            border: 1px solid #84fab0;
            margin-top: 20px;
        }

        .rejected-card {
            background: linear-gradient(135deg, rgba(250, 112, 154, 0.2) 0%, rgba(254, 225, 64, 0.2) 100%);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            border: 1px solid #fa709a;
            margin-top: 20px;
        }

        .result-text { font-size: 36px; font-weight: bold; color: white; }
        h3 { color: #d1b3ff !important; font-size: 28px !important;margin-top:100px, margin-bottom: 20px !important; }
    </style>
""", unsafe_allow_html=True)

# Enlarged Title
st.markdown('<p class="main-title">Loan Approval Predictor</p>', unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Applicant Details 📋")
person_age = st.sidebar.slider("Age", 18, 80, 30)
person_income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 50000)
person_emp_exp = st.sidebar.slider("Experience (Years)", 0, 50, 5)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 0, 500000, 10000)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 11.0)
cb_person_cred_hist_length = st.sidebar.slider("Credit History (Years)", 0, 40, 5)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)

person_gender = st.sidebar.selectbox("Gender", ["male", "female"])
person_education = st.sidebar.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME", "DEBTCONSOLIDATION"])
previous_loan_defaults_on_file = st.sidebar.selectbox("Previous Defaults?", ["No", "Yes"])

loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0

# --- MAIN UI LAYOUT ---
# Added gap="large" to give space between Summary and Model Status
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><center><h3>Summary 📊</h3></center>', unsafe_allow_html=True)
    st.write(f"**Income:** ${person_income:,}")
    st.write(f"**Loan:** ${loan_amnt:,} ({int(loan_percent_income*100)}% of income)")
    st.write(f"**Education:** {person_education}")
    st.write(f"**Credit Score:** {credit_score}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><center><h3>Model Status ⚙️</h3></center>', unsafe_allow_html=True)
    st.success("XGBoost Model Ready")
    st.write("**Algorithm:** Extreme Gradient Boosting")
    st.write("**Features:** 13 Input Features")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
correct_column_order = [
    'person_age', 'person_gender', 'person_education', 'person_income', 
    'person_emp_exp', 'person_home_ownership', 'loan_intent', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'previous_loan_defaults_on_file'
]

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Loan Approval 🎯", use_container_width=True):
    try:
        input_df = pd.DataFrame({
            'person_age': [person_age],
            'person_gender': [person_gender],
            'person_education': [person_education],
            'person_income': [person_income],
            'person_emp_exp': [person_emp_exp],
            'person_home_ownership': [person_home_ownership],
            'loan_intent': [loan_intent],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
            'credit_score': [credit_score],
            'previous_loan_defaults_on_file': [previous_loan_defaults_on_file] 
        })

        input_df = input_df[correct_column_order]
        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > 0.5 else 0    
        
        if prediction == 1:
            st.markdown(f"""<div class="approved-card"><div class="result-text">APPROVED ✅</div>
                        <h2 style="color:white;">Approval Confidence: {prob*100:.1f}%</h2></div>""", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""<div class="rejected-card"><div class="result-text">REJECTED ❌</div>
                        <h2 style="color:white;">Risk Score: {(1-prob)*100:.1f}%</h2></div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")