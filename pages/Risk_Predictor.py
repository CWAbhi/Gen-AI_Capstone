import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from src.feature_engineering import create_features

st.set_page_config(page_title="Risk Predictor", layout="wide")

st.markdown("# Real-time Risk Predictor")
st.markdown("Enter borrower details below to evaluate their credit risk score using our pre-trained machine learning models on the German Credit dataset.")

@st.cache_resource
def load_components():
    model = joblib.load('models/logistic_regression.pkl') if os.path.exists('models/logistic_regression.pkl') else None
    return model

model = load_components()

if model is None:
    st.error("Model not found. Please train a model via the **Model Training** page first.")
else:
    # Beautiful input form with columns based on German Credit Data
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(" Age", 18, 100, 35)
        sex = st.selectbox(" Sex", ["male", "female"])
        job = st.selectbox(" Job Category", [0, 1, 2, 3], format_func=lambda x: f"Category {x} (e.g. {['unskilled/resident', 'unskilled/non-resident', 'skilled', 'highly skilled'][x]})")
        housing = st.selectbox(" Housing", ["own", "free", "rent"])
        credit_amount = st.number_input(" Credit Amount (in DM)", min_value=0, value=2500)
        
    with col2:
        saving_accounts = st.selectbox(" Saving Accounts", ["NA", "little", "moderate", "quite rich", "rich"])
        checking_account = st.selectbox(" Checking Account", ["NA", "little", "moderate", "rich"])
        duration = st.slider(" Duration (in months)", 4, 72, 24)
        purpose = st.selectbox(" Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])
        
    # Predict button with animation
    if st.button(" Predict Credit Risk", use_container_width=True, type="primary"):
        with st.spinner("Analyzing borrower profile..."):
            
            # Combine data into DataFrame
            input_data = pd.DataFrame([{
                'Age': age,
                'Sex': sex,
                'Job': job,
                'Housing': housing,
                'Saving accounts': saving_accounts,
                'Checking account': checking_account,
                'Credit amount': credit_amount,
                'Duration': duration,
                'Purpose': purpose
            }])
            
            # Apply transformations
            features = create_features(input_data)
            
            # Simple imputation and predict logic since scaler/imputer files are removed
            if hasattr(model, "feature_names_in_"):
                expected_cols = model.feature_names_in_
                for col in expected_cols:
                    if col not in features.columns:
                        features[col] = 0
                features = features[expected_cols]
                features.fillna(features.median(), inplace=True)
            
            # Prediction logic
            risk_score = model.predict_proba(features)[0][1]
            prediction_label = "High Risk of Default" if risk_score > 0.5 else "Low Risk"
            
            st.markdown("---")
            col1_res, col2_res = st.columns([1, 2])
            
            with col1_res:
                if risk_score > 0.5:
                    st.error(f"### High Loan Risk\n\n**Loan Approval Recommendation:** ❌ **DENY**\n\n**Creditworthiness:** Poor\n\n**Default Probability:** {risk_score*100:.1f}%\n\nThe borrower has a high probability of defaulting on the loan.")
                else:
                    st.success(f"### Low Loan Risk\n\n**Loan Approval Recommendation:** ✅ **APPROVE**\n\n**Creditworthiness:** Good\n\n**Default Probability:** {risk_score*100:.1f}%\n\nThe borrower is considered healthy and likely to repay the loan.")
            
            with col2_res:
                # Display result with gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score * 100,
                    delta={'reference': 50, 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
                    number={'suffix': "%"},
                    title={'text': "Credit Risk Score", 'font': {'size': 24, 'color': '#1E3A8A'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1E3A8A"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#D1FAE5"},    # Green tint
                            {'range': [40, 70], 'color': "#FEF3C7"},   # Yellow tint
                            {'range': [70, 100], 'color': "#FEE2E2"}   # Red tint
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
