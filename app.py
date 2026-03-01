import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from src.feature_engineering import create_features

st.set_page_config(page_title="Risk Predictor", layout="wide", initial_sidebar_state="collapsed")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
load_css()

# Hide sidebar completely via CSS as well
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# Inference Engine: Risk Predictor")
st.markdown("Evaluate instantaneous credit default probabilities using compiled traditional classifiers.")

@st.cache_resource
def load_components():
    # Prefer Logistic Regression but fallback to Decision Tree
    path = 'models/logistic_regression.pkl'
    if not os.path.exists(path):
        if os.path.exists('models/decision_tree.pkl'):
            path = 'models/decision_tree.pkl'
        elif os.path.exists('models/random_forest.pkl'):
            path = 'models/random_forest.pkl'
        else:
            return None
    return joblib.load(path), os.path.basename(path)

model_data = load_components()

if model_data is None:
    st.error("Model pipeline not found in 'models/'. Please compile a model first.", icon="🚫")
else:
    model, model_name = model_data
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Borrower Profile")
        with st.form("risk_form", border=True):
            st.markdown("<span style='color: #6b7280; font-size: 14px;'>DEMOGRAPHICS</span>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            age = f_col1.number_input("Age (Years)", 18, 100, 35)
            sex = f_col2.selectbox("Sex", ["male", "female"])
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            st.markdown("<span style='color: #6b7280; font-size: 14px;'>EMPLOYMENT & HOUSING</span>", unsafe_allow_html=True)
            f_col3, f_col4 = st.columns(2)
            job = f_col3.selectbox("Job Category", [0, 1, 2, 3], format_func=lambda x: ['Unskilled (Resident)', 'Unskilled (Non-resident)', 'Skilled', 'Highly Skilled'][x])
            housing = f_col4.selectbox("Housing Status", ["own", "free", "rent"])
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            st.markdown("<span style='color: #6b7280; font-size: 14px;'>FINANCIALS</span>", unsafe_allow_html=True)
            f_col5, f_col6 = st.columns(2)
            saving_accounts = f_col5.selectbox("Savings Account", ["NA", "little", "moderate", "quite rich", "rich"])
            checking_account = f_col6.selectbox("Checking Account", ["NA", "little", "moderate", "rich"])
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            st.markdown("<span style='color: #6b7280; font-size: 14px;'>CREDIT REQUEST</span>", unsafe_allow_html=True)
            f_col7, f_col8 = st.columns(2)
            credit_amount = f_col7.number_input("Requested Amount (DM)", min_value=100, value=2500)
            duration = f_col8.slider("Term Duration (Months)", 4, 72, 24)
            purpose = st.selectbox("Loan Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Run Inference", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Inference Results")
        if submitted:
            with st.spinner("Executing model pipeline..."):
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
                
                features = create_features(input_data)
                
                actual_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
                
                if hasattr(actual_model, "feature_names_in_"):
                    expected_cols = actual_model.feature_names_in_
                    for col in expected_cols:
                        if col not in features.columns:
                            features[col] = 0
                    features = features[expected_cols]
                    features.fillna(features.median(), inplace=True)
                
                risk_score = actual_model.predict_proba(features)[0][1]
                
                with st.container(border=True):
                    if risk_score > 0.5:
                        st.error("### ⚠️ Application Flagged: HIGH RISK")
                        st.markdown(f"**Default Probability:** `{risk_score*100:.1f}%`")
                        st.markdown("The profile indicates a high likelihood of defaulting. Suggest requesting additional collateral or rejecting.")
                    else:
                        st.success("### ✅ Application Cleared: LOW RISK")
                        st.markdown(f"**Default Probability:** `{risk_score*100:.1f}%`")
                        st.markdown("The profile aligns with healthy borrower metrics. Pre-approval recommended.")
                    
                    st.markdown(f"<span style='color: #9ca3af; font-size: 12px;'>Inferred via: {model_name}</span>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    number={'suffix': "%", 'font': {'color': '#111827'}},
                    title={'text': "Risk Magnitude", 'font': {'size': 20, 'color': '#6b7280'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#d1d5db"},
                        'bar': {'color': "#ef4444" if risk_score > 0.5 else "#10b981"},
                        'bgcolor': "#f3f4f6",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': "#d1fae5"},    
                            {'range': [40, 60], 'color': "#fef3c7"},   
                            {'range': [60, 100], 'color': "#fee2e2"}   
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="background-color: #f9fafb; padding: 40px; border-radius: 8px; border: 1px dashed #d1d5db; text-align: center; color: #6b7280; height: 100%;">
                <h4>Awaiting Inputs</h4>
                <p>Fill out the borrower profile on the left and click "Run Inference" to execute the prediction.</p>
            </div>
            """, unsafe_allow_html=True)
