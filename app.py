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

# --- HEADER SECTION ---
st.markdown("<h1>Intelligent Credit Risk Scoring</h1>", unsafe_allow_html=True)
st.markdown("<p>Automated lending decision support system using traditional Machine Learning to evaluate borrower default probability.</p>", unsafe_allow_html=True)


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
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("### Borrower Profile")
        with st.form("risk_form", border=False):
            st.markdown("<span class='demographics-header'>DEMOGRAPHICS</span>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            age = f_col1.number_input("Age (Years)", 18, 100, 35)
            sex = f_col2.selectbox("Sex", ["male", "female"])
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<span class='demographics-header'>EMPLOYMENT & HOUSING</span>", unsafe_allow_html=True)
            f_col3, f_col4 = st.columns(2)
            
            # Map the raw indices [0,1,2,3] to polished, grammatically correct and distinct titles
            job_mapping = {
                0: 'Unemployed / Unskilled (Non-Resident)',
                1: 'Unskilled (Resident)',
                2: 'Skilled Employee / Official',
                3: 'Management / Highly Skilled / Self-Employed'
            }
            job = f_col3.selectbox("Job Category", [0, 1, 2, 3], format_func=lambda x: job_mapping[x])
            
            # Map raw strings to Title Case strings
            housing_mapping = {
                "own": "Owns Property",
                "free": "Lives For Free",
                "rent": "Renting"
            }
            housing = f_col4.selectbox("Housing Status", ["own", "free", "rent"], format_func=lambda x: housing_mapping[x])
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<span class='demographics-header'>FINANCIALS</span>", unsafe_allow_html=True)
            f_col5, f_col6 = st.columns(2)
            
            account_mapping = {
                "NA": "No Account",
                "little": "Little (< 200 DM)",
                "moderate": "Moderate (200 - 1000 DM)",
                "quite rich": "Rich (1000 - 2000 DM)",
                "rich": "Very Rich (> 2000 DM)"
            }
            saving_accounts = f_col5.selectbox("Savings Account", ["NA", "little", "moderate", "quite rich", "rich"], format_func=lambda x: account_mapping[x])
            
            check_mapping = {
                "NA": "No Checking Account",
                "little": "Negative / Little Balance",
                "moderate": "Moderate Balance",
                "rich": "High Balance"
            }
            checking_account = f_col6.selectbox("Checking Account", ["NA", "little", "moderate", "rich"], format_func=lambda x: check_mapping[x])
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<span class='demographics-header'>CREDIT REQUEST</span>", unsafe_allow_html=True)
            f_col7, f_col8 = st.columns(2)
            credit_amount = f_col7.number_input("Requested Amount (DM)", min_value=100, value=2500)
            duration = f_col8.slider("Term Duration (Months)", 4, 72, 24)
            
            # Use Capitalized/Professional names for purpose 
            purpose_raw = ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"]
            purpose_mapping = {
                "radio/TV": "Electronics (Radio/TV)",
                "education": "Education Funding",
                "furniture/equipment": "Furniture / Equipment",
                "car": "Vehicle Purchase",
                "business": "Business Venture",
                "domestic appliances": "Domestic Appliances",
                "repairs": "Home Repairs",
                "vacation/others": "Vacation / Other"
            }
            purpose = st.selectbox("Loan Purpose", purpose_raw, format_func=lambda x: purpose_mapping[x])
            
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
                
                with st.container(border=False):
                    if risk_score > 0.5:
                        st.error("### Application Flagged: HIGH RISK")
                        st.markdown(f"**Default Probability:** `{risk_score*100:.1f}%`")
                        st.markdown("The profile indicates a high likelihood of defaulting. Suggest requesting additional collateral or rejecting.")
                    else:
                        st.success("### Application Cleared: LOW RISK")
                        st.markdown(f"**Default Probability:** `{risk_score*100:.1f}%`")
                        st.markdown("The profile aligns with healthy borrower metrics. Pre-approval recommended.")
                    
                    st.markdown(f"<span style='color: #111827; font-size: 12px; font-weight: 500;'>Inferred via: {model_name} • Algorithm Pipeline Active</span>", unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    number={'suffix': "%", 'font': {'color': '#111827', 'size': 48, 'family': 'Inter'}},
                    title={'text': "Risk Magnitude", 'font': {'size': 20, 'color': '#6b7280', 'family': 'Inter'}},
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
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter'})
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🛡️</div>
                <h4>Awaiting Profile Data</h4>
                <p>Fill out the borrower profile on the left and click "Run Inference" to execute the prediction using traditional ML algorithms.</p>
            </div>
            """, unsafe_allow_html=True)
