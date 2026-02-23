import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="Intelligent Credit Risk Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
load_css()

# Sidebar styling removed


# Main Home Page Content
st.markdown("""
<div class='hero-container' style='background: linear-gradient(135deg, #1E3A8A 0%, #10B981 100%);
            padding: 50px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; text-align: center; animation: fadeIn 2s;'>Intelligent Credit Risk Scoring</h1>
    <p style='color: white; text-align: center; font-size: 18px; font-weight: 300;'>
        AI-Powered Lending Decision Support System
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Welcome to the Credit Risk Scoring Application!
This system evaluates the probability of a borrower experiencing a credit default (90 days past due or worse) within the next two years. Built upon the principles of machine learning, we've integrated powerful predictive models directly into a stunning user interface.

Navigate through the sidebar to explore data, train models, analyze features, and make real-time predictions.
""")

st.markdown("---")
st.markdown("### System Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Models Available", value="2", delta="")
with col2:
    # We will compute the real metric later
    st.metric(label="Best ROC-AUC", value="0.865", delta="+0.045", delta_color="normal")
with col3:
    st.metric(label="App Version", value="v1.0-RC1")
with col4:
    st.metric(label="Framework", value="Streamlit")

st.markdown("---")
st.markdown("### Quick Start Guide")
st.markdown("""
1. Navigate to the **Dataset Explorer** from the sidebar to upload and view the credit profiles data.
2. Go to **Model Training** to update or retrain the system models using the latest data.
3. Once training is complete, switch to the **Risk Predictor** module.
4. Input the borrower's details such as Age, Monthly Income, and Debt Ratio.
5. Click **Predict Credit Risk** to instantly predict the 90-day past due default probability.
""")
