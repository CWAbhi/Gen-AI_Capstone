import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("# About the Project")
st.markdown("Capstone Project Milestone 1 - Intelligent Credit Risk Scoring System.")

st.markdown("---")

st.markdown("""
### 🎯 Objective
To evaluate the probability of a borrower defaulting on a credit line in the next two years.

### 📚 Tech Stack
- **Languages:** Python
- **ML Frameworks:** Scikit-Learn, XGBoost, Imbalanced-Learn
- **Data Manipulation:** Pandas, NumPy
- **Front-end UI:** Streamlit
- **Visualizations:** Plotly Express, Plotly Graph Objects

### 🛠️ Architecture
1. **Data Preprocessing:** Handled missing values via median and mode imputation. Scaled numerical features using `StandardScaler`. Managed class imbalance using `SMOTE`.
2. **Feature Engineering:** Derived new risk predictors such as total lateness and credit utilization bins.
3. **Model Training:** Utilizes advanced ensembles and grid search for hyperparameter tuning. Models included: Logistic Regression, Decision Tree, Random Forest, XGBoost.
4. **Evaluation:** Implemented robust classification metrics beyond pure accuracy (F1, Recall, ROC-AUC).
5. **UI Layer:** A modular Streamlit application with custom CSS to ensure a professional and beautiful layout.
""")

st.markdown("---")

st.info("Built using Streamlit. Designed as per the Capstone requirements.")
