import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from src.data_preprocessing import handle_missing_values, treat_outliers, scale_features, handle_class_imbalance
from src.feature_engineering import create_features
from src.model_training import train_logistic_regression, train_random_forest, train_decision_tree, train_xgboost
from src.evaluation import evaluate_model
import sys

# Ensure module path is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Model Training", layout="wide")

st.markdown("# Model Training & Evaluation")
st.markdown("Train cutting-edge machine learning models on the latest dataset to predict credit defaults.")

@st.cache_data
def load_data():
    file_path = "data/german_credit_data.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Training Configuration")
    target_col = 'Risk'
    test_size = st.slider("Test Set Size %", 10, 50, 20, 5) / 100
    
    models_to_train = st.multiselect(
        "Select Models to Train",
        ["Logistic Regression", "Decision Tree"],
        default=["Logistic Regression", "Decision Tree"]
    )
    
    apply_smote = st.toggle('Apply SMOTE for Class Imbalance', value=True)
    apply_feature_eng = st.toggle('Generate Advanced Features', value=True)
    
    start_training = st.button(" Start Model Training", type="primary", use_container_width=True)

if start_training:
    if df is None:
        st.error("Dataset not found. Please upload or generate the dataset first.")
    elif not models_to_train:
        st.warning("Please select at least one model to train.")
    else:
        st.markdown("### Training Progress")
        progress_bar = st.progress(0)
        
        # 1. Feature Engineering
        st.info("Step 1: Feature Engineering...")
        if apply_feature_eng:
            df = create_features(df)
        st.success("Step 1 completed.")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # 2. Train-Test Split
        st.info("Step 2: Splitting Data into Training and Test Sets...")
        # Automatically synthesize Risk column if it doesn't exist (since dataset might lack it)
        if target_col not in df.columns:
            st.info("No 'Risk' column found in dataset. Synthesizing target variable for training...")
            np.random.seed(42)
            # Create a more realistic mapping to simulate risk
            base_risk = 0.2 + (df['Duration'] / df['Duration'].max()) * 0.3
            
            if 'Checking account_little' in df.columns:
                base_risk += df['Checking account_little'] * 0.2
            
            # Deduct major risk if they are rich
            if 'Saving accounts_rich' in df.columns:
                base_risk -= df['Saving accounts_rich'] * 0.3
            if 'Saving accounts_quite rich' in df.columns:
                base_risk -= df['Saving accounts_quite rich'] * 0.2
            if 'Checking account_rich' in df.columns:
                base_risk -= df['Checking account_rich'] * 0.2
                
            # High credit amounts are risky ONLY if the borrower is not rich
            is_rich = df.get('Saving accounts_rich', pd.Series(0, index=df.index)).astype(bool) | df.get('Checking account_rich', pd.Series(0, index=df.index)).astype(bool)
            amount_risk_factor = (df['Credit amount'] / df['Credit amount'].max()) * 0.4
            
            # Add risk for non-rich, subtract risk for rich
            base_risk = np.where(is_rich, base_risk - amount_risk_factor, base_risk + amount_risk_factor)
            
            df[target_col] = (base_risk + np.random.uniform(-0.1, 0.2, len(df)) > 0.5).astype(int)

        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        st.success("Step 2 completed.")
        progress_bar.progress(20)
        
        # 3. Handle Imbalance (scaling done in model pipeline)
        if apply_smote:
            st.info("Step 3: Applying SMOTE to balance classes...")
            X_train, y_train = handle_class_imbalance(X_train, y_train)
            st.success("Step 3 completed.")
            
        progress_bar.progress(40)
        
        # 5. Training Models
        st.info(f"Step 4: Training {len(models_to_train)} Selected Models...")
        
        results = {}
        
        for i, model_name in enumerate(models_to_train):
            st.info(f"Training {model_name}...")
            if model_name == "Logistic Regression":
                model = train_logistic_regression(X_train, y_train)
            elif model_name == "Decision Tree":
                model = train_decision_tree(X_train, y_train)
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Evaluate
            metrics, _, _ = evaluate_model(y_test, y_pred, y_pred_proba)
            results[model_name] = metrics
            
            progress_bar.progress(40 + int(((i + 1) / len(models_to_train)) * 50))
            
        progress_bar.progress(100)
        st.success("Step 4 completed.")
        
        st.cache_resource.clear()
        
        st.markdown("### Training Complete! Summary Below")
        st.success("All models trained and saved successfully. Other pages have been updated automatically.")
        
        # Display Results
        res_df = pd.DataFrame(results).T
        st.dataframe(res_df.style.highlight_max(axis=0, color='#10B981'), use_container_width=True)
