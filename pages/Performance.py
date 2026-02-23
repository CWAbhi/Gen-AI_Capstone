import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from src.evaluation import get_feature_importance
from utils.visualizations import plot_roc_curve, plot_confusion_matrix, plot_feature_importance

st.set_page_config(page_title="Model Performance", layout="wide")

st.markdown("# Model Performance Dashboard")
st.markdown("Dive deeper into Model Metrics, ROC Analysis, and Feature Importances.")

@st.cache_resource
def load_all_models():
    models = {}
    model_paths = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
    }
    
    for name, path in model_paths.items():
        full_path = f"models/{path}"
        if os.path.exists(full_path):
            models[name] = joblib.load(full_path)
    return models

models = load_all_models()

if not models:
    st.warning("No trained models discovered. Please execute training on the **Model Training** page first.")
else:
    selected_model_name = st.selectbox("Select Model to Evaluate", list(models.keys()))
    model = models[selected_model_name]
    
    tab1, tab2, tab3 = st.tabs([" Overview", " ROC Curve", " Feature Importance"])

    with tab1:
        st.subheader(f"{selected_model_name} Summary")
        # In a real app we would compute metrics dynamically here on an unseen test set or load cached results
        # Using placeholder dummy data for the dashboard demonstration as instructed
        metrics_df = pd.DataFrame([{
            'Accuracy': 0.942,
            'Precision': 0.81,
            'Recall': 0.76,
            'F1-Score': 0.78,
            'ROC-AUC': 0.89
        }]).T.rename(columns={0: 'Score'})
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.subheader("Confusion Matrix")
        # Dummy matrix
        cm = [[4500, 150], [200, 150]]
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
        
    with tab2:
        st.subheader("ROC Curve")
        # Dummy ROC data for visualization purposes
        fpr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
        tpr = [0.0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0]
        fig = plot_roc_curve(fpr, tpr, roc_auc=0.89)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Feature Importance")
        st.markdown("Understand which factors drive the model's decision.")
        # Get feature names from model if available
        features = None
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
            features = model.best_estimator_.feature_names_in_
            
        if features is not None:
            actual_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            
            df_imp = get_feature_importance(actual_model, features)
            if df_imp is not None:
                st.plotly_chart(plot_feature_importance(df_imp, top_n=10), use_container_width=True)
            else:
                st.info("Feature importance not supported by this model.")
        else:
            st.warning("Could not load feature names. Train the model first.")
