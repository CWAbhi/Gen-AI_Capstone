import streamlit as st
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.markdown("# Dataset Explorer")
st.markdown("Gain insights into the historical credit data before training our predictive models.")

if 'show_explorer' not in st.session_state:
    st.session_state.show_explorer = False

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:
    if st.button("Show Data Explorer"):
        st.session_state.show_explorer = True

if uploaded_file is not None and st.session_state.show_explorer:
    df = pd.read_csv(uploaded_file)
    target_col = 'SeriousDlqin2yrs' if 'SeriousDlqin2yrs' in df.columns else df.columns[-1]
    
    st.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns of credit data.")
    
    tab1, tab2 = st.tabs(["Raw Data & Stats", "Distributions"])
    
    with tab1:
        st.subheader("Sneak Peek at Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Data Quality Report")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        missing_df['% Missing'] = (missing_df['Missing Values'] / len(df)) * 100
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
        
    with tab2:
        st.subheader("Target Variable Distribution")
        fig = px.pie(df, names=target_col, title=f'{target_col} Distribution', 
                     color_discrete_sequence=['#10B981', '#1E3A8A'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Distributions")
        selected_col = st.selectbox("Select a feature to visualize", df.columns.drop(target_col))
        fig2 = px.histogram(df, x=selected_col, color=target_col, barmode='overlay',
                            title=f"Distribution of {selected_col} by {target_col}")
        st.plotly_chart(fig2, use_container_width=True)
