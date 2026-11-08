import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - App", layout="wide")

st.markdown("# 📊 Machine Learning Model Deployment App")
st.markdown("BITS Pilani M.Tech AIML - Assignment 2")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Controls")

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        
        if test_df.shape[1] > 1:
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
        else:
            st.error("Uploaded CSV file must contain features and a target column.")
            st.stop()
            
        model_path = model_options[selected_model_name]
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0, average='weighted')
            rec = recall_score(y_test, y_pred, zero_division=0, average='weighted')
            f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
            
            # Display Metrics Dashboard
            st.subheader(f"Performance Metrics: {selected_model_name}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1-Score", f"{f1:.4f}")
            
            # Confusion Matrix Plot
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
        else:
            st.error(f"Model file not found at `{model_path}` inside GitHub.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload your `test_data.csv` file using the sidebar panel.")
