import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment 2 App", layout="wide")

st.title("📊 Machine Learning Model Deployment App")
st.write("BITS Pilani M.Tech AIML - Assignment 2")

# Sidebar configuration
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())

    if "Class" in test_df.columns:
        X_test = test_df.drop(columns=["Class"])
        y_true_raw = test_df["Class"]

    
      y_test = y_true_raw

        if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            scaler = joblib.load("model/scaler.pkl")
            X_test_proc = scaler.transform(X_test)
        else:
            X_test_proc = X_test

        model = joblib.load(model_files[model_choice])

        # Predictions without retraining
        y_pred = model.predict(X_test_proc)
        try:
            y_proba = model.predict_proba(X_test_proc)[:, 1]
        except:
            y_proba = np.zeros_like(y_pred, dtype=float)

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display Metrics
        st.subheader(f"Performance Metrics: {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    else:
        st.error("Uploaded CSV must contain the target column named 'Class'.")
else:
    st.info("Please upload your `test_data.csv` file using the sidebar panel.")
