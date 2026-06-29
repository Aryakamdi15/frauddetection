import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("💳 Fraud Detection in Financial Transactions")

st.write("""
This project detects suspicious financial transactions using the Isolation Forest Machine Learning algorithm.

Upload a transaction dataset and the model will identify possible fraud transactions.
""")

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("model.pkl")
except:
    st.error("model.pkl not found!")
    st.stop()

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        if "Class" in df.columns:
            st.metric("Fraud Cases", int(df["Class"].sum()))
        else:
            st.metric("Fraud Cases", "Unknown")

    st.divider()

    st.subheader("Class Distribution")

    if "Class" in df.columns:

        fig, ax = plt.subplots()

        df["Class"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Fraud vs Genuine Transactions")

        st.pyplot(fig)

    st.divider()

    st.subheader("Run Fraud Detection")

    detect = st.button("Detect Fraud")

    if detect:

        test_data = df.copy()

        if "Class" in test_data.columns:
            test_data = test_data.drop("Class", axis=1)

        predictions = model.predict(test_data)

        predictions = np.where(predictions == -1, 1, 0)

        result = df.copy()

        result["Prediction"] = predictions 
        fraud_count = int((predictions == 1).sum())
        genuine_count = int((predictions == 0).sum())

        st.success("Fraud detection completed successfully!")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Transactions", fraud_count)

        with col2:
            st.metric("Genuine Transactions", genuine_count)

        if fraud_count > 0:
            st.error(f"⚠️ Alert! {fraud_count} suspicious transaction(s) detected.")
        else:
            st.success("✅ No suspicious transactions found.")

        st.divider()

        st.subheader("Prediction Results")

        st.dataframe(result)

        st.divider()

        st.subheader("Prediction Distribution")

        fig2, ax2 = plt.subplots()

        pd.Series(predictions).value_counts().rename(
            {0: "Genuine", 1: "Fraud"}
        ).plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2
        )

        ax2.set_ylabel("")

        st.pyplot(fig2)

        st.divider()

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

else:

    st.info("Please upload a CSV file to begin.")