import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import json, os, uuid, time

st.set_page_config(page_title="No-Code ML Explorer", layout="wide")
st.title("🚀 No-Code ML Dataset Explorer")

# ===============================================
# File Upload
# ===============================================
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("👀 Data Preview")
    st.write(df.head())

    # ===============================================
    # Automated EDA
    # ===============================================
    st.subheader("📊 Automated EDA Report")
    with st.spinner("Generating profiling report..."):
        profile = ProfileReport(df, explorative=True)
        profile.to_file("report.html")
        with open("report.html", "r", encoding="utf-8") as f:
            html(f.read(), height=600, scrolling=True)

    # ===============================================
    # Quick Chart
    # ===============================================
    st.subheader("📈 Quick Chart")
    cols = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", options=cols)
    y_axis = st.selectbox("Select Y-axis", options=cols)
    if st.button("Plot Chart"):
        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)

    # ===============================================
    # Model Training
    # ===============================================
    st.subheader("🤖 Train Your Model")

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    model_type = st.selectbox("Select Task Type", ["Classification", "Regression"])

    # Select features
    feature_cols = st.multiselect(
        "Select Feature Columns (X)", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col]
    )

    # Model selection
    model_choice = None
    hyperparams = {}
    if model_type == "Classification":
        model_choice = st.selectbox("Choose Model", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"])

        if model_choice == "LogisticRegression":
            C = st.number_input("Regularization strength (C)", 0.01, 10.0, 1.0)
            max_iter = st.number_input("Max iterations", 50, 1000, 200)
            hyperparams = {"C": C, "max_iter": max_iter}

        elif model_choice == "RandomForestClassifier":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth}

        elif model_choice == "XGBoostClassifier":
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
            n_estimators = st.slider("Number of estimators", 50, 500, 100)
            max_depth = st.slider("Max depth", 1, 20, 6)
            hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth}

    else:
        model_choice = st.selectbox("Choose Model", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"])

        if model_choice == "DecisionTreeRegressor":
            max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"max_depth": max_depth}

        elif model_choice == "XGBoostRegressor":
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
            n_estimators = st.slider("Number of estimators", 50, 500, 100)
            hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators}

        else:
            st.info("Linear Regression has no major hyperparameters to tune.")

    # Preprocessing options
    st.markdown("### 🧹 Preprocessing Options")
    scale_data = st.checkbox("Scale numerical features", value=True)
    handle_missing = st.checkbox("Handle missing values automatically", value=True)
    split_ratio = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8)

    # Prepare data
    X = df[feature_cols]
    y = df[target_col]

    if handle_missing:
        X = X.fillna(X.mean())

    if scale_data:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)

    # Train button
    if st.button("🚀 Train Model"):
        st.info(f"Training {model_choice}...")

        if model_choice == "LogisticRegression":
            model = LogisticRegression(**hyperparams)
        elif model_choice == "RandomForestClassifier":
            model = RandomForestClassifier(**hyperparams)
        elif model_choice == "XGBoostClassifier":
            model = XGBClassifier(**hyperparams)
        elif model_choice == "LinearRegression":
            model = LinearRegression()
        elif model_choice == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(**hyperparams)
        elif model_choice == "XGBoostRegressor":
            model = XGBRegressor(**hyperparams)
        else:
            st.error("Model not implemented.")
            st.stop()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Model training completed!")

        # ===============================================
        # Metrics and Visualization
        # ===============================================
        if model_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc*100:.2f}%")

            st.markdown("### 📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.markdown("### 📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("Mean Squared Error", f"{mse:.4f}")
            st.metric("R² Score", f"{r2:.4f}")

else:
    st.info("👆 Please upload a CSV file to get started.")
