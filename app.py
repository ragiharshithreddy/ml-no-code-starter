import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="No-Code ML Explorer", layout="wide")
st.title("🚀 No-Code ML Dataset Explorer")

# ===============================================
# File Upload
# ===============================================
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if df.empty:
        st.error("Uploaded file is empty. Please upload a valid CSV.")
        st.stop()

    st.subheader("👀 Data Preview")
    st.write(df.head())

    # ===============================================
    # Automated EDA
    # ===============================================
    st.subheader("📊 Automated EDA Report")
    try:
        with st.spinner("Generating profiling report..."):
            profile = ProfileReport(df, explorative=True, minimal=True,
                                    correlations={"auto": {"calculate": False}})
            report_html = profile.to_html()
            html(report_html, height=600, scrolling=True)
    except Exception as e:
        st.warning(f"Could not generate EDA report: {e}")

    # ===============================================
    # Flexible Charting
    # ===============================================
    st.subheader("📊 Flexible Charting")
    if len(df.columns) >= 1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"]
        )

        x_axis = st.selectbox("X-axis", options=df.columns, index=0)
        y_axis = st.selectbox("Y-axis", options=df.columns, index=min(1, len(df.columns)-1))

        chart_title = st.text_input("Chart Title", value=f"{chart_type} Chart")
        x_label = st.text_input("X-axis Label", value=x_axis)
        y_label = st.text_input("Y-axis Label", value=y_axis)

        color_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_col = st.selectbox("Color Column (Optional)", options=[None] + color_cols)

        palette_options = ["Plotly", "Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Pastel1", "Set1"]
        color_palette = st.selectbox("Color Palette", palette_options)

        if st.button("Generate Chart"):
            try:
                fig = None
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, color_continuous_scale=color_palette, title=chart_title)
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_col, color_continuous_scale=color_palette, title=chart_title)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, color_continuous_scale=color_palette, title=chart_title)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_col, color_discrete_sequence=px.colors.qualitative.Set1, title=chart_title)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_col, color_discrete_sequence=px.colors.qualitative.Set2, title=chart_title)
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_axis, values=y_axis, color=color_col, color_discrete_sequence=px.colors.qualitative.Set3, title=chart_title)

                st.plotly_chart(fig, use_container_width=True)

                # Download chart
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"{chart_type}_chart.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Chart generation failed: {e}")
    else:
        st.warning("Upload a CSV with at least one column to use charting.")

    # ===============================================
    # Model Training
    # ===============================================
    st.subheader("🤖 Train Your Model")

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    model_type = st.selectbox("Select Task Type", ["Classification", "Regression"])

    feature_cols = st.multiselect(
        "Select Feature Columns (X)",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    if target_col in feature_cols:
        st.warning("Target column should not be included in features.")
        feature_cols.remove(target_col)

    if model_type == "Classification":
        model_choice = st.selectbox("Choose Model", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"])
    else:
        model_choice = st.selectbox("Choose Model", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"])

    hyperparams = {}
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
    elif model_choice == "DecisionTreeRegressor":
        max_depth = st.slider("Max depth", 1, 50, 10)
        hyperparams = {"max_depth": max_depth}
    elif model_choice == "XGBoostRegressor":
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
        n_estimators = st.slider("Number of estimators", 50, 500, 100)
        hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators}

    st.markdown("### 🧹 Preprocessing Options")
    scale_data = st.checkbox("Scale numerical features", value=True)
    handle_missing = st.checkbox("Handle missing values automatically", value=True)
    split_ratio = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if handle_missing:
        X_numeric = X.select_dtypes(include=['number'])
        X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())
        X_non_numeric = X.select_dtypes(exclude=['number'])
        for col in X_non_numeric.columns:
            if X_non_numeric[col].isnull().any():
                X[col] = X[col].fillna(X_non_numeric[col].mode()[0])

    encoding_method = st.selectbox(
        "Select Encoding Method for Categorical Features",
        ["One-Hot Encoding", "Label Encoding", "Value Encoding (Ordinal)"]
    )

    if encoding_method == "One-Hot Encoding":
        X = pd.get_dummies(X, drop_first=True)
    elif encoding_method == "Label Encoding":
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
    elif encoding_method == "Value Encoding (Ordinal)":
        for col in X.select_dtypes(include=['object', 'category']).columns:
            mapping = {val: i for i, val in enumerate(X[col].unique())}
            X[col] = X[col].map(mapping)
            st.write(f"🔢 Ordinal mapping for `{col}`:", mapping)

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    if scale_data:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - split_ratio), random_state=42
        )
    except Exception as e:
       
