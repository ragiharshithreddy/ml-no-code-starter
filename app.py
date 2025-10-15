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

st.set_page_config(page_title="No-Code ML Explorer", layout="wide")
st.title("🚀 No-Code ML Dataset Explorer")

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
    # Quick Chart
    # ===============================================
    st.subheader("📈 Quick Chart")
    cols = df.columns.tolist()
    if len(cols) >= 2:
        x_axis = st.selectbox("Select X-axis", options=cols)
        y_axis = st.selectbox("Select Y-axis", options=cols)
        if st.button("Plot Chart"):
            try:
                fig = px.scatter(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart generation failed: {e}")
    else:
        st.warning("Need at least two columns to generate a chart.")

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

    # Model selection
    if model_type == "Classification":
        model_choice = st.selectbox("Choose Model", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"])
    else:
        model_choice = st.selectbox("Choose Model", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"])

    # Hyperparameters
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
                X[col] = X_non_numeric[col].fillna(X_non_numeric[col].mode()[0])

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
        st.error(f"Train-test split failed: {e}")
        st.stop()

    if st.button("🚀 Train Model"):
        st.info(f"Training {model_choice}...")

        # Initialize
        try:
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
        except Exception as e:
            st.error(f"Model creation failed: {e}")
            st.stop()

        # Train
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        st.success("✅ Model training completed!")

        y_pred = model.predict(X_test)

        # ===============================================
        # Evaluation
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
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)

            st.markdown("### 📊 Regression Metrics")
            st.metric("Mean Squared Error", f"{mse:.4f}")
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            st.metric("Mean Absolute Error", f"{mae:.4f}")
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("Explained Variance", f"{evs:.4f}")

        # ===============================================
        # Prediction on new data
        # ===============================================
        st.markdown("---")
        st.subheader("🔮 Predict New Data")

        with st.form("prediction_form"):
            input_data = {}
            for col in feature_cols:
                default_val = df[col].iloc[0] if col in df.columns else 0
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_data[col] = st.number_input(f"{col}", value=float(default_val))
                else:
                    input_data[col] = st.text_input(f"{col}", value=str(default_val))
            submit_pred = st.form_submit_button("Predict")

        if submit_pred:
            try:
                new_df = pd.DataFrame([input_data])
                # apply same encoding/scaling
                if encoding_method == "One-Hot Encoding":
                    new_df = pd.get_dummies(new_df)
                    new_df = new_df.reindex(columns=X.columns, fill_value=0)
                elif encoding_method == "Label Encoding":
                    for col in new_df.select_dtypes(include=['object']).columns:
                        new_df[col] = le.fit_transform(new_df[col].astype(str))
                elif encoding_method == "Value Encoding (Ordinal)":
                    for col in new_df.select_dtypes(include=['object']).columns:
                        mapping = {val: i for i, val in enumerate(df[col].unique())}
                        new_df[col] = new_df[col].map(mapping).fillna(0)

                if scale_data:
                    new_df = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)

                pred = model.predict(new_df)

                if model_type == "Classification":
                    label = le_target.inverse_transform(pred) if 'le_target' in locals() else pred
                    st.success(f"🎯 Predicted Class: **{label[0]}**")
                    # reveal entity if exists
                    match = df[df[target_col] == label[0]]
                    if not match.empty:
                        st.markdown("### 🧾 Matching Entity in Dataset")
                        st.write(match)
                else:
                    st.success(f"📈 Predicted Value: **{pred[0]:.4f}**")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")
