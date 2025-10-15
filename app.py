import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt

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
    # Quick Chart
    # ===============================================
    st.subheader("📈 Quick Chart")
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.warning("Need at least 2 columns to create a chart.")
    else:
        x_axis = st.selectbox("Select X-axis", options=cols)
        y_axis = st.selectbox("Select Y-axis", options=cols)
        if st.button("Plot Chart"):
            try:
                fig = px.scatter(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart generation failed: {e}")

    # ===============================================
    # Model Training
    # ===============================================
    st.subheader("🤖 Train Your Model")

    if len(df.columns) < 2:
        st.warning("Dataset needs at least one feature and one target column.")
        st.stop()

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

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle missing values
    if handle_missing:
        try:
            X_numeric = X.select_dtypes(include=['number'])
            X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())

            X_non_numeric = X.select_dtypes(exclude=['number'])
            for col in X_non_numeric.columns:
                if X_non_numeric[col].isnull().any():
                    X[col] = X_non_numeric[col].fillna(X_non_numeric[col].mode()[0])
        except Exception as e:
            st.warning(f"Error handling missing values: {e}")

    # Encoding
    encoding_method = st.selectbox(
        "Select Encoding Method for Categorical Features",
        ["One-Hot Encoding", "Label Encoding", "Value Encoding (Ordinal)"]
    )

    try:
        if encoding_method == "One-Hot Encoding":
            X = pd.get_dummies(X, drop_first=True)

        elif encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = le.fit_transform(X[col].astype(str))

        elif encoding_method == "Value Encoding (Ordinal)":
            for col in X.select_dtypes(include=['object', 'category']).columns:
                unique_vals = list(X[col].unique())
                mapping = {val: i for i, val in enumerate(unique_vals)}
                X[col] = X[col].map(mapping)
                st.write(f"🔢 Ordinal mapping for `{col}`:", mapping)
    except Exception as e:
        st.error(f"Encoding failed: {e}")
        st.stop()

    if y.dtype == 'object':
        try:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        except Exception as e:
            st.error(f"Target encoding failed: {e}")
            st.stop()

    # Scaling
    if scale_data:
        try:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        except Exception as e:
            st.warning(f"Scaling failed: {e}")

    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - split_ratio), random_state=42
        )
    except Exception as e:
        st.error(f"Train-test split failed: {e}")
        st.stop()

    # Train button
    if st.button("🚀 Train Model"):
        if X_train.empty or y_train.size == 0:
            st.error("Invalid training data. Check your selections.")
            st.stop()

        st.info(f"Training {model_choice}...")

        # Model initialization
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
            else:
                st.error("Model not implemented.")
                st.stop()
        except Exception as e:
            st.error(f"Model initialization failed: {e}")
            st.stop()

        # Training
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

        st.success("✅ Model training completed!")

        # ===============================================
        # Metrics and Visualization
        # ===============================================
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        if model_type == "Classification":
            try:
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
            except Exception as e:
                st.error(f"Error computing classification metrics: {e}")
        else:
            try:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
            except Exception as e:
                st.error(f"Error computing regression metrics: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")
