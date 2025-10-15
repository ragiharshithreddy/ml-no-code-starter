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

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="No-Code ML Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(uploaded_file):
    """Caches the DataFrame loading."""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data(show_spinner="Generating profiling report...")
def generate_profile_report(df):
    """Caches the ydata-profiling report generation."""
    profile = ProfileReport(df, explorative=True, minimal=True,
                            correlations={"auto": {"calculate": False}})
    return profile.to_html()

# --- Custom CSS for Enhanced UI/UX ---
st.markdown("""
<style>
    /* Main Streamlit container width */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Customize headers with a color theme */
    h1 {
        color: #4B0082; /* Indigo */
    }
    h2, h3 {
        color: #8A2BE2; /* Blue Violet */
    }
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #008080; /* Teal */
    }
    /* Sidebar styling */
    .st-emotion-cache-1cypcdb {
        background-color: #f0f2f6; /* Lighter sidebar background */
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar: File Upload and Info ---

with st.sidebar:
    st.title("📂 Data & Config")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid CSV.")
            st.stop()
        
        st.success("File uploaded successfully!")
        st.markdown("---")
        st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# --- Main Application Logic ---

st.title("🚀 No-Code ML Explorer")

if uploaded_file is None:
    st.info("👆 Please upload a CSV file in the sidebar to get started.")
    st.stop()

# Use tabs for clean navigation
tab_explore, tab_train, tab_predict = st.tabs(
    ["🔎 Data Exploration", "🧠 Model Training", "🔮 Prediction"]
)

# ====================================================================
# Tab 1: Data Exploration
# ====================================================================
with tab_explore:
    st.header("🔎 Data Exploration & Visualization")

    # --- Data Preview and Info ---
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- Automated EDA Report (Cached) ---
    with st.expander("📊 Automated EDA Report (Powered by ydata-profiling)", expanded=False):
        try:
            report_html = generate_profile_report(df)
            html(report_html, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Could not generate EDA report: {e}")

    st.markdown("---")

    # --- Flexible Charting ---
    st.subheader("📈 Flexible Charting")

    col1_chart, col2_chart = st.columns([1, 3])

    with col1_chart:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"]
        )

        # Columns for axes
        x_axis = st.selectbox("X-axis", options=df.columns, index=0)
        
        # y_axis is optional for Histogram
        y_options = [c for c in df.columns if c != x_axis]
        y_axis_default = y_options[0] if y_options else df.columns[0]
        y_axis = st.selectbox("Y-axis", options=[None] + y_options, index=y_options.index(y_axis_default) + 1 if y_axis_default in y_options else 0)
        
        # Color palette options
        color_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_col = st.selectbox("Color Column (Optional)", options=[None] + color_cols)

        palette_options = ["Plotly", "Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Pastel1", "Set1"]
        color_palette = st.selectbox("Color Palette", palette_options)
        
        chart_title = st.text_input("Chart Title", value=f"{chart_type} Chart: {x_axis} vs {y_axis if y_axis else 'Count'}")

    with col2_chart:
        if st.button("Generate Chart", use_container_width=True):
            if chart_type in ["Scatter", "Line", "Bar", "Box"] and y_axis is None:
                 st.error(f"Y-axis must be selected for a {chart_type} chart.")
            else:
                try:
                    fig = None
                    # Customizing colors for better separation of discrete/continuous
                    if chart_type == "Scatter":
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col,
                                         color_continuous_scale=color_palette if color_col and pd.api.types.is_numeric_dtype(df[color_col]) else None,
                                         color_discrete_sequence=px.colors.qualitative.Plotly if color_col and not pd.api.types.is_numeric_dtype(df[color_col]) else None,
                                         title=chart_title)
                    elif chart_type == "Line":
                        fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                    elif chart_type == "Bar":
                        fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, color=color_col, title=chart_title,
                                           color_discrete_sequence=px.colors.qualitative.Set1)
                    elif chart_type == "Box":
                        fig = px.box(df, x=x_axis, y=y_axis, color=color_col, title=chart_title,
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    elif chart_type == "Pie":
                        fig = px.pie(df, names=x_axis, values=y_axis, color=color_col, title=chart_title,
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                    
                    if fig:
                        # Update layout for cleaner look
                        fig.update_layout(
                            xaxis_title=x_axis,
                            yaxis_title=y_axis if y_axis else "Count",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Download chart button
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            label="Download Chart as PNG",
                            data=img_bytes,
                            file_name=f"{chart_type}_chart.png",
                            mime="image/png"
                        )

                except Exception as e:
                    st.error(f"Chart generation failed. Check your axis selections: {e}")
        else:
            st.info("Select options and click 'Generate Chart' to see the visualization.")


# ====================================================================
# Tab 2: Model Training
# ====================================================================
with tab_train:
    st.header("🧠 Train Your Model")

    # --- Configuration Section ---
    with st.expander("⚙️ Training Configuration", expanded=True):
        col_task, col_target, col_model = st.columns(3)
        with col_task:
            model_type = st.selectbox("Select Task Type", ["Classification", "Regression"])
        with col_target:
            target_col = st.selectbox("🎯 Select Target Column", df.columns)
        with col_model:
            if model_type == "Classification":
                model_choice = st.selectbox("Choose Model", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"])
            else:
                model_choice = st.selectbox("Choose Model", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"])

        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            [c for c in df.columns if c != target_col],
            default=[c for c in df.columns if c != target_col]
        )

        if not feature_cols:
            st.warning("Please select at least one feature column.")
            st.stop()

    # --- Hyperparameters ---
    with st.expander("🔧 Hyperparameters", expanded=False):
        hyperparams = {}
        if model_choice == "LogisticRegression":
            col_c, col_iter = st.columns(2)
            with col_c:
                C = st.number_input("Regularization strength (C)", 0.01, 10.0, 1.0, step=0.01)
            with col_iter:
                max_iter = st.number_input("Max iterations", 50, 1000, 200, step=50)
            hyperparams = {"C": C, "max_iter": max_iter, "solver": 'liblinear'} # use liblinear for small datasets
        
        elif model_choice == "RandomForestClassifier":
            col_n, col_d = st.columns(2)
            with col_n:
                n_estimators = st.slider("Number of trees", 10, 500, 100)
            with col_d:
                max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
        
        elif model_choice.startswith("XGBoost"):
            col_lr, col_ne, col_md = st.columns(3)
            with col_lr:
                learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)
            with col_ne:
                n_estimators = st.slider("Number of estimators", 50, 500, 100)
            if "Classifier" in model_choice or model_choice == "DecisionTreeRegressor":
                with col_md:
                    max_depth = st.slider("Max depth", 1, 20, 6)
            
            if model_choice == "XGBoostClassifier":
                hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth, "use_label_encoder": False, "eval_metric": 'logloss', "random_state": 42}
            elif model_choice == "XGBoostRegressor":
                hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators, "random_state": 42}

        elif model_choice == "DecisionTreeRegressor":
            max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"max_depth": max_depth, "random_state": 42}
    
    # --- Preprocessing ---
    with st.expander("🧹 Preprocessing Options", expanded=False):
        col_pp1, col_pp2, col_pp3 = st.columns(3)
        with col_pp1:
            scale_data = st.checkbox("Scale numerical features (StandardScaler)", value=True)
        with col_pp2:
            handle_missing = st.checkbox("Handle missing values (Mean/Mode Imputation)", value=True)
        with col_pp3:
            split_ratio = st.slider("Train-Test Split Ratio (Train size)", 0.5, 0.95, 0.8)

        encoding_method = st.selectbox(
            "Select Encoding Method for Categorical Features",
            ["One-Hot Encoding", "Label Encoding", "Value Encoding (Ordinal)"]
        )
    
    # --- Training Button & Logic ---
    if st.button("🚀 Train Model and Evaluate", use_container_width=True, type="primary"):
        
        # 1. Data Preparation
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store categorical column names for later prediction mapping
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle Missing Values
        if handle_missing:
            with st.spinner("Imputing missing values..."):
                X_numeric = X.select_dtypes(include=['number'])
                X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())
                X_non_numeric = X.select_dtypes(exclude=['number'])
                for col in X_non_numeric.columns:
                    if X_non_numeric[col].isnull().any():
                        X[col] = X_non_numeric[col].fillna(X_non_numeric[col].mode()[0])
            st.info("Missing values imputed.")

        # Encoding
        le = LabelEncoder()
        with st.spinner(f"Applying {encoding_method}..."):
            if encoding_method == "One-Hot Encoding":
                X = pd.get_dummies(X, drop_first=True)
                # Store columns for reindexing in prediction
                final_feature_cols = X.columns.tolist() 
            elif encoding_method == "Label Encoding":
                for col in cat_cols:
                    X[col] = le.fit_transform(X[col].astype(str))
                final_feature_cols = X.columns.tolist() 
            elif encoding_method == "Value Encoding (Ordinal)":
                for col in cat_cols:
                    mapping = {val: i for i, val in enumerate(X[col].unique())}
                    X[col] = X[col].map(mapping).fillna(-1) # Fill unmapped with -1
                final_feature_cols = X.columns.tolist() 
                
        
        # Target Encoding/Transformation
        if y.dtype == 'object' and model_type == 'Classification':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            # Storing classes for display
            target_classes = le_target.classes_
        elif y.dtype == 'object' and model_type == 'Regression':
             st.error("Cannot run Regression on a categorical target variable. Please choose Classification.")
             st.stop()
        else:
            target_classes = None

        # Scaling
        if scale_data:
            with st.spinner("Scaling numerical features..."):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)
            st.info("Features scaled.")

        # Train-Test Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1 - split_ratio), random_state=42, 
                stratify=y if model_type == 'Classification' and len(np.unique(y)) > 1 else None # Stratify for classification
            )
        except Exception as e:
            st.error(f"Train-test split failed. Check if your target column has sufficient data for stratification: {e}")
            st.stop()

        # 2. Model Initialization
        try:
            with st.spinner(f"Initializing {model_choice}..."):
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

        # 3. Training
        try:
            with st.spinner(f"Training {model_choice} on {len(X_train)} samples..."):
                model.fit(X_train, y_train)
            st.success("✅ Model training completed!")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        # 4. Evaluation
        y_pred = model.predict(X_test)
        
        st.markdown("---")
        st.subheader("💡 Evaluation Results")

        if model_type == "Classification":
            
            # --- FIX FOR KeyError: 'micro avg' ---
            
            # 1. Check for single class in test set
            unique_classes_test = len(np.unique(y_test))
            
            if unique_classes_test < 2:
                st.warning("⚠️ Warning: Only one class found in the test set. Micro/Macro F1-scores cannot be calculated. Displaying Accuracy only.")
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc*100:.2f}%")
                
                # Skip the remaining metric/report generation to prevent the error
                report = {} # Initialize empty report to avoid a crash later
            else:
                # 2. Proceed with full report generation
                col_acc, col_f1, col_prec, col_rec = st.columns(4)
                acc = accuracy_score(y_test, y_pred)
                
                # Check for zero_division='warn' and set explicit target_names for robustness
                target_names = target_classes if target_classes is not None else [str(i) for i in np.unique(y_test)]

                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=target_names)
                except Exception as e:
                    st.error(f"Failed to generate full classification report: {e}")
                    report = {} # Ensure report is still defined

                
                with col_acc: st.metric("Accuracy", f"{acc*100:.2f}%")
                
                # 3. Safely access micro/weighted average
                # Use 'weighted avg' as a fallback if 'micro avg' isn't available, or check if key exists
                
                if 'micro avg' in report:
                    micro_f1 = report['micro avg']['f1-score']
                    micro_precision = report['micro avg']['precision']
                    micro_recall = report['micro avg']['recall']
                elif 'weighted avg' in report:
                    # Fallback to weighted avg if micro is missing (e.g., in some binary cases)
                    st.warning("Could not find 'micro avg'. Using 'weighted avg' for general metrics.")
                    micro_f1 = report['weighted avg']['f1-score']
                    micro_precision = report['weighted avg']['precision']
                    micro_recall = report['weighted avg']['recall']
                else:
                    st.warning("Could not calculate F1/Precision/Recall averages.")
                    micro_f1, micro_precision, micro_recall = acc, acc, acc # Default to accuracy value

                with col_f1: st.metric("Micro F1-Score", f"{micro_f1:.4f}")
                with col_prec: st.metric("Micro Precision", f"{micro_precision:.4f}")
                with col_rec: st.metric("Micro Recall", f"{micro_recall:.4f}")

                # Confusion Matrix Visualization
                st.markdown("### 📉 Confusion Matrix")
                # ... (Confusion Matrix plotting logic remains the same) ...
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    
                    # Get class labels for display
                    labels = target_classes if target_classes is not None else np.unique(y_test)
                    
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                                xticklabels=labels, yticklabels=labels)
                    ax.set_ylabel("True Label")
                    ax.set_xlabel("Predicted Label")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot confusion matrix: {e}")

                st.markdown("### 📋 Classification Report")
                # Convert report to DataFrame and remove redundant/unnecessary keys for display
                report_df = pd.DataFrame(report).transpose()
                if 'accuracy' in report_df.index:
                    report_df = report_df.drop('accuracy')
                st.dataframe(report_df, use_container_width=True)

        else:
            # Regression Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)

            col_mse, col_rmse, col_mae, col_r2, col_evs = st.columns(5)
            with col_mse: st.metric("MSE", f"{mse:.4f}")
            with col_rmse: st.metric("RMSE", f"{rmse:.4f}")
            with col_mae: st.metric("MAE", f"{mae:.4f}")
            with col_r2: st.metric("R² Score", f"{r2:.4f}")
            with col_evs: st.metric("Explained Variance", f"{evs:.4f}")

            # Prediction vs True Plot
            st.markdown("### 📊 Actual vs Predicted Values")
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            fig = px.scatter(results_df, x='Actual', y='Predicted', 
                             title="Actual vs Predicted Values (Test Set)",
                             trendline="ols",  # Add a trendline
                             opacity=0.6)
            fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max()) # Ideal line
            st.plotly_chart(fig, use_container_width=True)


        # --- Feature Importance (for tree-based models) ---
        if model_choice in ["RandomForestClassifier", "DecisionTreeRegressor", "XGBoostClassifier", "XGBoostRegressor"]:
            st.markdown("---")
            st.subheader("🌲 Feature Importance")
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0]) # For linear models (LogisticRegression, LinearRegression)
                else:
                    st.warning("Feature importance is not available for this model type.")
                    st.stop()

                feature_importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                fig_imp = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h',
                                 title='Top 10 Feature Importances',
                                 color_discrete_sequence=['#4B0082'])
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not calculate/plot feature importance: {e}")
        
        # --- Store necessary variables in session state for Prediction tab ---
        st.session_state.model = model
        st.session_state.scaler = scaler if scale_data else None
        st.session_state.le_target = le_target if 'le_target' in locals() else None
        st.session_state.target_classes = target_classes
        st.session_state.feature_cols_input = feature_cols
        st.session_state.final_feature_cols = final_feature_cols
        st.session_state.model_type = model_type
        st.session_state.encoding_method = encoding_method
        st.session_state.cat_cols = cat_cols # Original categorical cols

        st.info("Model and preprocessing objects stored for live prediction in the 'Prediction' tab.")


# ====================================================================
# Tab 3: Prediction
# ====================================================================
with tab_predict:
    st.header("🔮 Predict New Data")
    
    # Check if a model has been trained
    if 'model' not in st.session_state:
        st.warning("Please train a model in the 'Model Training' tab first.")
        st.stop()

    model = st.session_state.model
    scaler = st.session_state.scaler
    le_target = st.session_state.le_target
    target_classes = st.session_state.target_classes
    feature_cols_input = st.session_state.feature_cols_input
    final_feature_cols = st.session_state.final_feature_cols
    model_type = st.session_state.model_type
    encoding_method = st.session_state.encoding_method
    cat_cols = st.session_state.cat_cols


    st.markdown("Enter values for the features below to get a live prediction.")

    with st.form("prediction_form"):
        input_data = {}
        cols_per_row = 3
        cols_list = st.columns(cols_per_row)
        
        for i, col in enumerate(feature_cols_input):
            col_index = i % cols_per_row
            with cols_list[col_index]:
                default_val = df[col].iloc[0] if col in df.columns else 0
                
                # Use st.text_input for categorical and st.number_input for numerical
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_data[col] = st.number_input(f"🔢 {col}", value=float(default_val), step=1.0)
                else:
                    unique_vals = list(df[col].unique())
                    if len(unique_vals) <= 15:
                        # Use selectbox for small number of categories
                        input_data[col] = st.selectbox(f"🏷️ {col}", options=unique_vals, index=unique_vals.index(str(default_val)) if str(default_val) in unique_vals else 0)
                    else:
                        # Use text input for many categories
                        input_data[col] = st.text_input(f"📝 {col}", value=str(default_val))
        
        st.markdown("---")
        submit_pred = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submit_pred:
        with st.spinner("Processing input and predicting..."):
            try:
                new_df = pd.DataFrame([input_data])
                
                # Apply same preprocessing pipeline
                
                # 1. Encoding
                if encoding_method == "One-Hot Encoding":
                    # One-hot encode the new data
                    new_df = pd.get_dummies(new_df)
                    # Reindex to match training columns, filling missing (new unseen categories) with 0
                    new_df = new_df.reindex(columns=final_feature_cols, fill_value=0)
                elif encoding_method == "Label Encoding":
                    # Re-fit/transform is NOT correct here, only transform. But LabelEncoder doesn't support unseen data easily.
                    # For simplicity in this no-code app, we will use a dictionary mapping from the full dataset.
                    for col in cat_cols:
                        le_temp = LabelEncoder()
                        le_temp.fit(df[col].astype(str).unique()) # Fit on ALL original data
                        new_df[col] = le_temp.transform(new_df[col].astype(str))
                elif encoding_method == "Value Encoding (Ordinal)":
                    for col in cat_cols:
                        mapping = {val: i for i, val in enumerate(df[col].unique())}
                        # Use the training set mapping, fill unseen with -1 (or 0)
                        new_df[col] = new_df[col].map(mapping).fillna(-1) 
                
                # 2. Scaling
                if scaler:
                    new_df = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)
                
                # 3. Prediction
                pred = model.predict(new_df)
                
                st.markdown("### ✅ Prediction Result")

                if model_type == "Classification":
                    if le_target:
                        label = le_target.inverse_transform(pred.astype(int))
                    elif target_classes is not None:
                        label = [target_classes[p] for p in pred.astype(int)]
                    else:
                        label = pred
                        
                    st.success(f"🎯 Predicted Class: **{label[0]}**")
                    
                    # Optional: Display probability (if available)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(new_df)[0]
                        proba_df = pd.DataFrame({
                            'Class': target_classes if target_classes is not None else np.arange(len(proba)),
                            'Probability': proba
                        }).sort_values('Probability', ascending=False)
                        st.markdown("##### Class Probabilities")
                        st.dataframe(proba_df, hide_index=True)
                        
                else:
                    st.success(f"📈 Predicted Value: **{pred[0]:.4f}**")

            except Exception as e:
                st.error(f"Prediction failed. Check that your input values are correct for the trained model: {e}")

# --- Footer for Version and Info ---
st.markdown("---")
col_footer1, col_footer2 = st.columns([4, 1])

with col_footer1:
    st.markdown("###### No-Code ML Explorer v1.0 | Built with Streamlit, scikit-learn & Plotly")

with col_footer2:
    with st.expander("Help/About", expanded=False):
        st.markdown("""
        **Data Exploration:** Uses `ydata-profiling` for automatic EDA.
        **Model Training:** Supports Classification (Logistic, RF, XGB) and Regression (Linear, DT, XGB).
        **Data Handling:** Automatic imputation (mean/mode) and scaling (StandardScaler) are optional.
        """)
