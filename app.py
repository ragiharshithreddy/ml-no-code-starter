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
import time # For simulation of Google Login/loading

# --- 1. Custom CSS for Super UI/UX (Glassmorphism) ---
GLASSY_CSS = """
<style>
    /* Global Styles */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Title and Header Styling */
    h1 {
        color: #E6E6FA; /* Lavender for contrast */
        font-weight: 800;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.4);
    }
    h2, h3 {
        color: #DDA0DD; /* Plum */
        padding-bottom: 5px;
    }
    
    /* --- Glassmorphism Effect for Cards and Containers --- */
    .stCard, [data-testid="stExpander"], [data-testid="stStatusContainer"], 
    [data-testid="stVerticalBlock"] > div:has(button), 
    .glass-card { /* Custom class for specific elements */
        background: rgba(255, 255, 255, 0.15); /* Transparency */
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px); /* Frosted Glass Effect */
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.4); /* Subtle Border/Light Reflection */
        padding: 15px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    
    .stCard:hover, [data-testid="stExpander"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }

    /* Metric Styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.2); 
        border-radius: 12px;
        padding: 10px;
        border-left: 5px solid #FFD700; /* Gold indicator */
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #7B68EE; /* MediumSlateBlue */
        font-weight: bold;
    }
    
    /* Sidebar styling for contrast/layering */
    .st-emotion-cache-1cypcdb {
        background: rgba(255, 255, 255, 0.05); /* Very slight tint */
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Set a background image for full Glassmorphism effect */
    .main {
        background: url("https://images.unsplash.com/photo-1542831371-29b0f74f9d13?fit=crop&w=1200&h=800&q=80") center center / cover no-repeat fixed;
    }
    
    /* Navbar styles - high contrast on top layer */
    .navbar-container {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(0, 0, 0, 0.7); /* Dark semi-transparent background */
        padding: 10px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 0 0 10px 10px;
        margin-bottom: 20px;
    }

    .nav-button {
        color: #E6E6FA !important;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 8px;
        transition: background 0.2s, color 0.2s;
    }

    .nav-button:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #FFFFFF !important;
    }
</style>
"""

# --- 2. Streamlit Page Configuration & CSS Injection ---
st.set_page_config(
    page_title="No-Code ML Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(GLASSY_CSS, unsafe_allow_html=True)

# --- 3. Caching Functions and Utility ---

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Caches the DataFrame loading."""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data(show_spinner="Generating exquisite profile report...")
def generate_profile_report(df):
    """Caches the ydata-profiling report generation."""
    profile = ProfileReport(df, explorative=True, minimal=True,
                            correlations={"auto": {"calculate": False}})
    return profile.to_html()

def download_data_button(df, filename_suffix="processed"):
    """Generates a button to download a DataFrame."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {filename_suffix.capitalize()} Data",
        data=csv,
        file_name=f"{filename_suffix}.csv",
        mime="text/csv",
        key=f"download_{filename_suffix}"
    )

# --- 4. Session State Initialization ---
# This holds the main DataFrame and preprocessing steps
if 'current_df' not in st.session_state:
    st.session_state.update({
        'logged_in': False,
        'current_df': None,
        'original_df': None,
        'target_col': None,
        'feature_cols': [],
        'model': None,
        'scaler': None,
        'le_target': None,
        'generated_code': "",
        'page': 'home' # For Navbar state
    })

# --- 5. Simulated Google Login ---
def google_login_page():
    st.title("🔒 Login Required (Optional)")
    st.markdown("### Access the full features by logging in.")
    
    col_login_main, col_login_spacer = st.columns([1, 1])

    with col_login_main:
        if st.button("🔑 Sign in with Google (Simulated)", use_container_width=True, type="primary"):
            with st.spinner("Connecting to Google..."):
                time.sleep(1) # Simulate network delay
                st.session_state.logged_in = True
                st.session_state.page = 'upload'
                st.rerun()
        
        # Option to skip login
        if st.button("➡️ Continue as Guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.page = 'upload'
            st.rerun()

# --- 6. Navbar and Page Routing ---
if not st.session_state.logged_in and st.session_state.page == 'home':
    google_login_page()
    st.stop()

# --- Page Selection Logic ---
PAGES = {
    'upload': "📁 Upload & Preview",
    'preprocess': "🧹 Preprocessing Pipeline",
    'explore': "📊 Data & Viz",
    'train': "🧠 Model Training",
    'predict': "🔮 Live Prediction"
}

def navbar_component():
    """Renders the simulated fixed navbar using Streamlit columns."""
    st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
    cols = st.columns(len(PAGES))
    
    page_keys = list(PAGES.keys())
    
    for i, (key, title) in enumerate(PAGES.items()):
        is_current = (st.session_state.page == key)
        
        # Add visual checkmark/icon for completed steps (optional logic)
        icon = "✅" if key in ['upload', 'preprocess'] and st.session_state.current_df is not None else "➡️"
        
        # Create a clickable button/link using markdown/HTML
        button_style = f"background: {'rgba(255, 255, 255, 0.3)' if is_current else 'transparent'}; font-weight: {'bold' if is_current else 'normal'};"
        
        if cols[i].button(f"{icon} {title}", key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

navbar_component()
st.title("✨ No-Code ML Explorer")

# --- 7. Sidebar: File Upload (Always present) ---
with st.sidebar:
    st.title("📂 File Management")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load and store in session state
            new_df = load_data(uploaded_file)
            
            # Check if a new file was uploaded (to reset state)
            if st.session_state.original_df is None or not st.session_state.original_df.equals(new_df):
                st.session_state.original_df = new_df
                st.session_state.current_df = new_df.copy()
                st.session_state.page = 'preprocess'
                st.success("File uploaded and state initialized!")
            
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    
    if st.session_state.current_df is not None:
        st.markdown("---")
        st.subheader("Data Status")
        st.info(f"Rows: {st.session_state.current_df.shape[0]}, Cols: {st.session_state.current_df.shape[1]}")
        download_data_button(st.session_state.current_df, "current_processed")
    else:
        st.warning("Please upload a CSV file to begin.")

# --- Router Logic ---

if st.session_state.current_df is None:
    if st.session_state.page != 'upload':
        st.session_state.page = 'upload'
    st.info("👆 Use the sidebar to upload a CSV and start preprocessing.")
    st.stop()

df = st.session_state.current_df.copy() # Use a copy for manipulation within the page

# ====================================================================
# Page: Data Manipulation & Preprocessing Pipeline (Step-by-Step)
# ====================================================================
if st.session_state.page == 'preprocess':
    st.header("🧹 Step-by-Step Preprocessing Pipeline")
    
    st.info(f"Current DataFrame Shape: {df.shape}")

    # --- Step 1: Column Dropping ---
    with st.expander("1️⃣ Feature Manipulation: Drop Columns", expanded=True):
        cols_to_drop = st.multiselect("Select columns to permanently drop:",
            options=df.columns.tolist(),
            key="cols_to_drop_step1"
        )
        if st.button("Apply Drop Columns", type="primary"):
            if cols_to_drop:
                try:
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.session_state.current_df = df.copy()
                    st.success(f"Successfully dropped {len(cols_to_drop)} columns.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error dropping columns: {e}")
            else:
                st.warning("No columns selected to drop.")

    # --- Step 2: Missing Value Handling ---
    with st.expander("2️⃣ Missing Value Handling", expanded=True):
        
        # Calculate missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if missing_cols.empty:
            st.info("No missing values found! Move to the next step.")
        else:
            st.dataframe(missing_cols.rename("Missing Count"))
            
            missing_method = st.selectbox("Select Imputation Method", ["Drop Rows (NaN)", "Mean/Mode Imputation (Simple)"], key="missing_method")
            
            if st.button("Apply Missing Value Handling", key="apply_missing_step2"):
                try:
                    if missing_method == "Drop Rows (NaN)":
                        rows_before = len(df)
                        df.dropna(inplace=True)
                        rows_after = len(df)
                        st.success(f"Dropped {rows_before - rows_after} rows with missing data.")
                    
                    elif missing_method == "Mean/Mode Imputation (Simple)":
                        for col in missing_cols.index:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(df[col].mean(), inplace=True)
                            else:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                        st.success("Imputed missing values using Mean (Numeric) or Mode (Categorical).")
                    
                    st.session_state.current_df = df.copy()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during imputation: {e}")


    # --- Step 3: Train-Test Split & Download ---
    with st.expander("3️⃣ Final Step: Train-Test Split", expanded=True):
        st.subheader("Prepare Data for Modeling")

        target_options = df.columns.tolist()
        st.session_state.target_col = st.selectbox("Select Target Column (Y)", target_options, key="split_target")
        split_ratio = st.slider("Train Size Ratio", 0.5, 0.95, 0.8, key="split_ratio_final")
        
        if st.button("Perform & Download Split", key="apply_split_step3", type="primary"):
            try:
                X = df.drop(columns=[st.session_state.target_col])
                y = df[st.session_state.target_col]
                
                stratify_val = y if len(np.unique(y)) > 1 and y.dtype != object else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1 - split_ratio), random_state=42, stratify=stratify_val
                )

                st.success("Data Split Complete! Download your files below.")
                
                col_train, col_test = st.columns(2)
                with col_train:
                    train_df = pd.concat([X_train, y_train], axis=1)
                    download_data_button(train_df, "train_data")
                with col_test:
                    test_df = pd.concat([X_test, y_test], axis=1)
                    download_data_button(test_df, "test_data")
                
                st.session_state.feature_cols = X.columns.tolist() # Update features for the next tab
                st.session_state.page = 'explore' # Automatically move to next step
            except Exception as e:
                st.error(f"Error during split: {e}")

# ... (Previous code for explore, train, predict tabs - ensuring they use st.session_state.current_df) ...
# Note: The code for 'explore', 'train', and 'predict' tabs below needs to be placed sequentially after
# the 'preprocess' section, replacing the original code blocks for these tabs.
# The core logic for training/prediction must be robustly updated to use `st.session_state.current_df`
# and the features/target selected in the `preprocess` tab.

# ====================================================================
# Page: Data Exploration (Using current_df)
# ====================================================================
if st.session_state.page == 'explore':
    st.header("📊 Data Exploration & Visualization")
    # Use st.session_state.current_df for exploration
    explore_df = st.session_state.current_df.copy()

    # --- Data Preview and EDA ---
    with st.expander("📦 Current Data Preview", expanded=True):
        st.dataframe(explore_df.head(), use_container_width=True)

    with st.expander("✨ Automated EDA Report (Interactive HTML)", expanded=False):
        try:
            report_html = generate_profile_report(explore_df)
            html(report_html, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Could not generate EDA report: {e}")

    st.markdown("---")
    st.subheader("📈 Interactive Chart Builder")

    col1_chart, col2_chart = st.columns([1, 3])

    with col1_chart:
        chart_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"], key="chart_type_explore")
        x_axis = st.selectbox("X-axis", options=explore_df.columns, index=0, key="x_axis_explore")
        y_options = [c for c in explore_df.columns if c != x_axis]
        y_axis = st.selectbox("Y-axis", options=[None] + y_options, index=min(1, len(y_options)), key="y_axis_explore")
        color_cols = explore_df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_col = st.selectbox("Color Column (Optional)", options=[None] + color_cols, key="color_col_explore")
        chart_title = st.text_input("Chart Title", value=f"{chart_type} Chart", key="title_explore")

    with col2_chart:
        if st.button("▶️ Generate Interactive Chart", use_container_width=True, type="primary"):
            if chart_type in ["Scatter", "Line", "Bar", "Box", "Pie"] and y_axis is None:
                st.error(f"⚠️ Y-axis is required for a **{chart_type}** chart.")
            else:
                with st.spinner(f"Rendering {chart_type} chart..."):
                    try:
                        # Chart generation logic here... (same as previous robust version)
                        fig = None
                        if chart_type == "Scatter": fig = px.scatter(explore_df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                        elif chart_type == "Line": fig = px.line(explore_df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                        elif chart_type == "Bar": fig = px.bar(explore_df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                        elif chart_type == "Histogram": fig = px.histogram(explore_df, x=x_axis, color=color_col, title=chart_title)
                        elif chart_type == "Box": fig = px.box(explore_df, x=x_axis, y=y_axis, color=color_col, title=chart_title)
                        elif chart_type == "Pie": fig = px.pie(explore_df, names=x_axis, values=y_axis, color=color_col, title=chart_title)
                        
                        if fig:
                            fig.update_layout(hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)
                            try:
                                img_bytes = fig.to_image(format="png")
                                st.download_button(label="Download PNG Image", data=img_bytes, file_name=f"{chart_type}_chart.png", mime="image/png")
                            except ValueError:
                                st.warning("Image export failed. Install `kaleido` via `pip install kaleido` to enable PNG download.")
                    except Exception as e:
                        st.error(f"Chart generation failed. Check data types: {e}")

# ====================================================================
# Page: Model Training (Using current_df and selected features/target)
# ====================================================================
if st.session_state.page == 'train':
    st.header("🧠 Train Your Modular Model")
    
    # Check if target/features are set from preprocessing
    if not st.session_state.target_col or not st.session_state.feature_cols:
        st.warning("Please complete the 'Preprocessing Pipeline' tab first.")
        st.stop()
        
    target_col = st.session_state.target_col
    feature_cols = st.session_state.feature_cols
    
    # --- Modular Box 1: Configuration ---
    with st.expander("⚙️ Model Configuration", expanded=True):
        col_task, col_model = st.columns(2)
        with col_task:
            model_type = st.selectbox("1. Select Task Type", ["Classification", "Regression"], key="model_type_train")
            st.info(f"Target Column (Y): **{target_col}**")
        with col_model:
            if model_type == "Classification":
                model_choice = st.selectbox("2. Choose Algorithm", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"], key="model_choice_cls")
            else:
                model_choice = st.selectbox("2. Choose Algorithm", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"], key="model_choice_reg")
        
        st.info(f"Features (X) used: {len(feature_cols)} columns.")

    # --- Modular Box 2: Preprocessing (Encoding/Scaling) ---
    with st.expander("🧹 Remaining Preprocessing Options", expanded=True):
        col_pp1, col_pp2, col_pp3 = st.columns(3)
        with col_pp1: scale_data = st.checkbox("Scale numerical features (StandardScaler)", value=True, key="scale_train")
        with col_pp2: 
             # Skip missing handling if it was already applied in the previous step
             st.info("Missing values handled in Preprocess tab.")
        with col_pp3: split_ratio = st.slider("Train-Test Split Ratio", 0.5, 0.95, 0.8, key="split_ratio_train")

        encoding_method = st.selectbox("Categorical Feature Encoding", ["One-Hot Encoding", "Label Encoding", "Value Encoding (Ordinal)"])

    # --- Modular Box 3: Hyperparameters ---
    hyperparams = {}
    with st.expander("🛠️ Hyperparameter Tuning", expanded=False):
        # ... (Hyperparameter controls based on model_choice, same as previous robust version) ...
        if model_choice == "LogisticRegression":
            C = st.number_input("Regularization strength (C)", 0.01, 10.0, 1.0, step=0.01)
            max_iter = st.number_input("Max iterations", 50, 1000, 200, step=50)
            hyperparams = {"C": C, "max_iter": max_iter, "solver": 'liblinear'}
        elif model_choice == "RandomForestClassifier":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
        elif model_choice.startswith("XGBoost"):
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)
            n_estimators = st.slider("Number of estimators", 50, 500, 100)
            if "Classifier" in model_choice or model_choice == "DecisionTreeRegressor":
                max_depth = st.slider("Max depth", 1, 20, 6)
            
            if model_choice == "XGBoostClassifier":
                hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators, "max_depth": max_depth, "use_label_encoder": False, "eval_metric": 'logloss', "random_state": 42}
            elif model_choice == "XGBoostRegressor":
                hyperparams = {"learning_rate": learning_rate, "n_estimators": n_estimators, "random_state": 42}
        elif model_choice == "DecisionTreeRegressor":
            max_depth = st.slider("Max depth", 1, 50, 10)
            hyperparams = {"max_depth": max_depth, "random_state": 42}
        # ... (End Hyperparameter controls) ...
        

    # --- Training Button & Logic ---
    if st.button("🚀 Train Model and Evaluate", use_container_width=True, type="primary"):
        
        # --- 8. Robust Training Pipeline ---
        with st.status("Initializing Training Pipeline...", expanded=True) as status:
            
            X = st.session_state.current_df[feature_cols].copy()
            y = st.session_state.current_df[target_col].copy()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Code generation setup
            code_snippet = "### Generated Python Code\n\n"
            
            # Encoding
            try:
                if cat_cols:
                    status.write(f"Applying {encoding_method} encoding...")
                    if encoding_method == "One-Hot Encoding":
                        X = pd.get_dummies(X, drop_first=True)
                        code_snippet += "X = pd.get_dummies(X, drop_first=True)\n"
                    elif encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        for col in cat_cols:
                            X[col] = le.fit_transform(X[col].astype(str))
                        code_snippet += "le = LabelEncoder()\nfor col in X.select_dtypes(include=['object']): X[col] = le.fit_transform(X[col].astype(str))\n"
                    elif encoding_method == "Value Encoding (Ordinal)":
                        for col in cat_cols:
                            mapping = {val: i for i, val in enumerate(X[col].unique())}
                            X[col] = X[col].map(mapping).fillna(-1) 
                        code_snippet += "# Custom Ordinal Mapping applied\n"
            except Exception as e:
                status.error(f"Encoding failed: {e}")
                st.stop()

            # Target Encoding/Transformation
            le_target = None
            target_classes = None
            if y.dtype == 'object' and model_type == 'Classification':
                try:
                    status.write("Encoding target variable...")
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                    target_classes = le_target.classes_
                    code_snippet += "le_target = LabelEncoder(); y = le_target.fit_transform(y)\n"
                except Exception as e:
                    status.error(f"Target encoding failed: {e}")
                    st.stop()
            
            # Scaling
            scaler = None
            if scale_data:
                try:
                    status.write("Scaling features (StandardScaler)...")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns)
                    code_snippet += "scaler = StandardScaler(); X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)\n"
                except Exception as e:
                    status.error(f"Scaling failed: {e}")
                    st.stop()
            
            final_feature_cols = X.columns.tolist() 

            # Train-Test Split
            try:
                status.write("Splitting data...")
                stratify_val = y if model_type == 'Classification' and len(np.unique(y)) > 1 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1 - split_ratio), random_state=42, stratify=stratify_val
                )
                code_snippet += f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={1 - split_ratio:.2f}, random_state=42)\n\n"
            except Exception as e:
                status.error(f"Train-test split failed: {e}")
                st.stop()

            # Model Initialization and Training
            try:
                status.write(f"Initializing and training **{model_choice}**...")
                if model_choice == "LogisticRegression": model = LogisticRegression(**hyperparams)
                elif model_choice == "RandomForestClassifier": model = RandomForestClassifier(**hyperparams)
                elif model_choice == "XGBoostClassifier": model = XGBClassifier(**hyperparams)
                elif model_choice == "LinearRegression": model = LinearRegression()
                elif model_choice == "DecisionTreeRegressor": model = DecisionTreeRegressor(**hyperparams)
                elif model_choice == "XGBoostRegressor": model = XGBRegressor(**hyperparams)
                
                model.fit(X_train, y_train)
                code_snippet += f"model = {model_choice}({', '.join([f'{k}={repr(v)}' for k, v in hyperparams.items()])})\n"
                code_snippet += "model.fit(X_train, y_train)\n"
                
                status.update(label="✅ Training Completed Successfully! See Results Below.", state="complete", expanded=False)
            except Exception as e:
                status.error(f"Training failed: {e}. Check data types and hyperparameter values.")
                st.stop()
        
        # Store results in session state
        st.session_state.update({
            'model': model, 'scaler': scaler, 'le_target': le_target, 'target_classes': target_classes,
            'feature_cols_input': feature_cols, 'final_feature_cols': final_feature_cols,
            'model_type': model_type, 'encoding_method': encoding_method, 'cat_cols': cat_cols,
            'generated_code': code_snippet
        })

        # --- Evaluation (Same robust logic as before) ---
        st.markdown("---")
        st.subheader("💡 Evaluation Results")
        y_pred = model.predict(X_test)
        
        # Classification Metrics (Robust against 'micro avg' error)
        if model_type == "Classification":
            # ... (Classification metrics and plotting using the robust logic from previous answer) ...
            col_acc, col_f1, col_prec, col_rec = st.columns(4)
            acc = accuracy_score(y_test, y_pred)
            unique_classes_test = len(np.unique(y_test))
            
            if unique_classes_test < 2:
                with col_acc: st.metric("Accuracy", f"{acc*100:.2f}%")
                st.warning("Only one class in test set. Micro/Macro F1-scores cannot be calculated.")
                report = {}
            else:
                target_names = target_classes if target_classes is not None else [str(i) for i in np.unique(y_test)]
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=target_names)

                with col_acc: st.metric("Accuracy", f"{acc*100:.2f}%")
                micro_f1 = report.get('micro avg', report.get('weighted avg', {'f1-score': acc}))['f1-score']
                micro_precision = report.get('micro avg', report.get('weighted avg', {'precision': acc}))['precision']
                micro_recall = report.get('micro avg', report.get('weighted avg', {'recall': acc}))['recall']
                
                with col_f1: st.metric("Micro F1-Score", f"{micro_f1:.4f}")
                with col_prec: st.metric("Micro Precision", f"{micro_precision:.4f}")
                with col_rec: st.metric("Micro Recall", f"{micro_recall:.4f}")
                
                st.markdown("### 📉 Confusion Matrix")
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    labels = target_classes if target_classes is not None else np.unique(y_test)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax, xticklabels=labels, yticklabels=labels)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot confusion matrix: {e}")
                
                st.markdown("### 📋 Detailed Classification Report")
                report_df = pd.DataFrame(report).transpose()
                if 'accuracy' in report_df.index: report_df = report_df.drop('accuracy')
                st.dataframe(report_df, use_container_width=True)

        # Regression Metrics
        else:
            # ... (Regression metrics and plotting same as previous robust version) ...
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)

            col_mse, col_rmse, col_mae, col_r2, col_evs = st.columns(5)
            with col_mse: st.metric("MSE", f"{mse:.4f}")
            with col_rmse: st.metric("RMSE", f"{rmse:.4f}")
            with col_mae: st.metric("MAE", f"{mae:.4f}")
            with col_r2: st.metric("R² Score", f"{r2:.4f}")
            with col_evs: st.metric("Explained Variance", f"{evs:.4f}")

            st.markdown("### 📊 Actual vs Predicted Values")
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted Values", trendline="ols")
            fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
            st.plotly_chart(fig, use_container_width=True)


        # Feature Importance
        if model_choice in ["RandomForestClassifier", "DecisionTreeRegressor", "XGBoostClassifier", "XGBoostRegressor"]:
            st.markdown("---")
            st.subheader("🌲 Feature Importance")
            try:
                importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
                feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                fig_imp = px.bar(feature_importance_df.head(10), x='Importance', y='Feature', orientation='h', title='Top 10 Feature Importances', color_discrete_sequence=['#4B0082'])
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot feature importance: {e}")
        
    # --- Modular Box 4: Generated Code Viewer ---
    st.markdown("---")
    if st.session_state.get('generated_code'):
        if st.checkbox("💡 See Generated Python Code"):
            st.code(st.session_state.generated_code, language='python')


# ====================================================================
# Page: Prediction (Using trained model and robust prediction logic)
# ====================================================================
if st.session_state.page == 'predict':
    st.header("🔮 Live Prediction Interface")
    
    if st.session_state.model is None:
        st.warning("Please train a model in the 'Model Training' tab first.")
        st.stop()

    # Get stored objects
    model = st.session_state.model
    scaler = st.session_state.scaler
    le_target = st.session_state.le_target
    cat_cols = st.session_state.cat_cols
    feature_cols_input = st.session_state.feature_cols_input
    final_feature_cols = st.session_state.final_feature_cols
    encoding_method = st.session_state.encoding_method
    model_type = st.session_state.model_type
    
    # --- Modular Box 1: Prediction Form ---
    with st.expander("📝 Enter New Data for Prediction (Using Original Columns)", expanded=True):
        
        with st.form("prediction_form"):
            input_data = {}
            cols_per_row = 3
            cols_list = st.columns(cols_per_row)
            
            for i, col in enumerate(feature_cols_input):
                col_index = i % cols_per_row
                with cols_list[col_index]:
                    default_val = st.session_state.current_df[col].iloc[0] if col in st.session_state.current_df.columns else 0
                    
                    if pd.api.types.is_numeric_dtype(st.session_state.current_df[col]):
                        input_data[col] = st.number_input(f"🔢 {col}", value=float(default_val), step=1.0, key=f"pred_input_{col}")
                    else:
                        unique_vals = list(st.session_state.current_df[col].astype(str).unique())
                        if len(unique_vals) <= 15:
                            input_data[col] = st.selectbox(f"🏷️ {col}", options=unique_vals, index=unique_vals.index(str(default_val)) if str(default_val) in unique_vals else 0, key=f"pred_input_{col}")
                        else:
                            input_data[col] = st.text_input(f"📝 {col}", value=str(default_val), key=f"pred_input_{col}")
            
            st.markdown("---")
            submit_pred = st.form_submit_button("Submit Prediction", type="primary", use_container_width=True)

    # --- Modular Box 2: Results (Robust Prediction Logic) ---
    if submit_pred:
        with st.spinner("Analyzing input and calculating prediction..."):
            try:
                new_df = pd.DataFrame([input_data])
                
                # Apply Encoding
                if encoding_method == "One-Hot Encoding":
                    new_df = pd.get_dummies(new_df)
                    # Align columns with training data, filling missing (unseen categories) with 0
                    new_df = new_df.reindex(columns=final_feature_cols, fill_value=0)
                
                elif encoding_method == "Label Encoding":
                    for col in cat_cols:
                        le_temp = LabelEncoder()
                        # Fit on original data to ensure all possible classes are covered
                        le_temp.fit(st.session_state.current_df[col].astype(str).unique()) 
                        new_df[col] = le_temp.transform(new_df[col].astype(str))
                
                elif encoding_method == "Value Encoding (Ordinal)":
                    for col in cat_cols:
                        mapping = {val: i for i, val in enumerate(st.session_state.current_df[col].unique())}
                        # Use the training set mapping, fill unseen with -1
                        new_df[col] = new_df[col].map(mapping).fillna(-1) 
                        
                # Apply Scaling (if scaler exists)
                if scaler:
                    new_df = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)
                
                # Check for final feature alignment before prediction (Crucial Error Catch)
                if len(new_df.columns) != len(final_feature_cols) or not all(new_df.columns == final_feature_cols):
                    st.error("Feature misalignment detected after preprocessing. Please retrain the model.")
                    st.stop()
                    
                pred = model.predict(new_df)
                
                st.markdown("### ✅ Prediction Result")

                if model_type == "Classification":
                    label = le_target.inverse_transform(pred.astype(int)) if le_target else pred
                    st.success(f"🎯 Predicted Class: **{label[0]}**")
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(new_df)[0]
                        proba_df = pd.DataFrame({
                            'Class': st.session_state.target_classes if st.session_state.target_classes is not None else np.arange(len(proba)),
                            'Probability': proba
                        }).sort_values('Probability', ascending=False)
                        st.markdown("##### Confidence Scores")
                        st.dataframe(proba_df, hide_index=True)
                        
                else:
                    st.success(f"📈 Predicted Value: **{pred[0]:.4f}**")

            except Exception as e:
                st.error(f"Prediction failed due to model/data mismatch. Error: {e}")
