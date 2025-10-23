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
import time # For simulation of Google Login/loading and training time

# --- New Imports for Added Features ---
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess  # For running Kaggle simulation
import json        # For Kaggle results
import os          # For file management in simulation
import sys         # To get python executable path

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
        label=f"⬇ Download {filename_suffix.capitalize()} Data",
        data=csv,
        file_name=f"{filename_suffix}.csv",
        mime="text/csv",
        key=f"download_{filename_suffix}"
    )

# --- NEW: SMTP Email Function ---
def send_email_notification(sender_email, app_password, recipient_email, results, model_name, task_type):
    """Sends an email with the training results."""
    # Check if credentials are provided
    if not sender_email or not app_password or not recipient_email:
        return False, "Skipping email: Missing SMTP credentials."
        
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = f"✅ ML Model Training Complete: {model_name}"
        message["From"] = sender_email
        message["To"] = recipient_email

        # Create the email body
        text_body = f"""
        Hello,
        
        Your {model_name} ({task_type}) model has finished training.
        
        Here are the results:
        {json.dumps(results, indent=2)}
        
        Regards,
        No-Code ML Explorer
        """
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color:#4B0082;">ML Training Complete</h2>
            <p>Your <b>{model_name} ({task_type})</b> model has finished training.</p>
            <h3 style="color:#DDA0DD;">Performance Metrics:</h3>
            <pre style="background-color:#f4f4f4; border:1px solid #ddd; padding:10px; border-radius:5px;">
{json.dumps(results, indent=2)}
            </pre>
        </body>
        </html>
        """
        
        message.attach(MIMEText(text_body, "plain"))
        message.attach(MIMEText(html_body, "html"))

        # Connect to Gmail's SMTP server
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {e}"


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
        'page': 'home', # For Navbar state
        'local_results': {},  # NEW: For comparison
        'kaggle_results': {} # NEW: For comparison
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
        if st.button("➡ Continue as Guest", use_container_width=True):
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
        icon = "✅" if key in ['upload', 'preprocess'] and st.session_state.current_df is not None else "➡"
        
        # Create a clickable button/link
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
                # Reset results on new file upload
                st.session_state.local_results = {}
                st.session_state.kaggle_results = {}
                st.session_state.model = None
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
    with st.expander("3️⃣ Final Step: Select Target (No Split Needed Here)", expanded=True):
        st.subheader("Prepare Data for Modeling")

        target_options = df.columns.tolist()
        # Use st.selectbox to just select the target, not split
        selected_target = st.selectbox("Select Target Column (Y)", target_options, key="split_target")
        
        if st.button("Confirm Target and Proceed", key="apply_split_step3", type="primary"):
            st.session_state.target_col = selected_target
            st.session_state.feature_cols = [col for col in df.columns if col != selected_target]
            
            st.success(f"Target set to '{selected_target}'. Features updated.")
            st.session_state.page = 'explore' # Automatically move to next step
            st.rerun()

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
        if st.button("▶ Generate Interactive Chart", use_container_width=True, type="primary"):
            if chart_type in ["Scatter", "Line", "Bar", "Box", "Pie"] and y_axis is None:
                st.error(f"⚠ Y-axis is required for a *{chart_type}* chart.")
            else:
                with st.spinner(f"Rendering {chart_type} chart..."):
                    try:
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
                                st.warning("Image export failed. Install kaleido via pip install kaleido to enable PNG download.")
                    except Exception as e:
                        st.error(f"Chart generation failed. Check data types: {e}")

# ====================================================================
# Page: Model Training (Using current_df and selected features/target)
# ====================================================================
if st.session_state.page == 'train':
    st.header("🧠 Train Your Modular Model")
    
    # Check if target/features are set from preprocessing
    if not st.session_state.target_col or not st.session_state.feature_cols:
        st.warning("Please complete the 'Preprocessing Pipeline' tab first to select a target.")
        st.stop()
        
    target_col = st.session_state.target_col
    feature_cols = st.session_state.feature_cols
    
    # --- Modular Box 1: Configuration ---
    with st.expander("⚙ Model Configuration", expanded=True):
        col_task, col_model = st.columns(2)
        with col_task:
            model_type = st.selectbox("1. Select Task Type", ["Classification", "Regression"], key="model_type_train")
            st.info(f"Target Column (Y): *{target_col}*")
        with col_model:
            if model_type == "Classification":
                model_choice = st.selectbox("2. Choose Algorithm", ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"], key="model_choice_cls")
            else:
                model_choice = st.selectbox("2. Choose Algorithm", ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"], key="model_choice_reg")
        
        st.info(f"Features (X) used: {len(feature_cols)} columns.")
    
    # --- Modular Box 2: Preprocessing (Encoding/Scaling) ---
    with st.expander("🧹 Remaining Preprocessing Options", expanded=True):
        col_pp1, col_pp2 = st.columns(2)
        with col_pp1: 
            scale_data = st.checkbox("Scale numerical features (StandardScaler)", value=True, key="scale_train")
            encoding_method = st.selectbox("Categorical Feature Encoding", ["One-Hot Encoding", "Label Encoding", "Value Encoding (Ordinal)"])
        with col_pp2: 
            split_ratio = st.slider("Train-Test Split Ratio", 0.5, 0.95, 0.8, key="split_ratio_train")
            st.info("Missing values handled in Preprocess tab.")

    # --- Modular Box 3: Hyperparameters ---
    hyperparams = {}
    with st.expander("🛠 Hyperparameter Tuning", expanded=False):
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
    
    # --- NEW: Modular Box 4: Execution Target ---
    with st.expander("🎯 Execution and Notifications", expanded=True):
        exec_target = st.radio(
            "Select Execution Target",
            ["Run Locally (Enables Prediction Tab)", "Run on Kaggle (Simulated)"],
            key="exec_target",
            help="Local runs train the model in this session for live predictions. Kaggle simulation tests the remote pipeline but disables the predict tab."
        )
        
        st.markdown("---")
        st.subheader("📧 Email Notification on Completion (Optional)")
        st.warning("For Gmail, you must use a 16-digit 'App Password'. Do not use your regular password.")
        
        col_em1, col_em2 = st.columns(2)
        
        # ***************************************************************
        # --- UNCOMMENTED SMTP CREDENTIAL INPUTS ARE BELOW ---
        # ***************************************************************
        with col_em1:
            sender_email = st.text_input("rhreddy4748@gmail.com", key="smtp_sender")
            recipient_email = st.text_input("Recipient Email", key="smtp_recipient")

        with col_em2:
            app_password = st.text_input("zujpoggswfcpxwjs", type="password", key="smtp_password")


    # --- Training Button & Logic ---
    if st.button("🚀 Train Model and Evaluate", use_container_width=True, type="primary"):
        
        # --- 8. Robust Training Pipeline ---
        with st.status("Initializing Training Pipeline...", expanded=True) as status:
            
            X = st.session_state.current_df[feature_cols].copy()
            y = st.session_state.current_df[target_col].copy()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Code generation setup
            code_snippet = "### Generated Python Code\n"
            code_imports = """
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

"""
            code_snippet += code_imports
            
            # Encoding
            try:
                if cat_cols:
                    status.write(f"Applying {encoding_method} encoding...")
                    if encoding_method == "One-Hot Encoding":
                        X = pd.get_dummies(X, drop_first=True)
                        code_snippet += "X = pd.get_dummies(X, drop_first=True)\n"
                    elif encoding_method == "Label Encoding":
                        le_map = {}
                        for col in cat_cols:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            le_map[col] = list(le.classes_)
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
                    code_snippet += f"le_target = LabelEncoder(); y = le_target.fit_transform(y)\n# Classes: {list(target_classes)}\n"
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
            
            # --- Model Training (Local vs. Kaggle Sim) ---
            
            # Store results here
            current_run_results = {"Execution Type": exec_target.split(" ")[0]}
            
            if exec_target == "Run Locally (Enables Prediction Tab)":
                try:
                    status.write(f"Initializing and training *{model_choice}* locally...")
                    if model_choice == "LogisticRegression": model = LogisticRegression(**hyperparams)
                    elif model_choice == "RandomForestClassifier": model = RandomForestClassifier(**hyperparams)
                    elif model_choice == "XGBoostClassifier": model = XGBClassifier(**hyperparams)
                    elif model_choice == "LinearRegression": model = LinearRegression()
                    elif model_choice == "DecisionTreeRegressor": model = DecisionTreeRegressor(**hyperparams)
                    elif model_choice == "XGBoostRegressor": model = XGBRegressor(**hyperparams)
                    
                    code_snippet += f"model = {model_choice}({', '.join([f'{k}={repr(v)}' for k, v in hyperparams.items()])})\n"
                    
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    code_snippet += "model.fit(X_train, y_train)\n"
                    
                    status.write(f"Local training complete in {training_time:.2f}s.")
                    
                    # Store results in session state
                    st.session_state.update({
                        'model': model, 'scaler': scaler, 'le_target': le_target, 'target_classes': target_classes,
                        'feature_cols_input': feature_cols, 'final_feature_cols': final_feature_cols,
                        'model_type': model_type, 'encoding_method': encoding_method, 'cat_cols': cat_cols,
                        'generated_code': code_snippet
                    })
                    
                    # Evaluate
                    status.write("Evaluating local model...")
                    y_pred = model.predict(X_test)
                    current_run_results["Training Time (s)"] = round(training_time, 4)

                except Exception as e:
                    status.error(f"Local training failed: {e}. Check data types and hyperparameter values.")
                    st.stop()
            
            elif exec_target == "Run on Kaggle (Simulated)":
                status.write("Disabling 'Predict' tab. Model object will not be stored.")
                st.session_state.model = None # Disable prediction tab
                
                try:
                    # 1. Package data
                    status.write("Packaging data for Kaggle simulation...")
                    train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
                    test_df = pd.concat([X_test, y_test.rename(target_col)], axis=1)
                    train_df.to_csv("sim_train_data.csv", index=False)
                    test_df.to_csv("sim_test_data.csv", index=False)

                    # 2. Create the execution script
                    status.write("Generating Kaggle execution script (kaggle_train_script.py)...")
                    kaggle_script_code = f"""
{code_imports}

def run_training():
    print("Kaggle Sim: Loading data...")
    try:
        train_df = pd.read_csv("sim_train_data.csv")
        test_df = pd.read_csv("sim_test_data.csv")
    except FileNotFoundError:
        print("Kaggle Sim Error: Data files not found.")
        with open("sim_results.json", 'w') as f:
            json.dump({{'error': 'Data files not found.'}}, f)
        return

    target_col = "{target_col}"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Re-align columns just in case (e.g., after one-hot)
    X_train = X_train.reindex(columns={final_feature_cols}, fill_value=0)
    X_test = X_test.reindex(columns={final_feature_cols}, fill_value=0)

    print("Kaggle Sim: Initializing {model_choice}...")
    model = {model_choice}({', '.join([f'{k}={repr(v)}' for k, v in hyperparams.items()])})
    
    print("Kaggle Sim: Starting training...")
    start_time = time.time()
    # Simulate faster Kaggle hardware
    time.sleep(np.random.uniform(0.5, 2.0)) # Fake compute time
    model.fit(X_train, y_train)
    training_time = (time.time() - start_time) * 0.5 # Simulate 2x faster hardware
    
    print(f"Kaggle Sim: Training complete in {{training_time:.2f}}s.")
    
    print("Kaggle Sim: Evaluating...")
    y_pred = model.predict(X_test)
    
    results = {{
        "Training Time (s)": round(training_time, 4),
        "Execution Type": "Kaggle"
    }}
    
    if "{model_type}" == "Classification":
        results["Accuracy"] = accuracy_score(y_test, y_pred)
        # Convert NumPy types to native Python types for JSON
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results["F1-Score (Micro)"] = report.get('micro avg', report.get('weighted avg', {{}})).get('f1-score', 0)
        
    else: # Regression
        results["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        results["R2 Score"] = r2_score(y_test, y_pred)
    
    print("Kaggle Sim: Saving results to sim_results.json...")
    with open("sim_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if _name_ == "_main_":
    run_training()
"""
                    with open("kaggle_train_script.py", "w") as f:
                        f.write(kaggle_script_code)

                    # 3. Run the simulation
                    status.write("Simulating remote execution on Kaggle (running script)...")
                    # Use sys.executable to ensure the correct python env is used
                    process = subprocess.run([sys.executable, "kaggle_train_script.py"], capture_output=True, text=True, timeout=120)
                    
                    if process.returncode != 0:
                        status.error(f"Kaggle simulation failed: {process.stderr}")
                        st.stop()
                    
                    status.write(f"Kaggle sim output: {process.stdout}")

                    # 4. Fetch results
                    status.write("Fetching results from Kaggle simulation...")
                    if os.path.exists("sim_results.json"):
                        with open("sim_results.json", 'r') as f:
                            kaggle_run_metrics = json.load(f)
                        
                        # Populate metrics for display
                        current_run_results.update(kaggle_run_metrics)
                        st.session_state.kaggle_results = current_run_results
                        
                        # Need y_pred for local display, so we re-predict (or parse from script)
                        # For this sim, we'll just show metrics, not plots
                        y_pred = None 
                    
                    else:
                        status.error("Simulation ran but results.json file was not found.")
                        st.stop()
                    
                    # Clean up simulation files
                    if os.path.exists("sim_train_data.csv"): os.remove("sim_train_data.csv")
                    if os.path.exists("sim_test_data.csv"): os.remove("sim_test_data.csv")
                    if os.path.exists("kaggle_train_script.py"): os.remove("kaggle_train_script.py")
                    if os.path.exists("sim_results.json"): os.remove("sim_results.json")

                except Exception as e:
                    status.error(f"Kaggle simulation failed: {e}")
                    st.stop()

            # --- Evaluation (If local run) ---
            st.markdown("---")
            st.subheader("💡 Evaluation Results")
            
            # If Kaggle run, just show the metrics from the JSON
            if exec_target == "Run on Kaggle (Simulated)":
                st.info("Displaying metrics from simulated Kaggle run.")
                if model_type == "Classification":
                    col_acc, col_f1, col_time = st.columns(3)
                    with col_acc: st.metric("Accuracy", f"{current_run_results.get('Accuracy', 0)*100:.2f}%")
                    with col_f1: st.metric("Micro F1-Score", f"{current_run_results.get('F1-Score (Micro)', 0):.4f}")
                    with col_time: st.metric("Training Time", f"{current_run_results.get('Training Time (s)', 0):.2f}s")
                else: # Regression
                    col_rmse, col_r2, col_time = st.columns(3)
                    with col_rmse: st.metric("RMSE", f"{current_run_results.get('RMSE', 0):.4f}")
                    with col_r2: st.metric("R² Score", f"{current_run_results.get('R2 Score', 0):.4f}")
                    with col_time: st.metric("Training Time", f"{current_run_results.get('Training Time (s)', 0):.2f}s")
                st.warning("Plots (e.g., Confusion Matrix) are not generated for simulated Kaggle runs.")

            # If Local run, show full evaluation and plots
            if exec_target == "Run Locally (Enables Prediction Tab)":
                if model_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    current_run_results["Accuracy"] = round(acc, 4)
                    
                    col_acc, col_f1, col_prec, col_rec = st.columns(4)
                    with col_acc: st.metric("Accuracy", f"{acc*100:.2f}%")
                    
                    unique_classes_test = len(np.unique(y_test))
                    if unique_classes_test < 2:
                        st.warning("Only one class in test set. Micro/Macro F1-scores cannot be calculated.")
                        report = {}
                    else:
                        target_names = target_classes if target_classes is not None else [str(i) for i in np.unique(y_test)]
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=target_names)
                        
                        micro_f1 = report.get('micro avg', report.get('weighted avg', {'f1-score': acc}))['f1-score']
                        micro_precision = report.get('micro avg', report.get('weighted avg', {'precision': acc}))['precision']
                        micro_recall = report.get('micro avg', report.get('weighted avg', {'recall': acc}))['recall']
                        
                        current_run_results["F1-Score (Micro)"] = round(micro_f1, 4)
                        
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
                
                else: # Regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    current_run_results["RMSE"] = round(rmse, 4)
                    current_run_results["R2 Score"] = round(r2, 4)

                    col_mse, col_rmse, col_mae, col_r2 = st.columns(4)
                    with col_mse: st.metric("MSE", f"{mse:.4f}")
                    with col_rmse: st.metric("RMSE", f"{rmse:.4f}")
                    with col_mae: st.metric("MAE", f"{mae:.4f}")
                    with col_r2: st.metric("R² Score", f"{r2:.4f}")

                    st.markdown("### 📊 Actual vs Predicted Values")
                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                    fig = px.scatter(results_df, x='Actual', y='Predicted', title="Actual vs Predicted Values", trendline="ols")
                    fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                    st.plotly_chart(fig, use_container_width=True)

                # Store local results
                st.session_state.local_results = current_run_results
                
                # Feature Importance (Local run only)
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
            
            # --- Send Email Notification ---
            if sender_email and app_password and recipient_email:
                status.write("Sending email notification...")
                success, message = send_email_notification(
                    sender_email, app_password, recipient_email,
                    current_run_results, model_choice, model_type
                )
                if success:
                    status.write(f"✅ {message}")
                else:
                    status.warning(f"⚠ {message}")
            else:
                status.write("Skipping email: Not all SMTP fields were provided.")
            
            # Final status update
            status.update(label="✅ Training Pipeline Completed!", state="complete", expanded=False)

    # --- NEW: Modular Box 5: Comparison Dashboard ---
    st.markdown("---")
    st.subheader("📊 Comparison Dashboard")
    
    results_data = {}
    if st.session_state.local_results:
        results_data["Local"] = st.session_state.local_results
    if st.session_state.kaggle_results:
        results_data["Kaggle"] = st.session_state.kaggle_results
        
    if not results_data:
        st.info("Run a model (either locally or on Kaggle) to see results here.")
    else:
        # Transpose for better readability (Metrics as rows, Runs as columns)
        comparison_df = pd.DataFrame(results_data).transpose()
        st.dataframe(comparison_df, use_container_width=True)


    # --- Modular Box 6: Generated Code Viewer ---
    st.markdown("---")
    if st.session_state.get('generated_code') and exec_target == "Run Locally (Enables Prediction Tab)":
        if st.checkbox("💡 See Generated Python Code (for Local Run)"):
            st.code(st.session_state.generated_code, language='python')


# ====================================================================
# Page: Prediction (Using trained model and robust prediction logic)
# ====================================================================
if st.session_state.page == 'predict':
    st.header("🔮 Live Prediction Interface")
    
    if st.session_state.model is None:
        st.warning("Please train a model in the 'Model Training' tab first.")
        st.info("Note: The 'Predict' tab is only enabled after a *local* model run.")
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
                        if len(unique_vals) <= 25: # Increased limit for usability
                            input_data[col] = st.selectbox(f"🏷 {col}", options=unique_vals, index=unique_vals.index(str(default_val)) if str(default_val) in unique_vals else 0, key=f"pred_input_{col}")
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
                        
                        # Handle unseen labels by adding them to classes
                        current_val = new_df[col].iloc[0]
                        if current_val not in le_temp.classes_:
                            le_temp.classes_ = np.append(le_temp.classes_, current_val)
                            
                        new_df[col] = le_temp.transform(new_df[col].astype(str))
                
                elif encoding_method == "Value Encoding (Ordinal)":
                    for col in cat_cols:
                        mapping = {val: i for i, val in enumerate(st.session_state.current_df[col].unique())}
                        # Use the training set mapping, fill unseen with -1
                        new_df[col] = new_df[col].map(mapping).fillna(-1) 
                
                # Apply Scaling (if scaler exists)
                if scaler:
                    # Get only the columns that were originally scaled
                    # This is tricky if one-hot encoding was used.
                    # Safter: try to transform using all final columns, errors will be caught
                    try:
                        new_df_scaled = scaler.transform(new_df[final_feature_cols])
                        new_df = pd.DataFrame(new_df_scaled, columns=final_feature_cols)
                    except ValueError:
                         st.error(f"Scaler failed. Column mismatch. Scaler expected {scaler.n_features_in_}, but got {len(new_df.columns)}.")
                         st.stop()

                
                # Check for final feature alignment before prediction (Crucial Error Catch)
                if not all(col in new_df.columns for col in final_feature_cols) or len(new_df.columns) < len(final_feature_cols):
                    st.error("Feature misalignment after preprocessing. Re-aligning...")
                    new_df = new_df.reindex(columns=final_feature_cols, fill_value=0)

                # Ensure order is correct
                new_df = new_df[final_feature_cols]

                pred = model.predict(new_df)
                
                st.markdown("### ✅ Prediction Result")

                if model_type == "Classification":
                    label = le_target.inverse_transform(pred.astype(int)) if le_target else pred
                    st.success(f"🎯 Predicted Class: *{label[0]}*")
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(new_df)[0]
                        proba_df = pd.DataFrame({
                            'Class': st.session_state.target_classes if st.session_state.target_classes is not None else np.arange(len(proba)),
                            'Probability': proba
                        }).sort_values('Probability', ascending=False)
                        st.markdown("##### Confidence Scores")
                        st.dataframe(proba_df, hide_index=True)
                        
                else:
                    st.success(f"📈 Predicted Value: *{pred[0]:.4f}*")

            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
                st.exception(e)
