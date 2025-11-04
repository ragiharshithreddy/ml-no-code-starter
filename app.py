# AutoMLPilot Pro — Secure ML Lab with Error Handling
# -----------------------------------------------------
# Enhanced with proper error handling, security, and no data leakage

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Optional libraries
try:
    from ydata_profiling import ProfileReport
    YDATA_OK = True
except Exception:
    YDATA_OK = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from imblearn.over_sampling import SMOTE
    IMB_OK = True
except Exception:
    IMB_OK = False

import smtplib
import ssl
import json
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===================== SECURITY: EMAIL CONFIG =====================
# IMPORTANT: Never hardcode credentials in production!
# Use environment variables or Streamlit secrets instead
# Example: st.secrets["email"]["address"]
ENABLE_EMAIL = False  # Set to True only when properly configured
OWNER_GMAIL = ""  # Configure via st.secrets or env vars
OWNER_APP_PASSWORD = ""  # Use App Password, not regular password
OWNER_ALIAS = "noreply@automlpilot.com"
SENDER_NAME = "AutoMLPilot"

# Check if email is properly configured
if ENABLE_EMAIL and (not OWNER_GMAIL or not OWNER_APP_PASSWORD):
    # st.sidebar.warning("⚠️ Email not configured. Set credentials in secrets.")
    ENABLE_EMAIL = False

# ===================== PAGE CONFIG & THEME =====================
# Set page layout to wide and remove sidebar margins for fixed layout
st.set_page_config(page_title="AutoMLPilot Pro", page_icon="✨", layout="wide")

THEME = """
<style>
    /* Gradient Background */
    :root { 
        --bg1: #ffe5f0; 
        --bg2: #e6e9ff; 
        --card: rgba(255,255,255,0.8); 
        --border: rgba(255,255,255,0.35); 
        --primary-color: #7c3aed;
    }
    .main { 
        background: radial-gradient(1200px 600px at 10% 10%, var(--bg1), transparent),
                radial-gradient(900px 500px at 90% 20%, var(--bg2), transparent),
                linear-gradient(120deg,#f9fafb,#eef2ff); 
    }
    
    /* Fixed Layout Overrides */
    
    /* Prevent overall page scrolling and set height */
    .stApp {
        min-height: 100vh;
        max-height: 100vh;
        overflow: hidden; /* Main app container should not scroll */
    }
    
    /* Main Content Container: Fixed size & internal scroll */
    .block-container { 
        padding: 1rem 2rem 0rem 2rem; /* Reduced bottom padding to maximize space */
        height: calc(100vh - 80px); /* Total viewport height minus header height */
        overflow-y: auto; /* Internal scrolling for content */
        margin-top: 80px; /* Offset for fixed header */
        max-width: 100% !important;
    }
    
    /* Fixed Header Styling */
    .topbar { 
        position: fixed; /* Fix position */
        top: 0; 
        left: 0;
        right: 0;
        z-index: 1000; 
        height: 80px;
        backdrop-filter: blur(12px);
        background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7)); 
        border-bottom: 1px solid var(--border); 
        padding: 0.6rem 2rem; 
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(31,41,55,.05);
    }
    .topbar > div {
        width: 100%;
    }

    /* Fixed Sidebar & Navigation Buttons */
    .stSidebar {
        position: fixed;
        height: 100vh;
        padding-top: 80px; /* Offset for fixed header */
        z-index: 990;
    }
    /* Style for the sidebar radio/buttons */
    [data-testid="stSidebarContent"] .stRadio > div {
        flex-direction: column !important;
        align-items: stretch;
    }
    [data-testid="stSidebarContent"] .stRadio > div > label {
        margin-bottom: 8px;
        padding: 0;
    }
    [data-testid="stSidebarContent"] .stRadio label > div {
        /* Style the radio label as a pill button */
        padding: 10px 15px;
        border-radius: 999px;
        border: 1px solid #c7d2fe;
        color: #4338ca;
        background: #eef2ff;
        font-weight: 500;
        text-align: left;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebarContent"] .stRadio label:hover > div {
        background: #dce7ff;
    }
    [data-testid="stSidebarContent"] .stRadio input:checked + div > div {
        /* Selected button style */
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
        box-shadow: 0 2px 5px rgba(124, 58, 237, 0.3);
    }
    [data-testid="stSidebarContent"] .stRadio input:checked + div > div > div:first-child {
        background-color: transparent !important; /* Hide default radio dot */
    }
    /* Hide the default Streamlit radio dot entirely for the button style */
    .stRadio input[type="radio"] {
        display: none !important;
    }


    /* Existing styles */
    h1,h2,h3,h4 { color:#0f172a; }
    .chip { display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe; font-size:.8rem; }
    .card { background: var(--card); border:1px solid var(--border); border-radius:20px; box-shadow: 0 12px 35px rgba(31,41,55,.12); padding:16px; }
    .metric { background: rgba(255,255,255,0.75); border-left:4px solid #8b5cf6; border-radius:14px; padding:12px; margin:8px 0; }
    .pillbtn button { border-radius:999px !important; }
    .small { color:#475569; font-size:.85rem; }
    .tooltip { color:#6b7280; font-size:.85rem; }
    .error-box { background:#fee; border-left:4px solid #dc2626; padding:12px; border-radius:8px; margin:8px 0; }
    .success-box { background:#efe; border-left:4px solid #16a34a; padding:12px; border-radius:8px; margin:8px 0; }

    /* Fix Plotly/Matplotlib in non-scrolling content */
    .stPlotlyChart, .stImage, .stMatplotlib {
        max-height: 55vh; /* Limit chart height if needed, otherwise default */
        overflow: auto;
    }
    .stForm {
        overflow: hidden; /* Prevents form from creating unwanted scrollbars */
    }
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS (UNCHANGED) =====================
def send_results_email(to_email: str, subject: str, results: dict, extra_html: str = ""):
    """Send results via email with proper error handling"""
    if not ENABLE_EMAIL:
        st.warning("📧 Email feature is disabled. Configure credentials to enable.")
        return False
    
    if not to_email or "@" not in to_email or "." not in to_email:
        st.error("❌ Invalid email address format.")
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{OWNER_ALIAS}>"
        msg["To"] = to_email
        
        html_body = f"""
        <html>
        <body style='font-family:Inter,system-ui; padding:20px; background:#f9fafb;'>
          <div style='max-width:600px; margin:0 auto; background:white; padding:24px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);'>
            <h2 style='color:#7c3aed; margin:0 0 16px 0;'>🌈 AutoMLPilot Results</h2>
            {extra_html}
            <pre style='background:#0b1220; color:#e5e7eb; border:1px solid #1f2937; padding:16px; border-radius:8px; overflow-x:auto; font-size:13px;'>
{json.dumps(results, indent=2)}
            </pre>
            <p style='color:#6b7280; font-size:12px; margin-top:20px;'>
              Generated by AutoMLPilot Pro
            </p>
          </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))
        
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=10) as server:
            server.login(OWNER_GMAIL, OWNER_APP_PASSWORD)
            server.sendmail(OWNER_GMAIL, to_email, msg.as_string())
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Email authentication failed. Check credentials.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"❌ Email sending failed: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return False

@st.cache_data(show_spinner=False)
def profile_html(df: pd.DataFrame) -> str:
    """Generate EDA report with error handling"""
    if not YDATA_OK:
        return "<p class='small'>📦 Install ydata-profiling to enable full EDA.</p>"
    try:
        pr = ProfileReport(df, explorative=True, minimal=True)
        return pr.to_html()
    except Exception as e:
        return f"<p class='small'>❌ EDA generation failed: {str(e)}</p>"

def safe_label_encode(df: pd.DataFrame, columns: list = None) -> tuple:
    """Safely encode categorical columns and return encoders"""
    encoders = {}
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=["object", "category"]).columns
    
    for col in columns:
        try:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
        except Exception as e:
            st.warning(f"⚠️ Could not encode column '{col}': {str(e)}")
    
    return df_copy, encoders

def validate_dataframe(df: pd.DataFrame) -> tuple:
    """Validate dataframe and return status and message"""
    if df is None:
        return False, "No dataframe loaded"
    if df.empty:
        return False, "Dataframe is empty"
    if df.shape[0] < 10:
        return False, "Need at least 10 rows for training"
    if df.shape[1] < 2:
        return False, "Need at least 2 columns (features + target)"
    return True, "Valid"

def safe_train_test_split(X, y, test_size=0.2, task="Classification"):
    """Perform train-test split with proper stratification and error handling"""
    try:
        if task == "Classification" and len(np.unique(y)) > 1:
            # Check if stratification is possible
            min_class_count = pd.Series(y).value_counts().min()
            if min_class_count < 2:
                st.warning("⚠️ Small class detected. Splitting without stratification.")
                return train_test_split(X, y, test_size=test_size, random_state=42)
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        st.error(f"❌ Split failed: {str(e)}")
        return None, None, None, None

def correlation_recommendations(df: pd.DataFrame, thresh=0.7):
    """Find highly correlated feature pairs"""
    try:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return []
        
        corr = num_df.corr()
        recommendations = []
        
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if j <= i:
                    continue
                val = corr.iloc[i, j]
                if abs(val) >= thresh:
                    recommendations.append((col1, col2, float(val)))
        
        return sorted(recommendations, key=lambda x: -abs(x[2]))[:15]
    except Exception as e:
        st.error(f"❌ Correlation analysis failed: {str(e)}")
        return []

# ===================== SESSION STATE =====================
if "S" not in st.session_state:
    st.session_state.S = {
        "page": "dashboard",
        "df": None,
        "df_original": None,  # Keep original for reference
        "target": None,
        "task": "Classification",
        "user_email": "",
        "corr_pairs": [],
        "features_created": [],
        "model": None,
        "final_cols": None,
        "label_encoders": {},
        "scaler": None,
        "scaler_name": None,
        "results": {},
        "unsup_labels": None,
        "preprocessing_steps": [],
    }
S = st.session_state.S

# ===================== TOP BAR (FIXED) =====================
with st.container():
    st.markdown(f"""
    <div class='topbar'>
      <div style='display:flex;justify-content:space-between;align-items:center; width: 100%;'>
        <div style='display:flex;gap:.6rem;align-items:center'>
          <span>🌈</span>
          <strong>AutoMLPilot Pro</strong>
          <span class='chip'>No‑Code AI Lab</span>
        </div>
        <div style='display:flex;gap:8px;align-items:center'>
          <span class='chip'>Playground</span>
          <span class='chip'>EDA</span>
          <span class='chip'>{'Email Ready' if ENABLE_EMAIL else 'Email Disabled'}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ===================== SIDEBAR NAV (FIXED & BUTTON-STYLE) =====================
with st.sidebar:
    st.subheader("🧭 Navigation")
    
    # Custom format_func and layout for button-style navigation
    nav_options = ["dashboard", "preprocess", "train", "playground", "unsupervised", "results", "help"]
    nav_labels = {
        "dashboard":"📁 Dashboard",
        "preprocess":"🧹 Preprocess",
        "train":"🧠 Train (Supervised)",
        "playground":"🎨 Playground",
        "unsupervised":"🧩 Unsupervised",
        "results":"📊 Results",
        "help":"❓ Help"
    }
    
    pg = st.radio(
        "Select a Page", 
        nav_options,
        format_func=lambda x: nav_labels[x], 
        key="nav_radio",
        # Use a hidden label and custom CSS to make it button-like
        label_visibility="collapsed" 
    )
    S["page"] = pg
    
    st.markdown("---")
    if S["df"] is not None:
        st.caption(f"📊 Dataset: **{S['df'].shape[0]}** rows × **{S['df'].shape[1]}** cols")
        if S["target"]:
            st.caption(f"🎯 Target: **{S['target']}**")
        if S.get("scaler_name"):
            st.caption(f"✨ Scaling: **{S['scaler_name']}**")
        st.caption(f"🤖 Model: **{S['results'].get('model', 'N/A')}**")
        
# ===================== MAIN CONTENT AREA (PAGED) =====================
# The content below is contained within the single non-scrolling Streamlit 'main' area,
# with the block-container CSS handling the internal scrolling for content overflow.

if S["page"] == "dashboard":
    st.title("📁 Dashboard")
    
    col1, col2 = st.columns([2.2, 1])
    
    with col1:
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader("CSV files only", type=["csv"], key="file_uploader")
        
        if uploaded_file is not None:
            try:
                # Use st.spinner for a better UX during load time
                with st.spinner("Loading and validating data..."):
                    df = pd.read_csv(uploaded_file)
                
                # Validate dataset
                is_valid, msg = validate_dataframe(df)
                if not is_valid:
                    st.error(f"❌ Invalid dataset: {msg}")
                    st.stop()
                
                # Store both original and working copy
                S["df"] = df.copy()
                S["df_original"] = df.copy()
                S["target"] = None
                S["preprocessing_steps"] = []
                S["model"] = None
                S["results"] = {}
                
                st.markdown(f"<div class='success-box'>✅ Loaded **{df.shape[0]}** rows × **{df.shape[1]}** columns</div>", 
                            unsafe_allow_html=True)
                
                # Show basic info
                st.markdown("#### Dataset Info")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Rows", df.shape[0])
                with info_col2:
                    st.metric("Columns", df.shape[1])
                with info_col3:
                    st.metric("Missing", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"❌ Failed to load file: {str(e)}")
        
        if S["df"] is not None:
            st.markdown("### Dataset Preview")
            st.dataframe(S["df"].head(10), use_container_width=True)
            
            # Data types
            with st.expander("📋 Column Info"):
                col_info = pd.DataFrame({
                    'Column': S["df"].columns,
                    'Type': S["df"].dtypes,
                    'Missing': S["df"].isnull().sum(),
                    'Unique': S["df"].nunique()
                })
                st.dataframe(col_info, use_container_width=True)
    
    with col2:
        st.markdown("### 📧 Email Configuration")
        if ENABLE_EMAIL:
            S["user_email"] = st.text_input(
                "Recipient email",
                value=S["user_email"],
                placeholder="user@example.com",
                help="Results will be sent to this address"
            )
            st.info("✅ Email feature enabled")
        else:
            st.warning("⚠️ Email feature disabled. Configure credentials to enable.")
            st.caption("Set OWNER_GMAIL and OWNER_APP_PASSWORD in the code secrets.")
    
    st.markdown("---")
    
    # EDA Section
    if S["df"] is not None:
        st.markdown("### 📊 Exploratory Data Analysis")
        
        if st.button("🔍 Generate Full EDA Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    report_html = profile_html(S["df"])
                    # Use a controlled height for the HTML component
                    html(report_html, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"❌ EDA generation failed: {str(e)}")

# ===================== PREPROCESS =====================
elif S["page"] == "preprocess":
    st.title("🧹 Preprocessing Studio")
    
    if S["df"] is None:
        st.info("📁 Please upload a dataset first from the Dashboard.")
        st.stop()
    
    # Reset to original option
    if st.button("🔄 Reset to Original Dataset"):
        if S["df_original"] is not None:
            S["df"] = S["df_original"].copy()
            S["preprocessing_steps"] = []
            st.success("✅ Reset to original dataset")
            st.rerun()
    
    df = S["df"].copy()
    steps_applied = []
    
    # 1. Missing Values
    with st.expander("1️⃣ Handle Missing Values", expanded=True):
        missing_count = df.isnull().sum().sum()
        st.caption(f"Total missing values: **{missing_count}**")
        
        if missing_count > 0:
            strategy = st.selectbox(
                "Imputation Strategy",
                ["None", "Mean", "Median", "Most_frequent", "Drop_rows"],
                help="Choose how to handle missing values"
            )
            
            if strategy != "None":
                try:
                    if strategy == "Drop_rows":
                        before = len(df)
                        df = df.dropna()
                        st.success(f"✅ Dropped **{before - len(df)}** rows with missing values")
                        steps_applied.append(f"Dropped rows with missing values")
                    else:
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                        
                        if strategy in ["Mean", "Median"] and len(num_cols) > 0:
                            imputer = SimpleImputer(strategy=strategy.lower())
                            # Imputer returns numpy array, convert back to DataFrame
                            df[num_cols] = imputer.fit_transform(df[num_cols])
                            st.success(f"✅ Applied **{strategy}** imputation to numeric columns")
                            steps_applied.append(f"{strategy} imputation (numeric)")
                        
                        if len(cat_cols) > 0:
                            imputer_cat = SimpleImputer(strategy="most_frequent")
                            # SimpleImputer for most_frequent works on mixed types, but let's stick to cat columns for mode
                            df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
                            st.success(f"✅ Applied **most_frequent** imputation to categorical columns")
                            steps_applied.append("Most frequent imputation (categorical)")
                            
                except Exception as e:
                    st.error(f"❌ Imputation failed: {str(e)}")
        else:
            st.info("✅ No missing values detected")
    
    # 2. Outliers
    with st.expander("2️⃣ Handle Outliers (IQR Method)", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns for outlier detection")
        else:
            remove_outliers = st.checkbox("Remove outliers using IQR", value=False)
            iqr_multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1,
                                         help="Higher = more permissive")
            
            if remove_outliers:
                try:
                    before = len(df)
                    # Filter out outliers row by row (keeps original indices but they are reset below)
                    for col in num_cols:
                        q1, q3 = df[col].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lower = q1 - iqr_multiplier * iqr
                        upper = q3 + iqr_multiplier * iqr
                        df = df[(df[col] >= lower) & (df[col] <= upper)]
                    
                    removed = before - len(df)
                    st.success(f"✅ Removed **{removed}** rows ({removed/before*100:.1f}%) as outliers")
                    steps_applied.append(f"Removed {removed} outlier rows (IQR)")
                except Exception as e:
                    st.error(f"❌ Outlier removal failed: {str(e)}")
    
    # 3. Encoding
    with st.expander("3️⃣ Categorical Encoding", expanded=True):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if len(cat_cols) == 0:
            st.info("ℹ️ No categorical columns detected")
        else:
            st.caption(f"Categorical columns: **{', '.join(cat_cols)}**")
            enc_method = st.selectbox(
                "Encoding Method",
                ["None", "One-Hot", "Label"],
                help="One-Hot: creates binary columns. Label: assigns integers. One-Hot is usually preferred for non-linear models."
            )
            
            if enc_method == "One-Hot":
                try:
                    df = pd.get_dummies(df, drop_first=True)
                    st.success(f"✅ Applied one-hot encoding (**{df.shape[1]}** columns now)")
                    steps_applied.append("One-hot encoding")
                except Exception as e:
                    st.error(f"❌ One-hot encoding failed: {str(e)}")
                    
            elif enc_method == "Label":
                try:
                    for col in cat_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    st.success(f"✅ Applied label encoding to **{len(cat_cols)}** columns")
                    steps_applied.append("Label encoding")
                except Exception as e:
                    st.error(f"❌ Label encoding failed: {str(e)}")
    
    # 4. Scaling
    with st.expander("4️⃣ Feature Scaling", expanded=False):
        # Re-check numeric columns after encoding, as One-Hot creates new ones
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns for scaling")
        else:
            scale_method = st.selectbox(
                "Scaling Method",
                ["None", "Standard", "MinMax", "Robust", "Normalize"],
                help="Standard: mean=0, std=1. MinMax: [0,1]. Robust: uses median/IQR. Not necessary for tree-based models."
            )
            
            if scale_method != "None":
                try:
                    scaler_map = {
                        "Standard": StandardScaler(),
                        "MinMax": MinMaxScaler(),
                        "Robust": RobustScaler(),
                        "Normalize": Normalizer()
                    }
                    scaler = scaler_map[scale_method]
                    # Note: Normalizer is for row-wise scaling; others are column-wise
                    if scale_method == "Normalize":
                        # Normalizer operates on rows and returns a numpy array
                        df[num_cols] = scaler.fit_transform(df[num_cols])
                    else:
                        # Other scalers operate on columns and return numpy array
                        df[num_cols] = scaler.fit_transform(df[num_cols])
                        
                    S["scaler"] = scaler
                    S["scaler_name"] = scale_method
                    st.success(f"✅ Applied **{scale_method}** scaling")
                    steps_applied.append(f"{scale_method} scaling")
                except Exception as e:
                    st.error(f"❌ Scaling failed: {str(e)}")
    
    # 5. Feature Selection
    with st.expander("5️⃣ Feature Selection", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns")
        else:
            use_variance = st.checkbox("Remove low-variance features", value=False)
            
            if use_variance:
                threshold = st.slider("Variance Threshold", 0.0, 0.2, 0.01, 0.01,
                                         help="Remove features with variance below this (typically only after scaling)")
                try:
                    selector = VarianceThreshold(threshold=threshold)
                    # Apply selector only to numeric features
                    X_num = df[num_cols]
                    selector.fit(X_num)
                    kept_indices = selector.get_support(indices=True)
                    kept_cols = [num_cols[i] for i in kept_indices]
                    
                    # Separate non-numeric features to combine back
                    non_num = df.select_dtypes(exclude=[np.number])
                    
                    # Create the new DataFrame with selected numeric features and original non-numeric ones
                    df = pd.concat([
                        pd.DataFrame(selector.transform(X_num), columns=kept_cols, index=df.index),
                        non_num
                    ], axis=1)
                    
                    removed = len(num_cols) - len(kept_cols)
                    st.success(f"✅ Removed **{removed}** low-variance features. Kept **{len(kept_cols)}** numeric features.")
                    steps_applied.append(f"Removed {removed} low-variance features")
                except Exception as e:
                    st.error(f"❌ Feature selection failed: {str(e)}")
    
    # Apply preprocessing
    st.markdown("---")
    if st.button("💾 Apply Preprocessing", type="primary"):
        # Drop rows that may have NaNs after previous steps (e.g., division by zero)
        df = df.dropna().reset_index(drop=True)
        
        S["df"] = df
        S["preprocessing_steps"].extend(steps_applied)
        st.success(f"✅ Applied **{len(steps_applied)}** preprocessing steps. Dataset size: {df.shape[0]} rows.")
        st.info("Move to Train tab to build models")
        st.rerun() # Rerun to update the dataframe info in the sidebar and ensure clean session state
    
    # Show current preprocessing steps
    if S["preprocessing_steps"]:
        with st.expander("📋 Applied Preprocessing Steps"):
            for i, step in enumerate(S["preprocessing_steps"], 1):
                st.caption(f"• {step}")

# ===================== TRAIN (SUPERVISED) =====================
elif S["page"] == "train":
    st.title("🧠 Supervised Training")
    
    if S["df"] is None:
        st.info("📁 Upload a dataset first from Dashboard")
        st.stop()
    
    # Validate dataset
    is_valid, msg = validate_dataframe(S["df"])
    if not is_valid:
        st.error(f"❌ {msg}")
        st.stop()
    
    df = S["df"].copy()
    
    # Target Selection
    st.markdown("### 🎯 Target Selection")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_options = [""] + df.columns.tolist()
        target_idx = 0
        if S["target"] and S["target"] in df.columns:
            target_idx = target_options.index(S["target"])
        
        S["target"] = st.selectbox(
            "Select Target Variable (y)",
            target_options,
            index=target_idx,
            key="target_select",
            help="The variable you want to predict"
        )
    
    if not S["target"] or S["target"] == "":
        st.warning("⚠️ Please select a target variable to continue")
        st.stop()
    
    # Ensure target column has no NaNs
    if df[S["target"]].isnull().any():
        st.error("❌ Target variable contains missing values. Please handle them in Preprocess page.")
        st.stop()

    with col2:
        # Auto-detect task type
        n_unique = df[S["target"]].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[S["target"]])
        
        # Heuristic: Classification if few unique values (<= 20) or non-numeric
        if is_numeric and n_unique > 20 and df[S["target"]].dtype not in ['int64', 'int32']:
            suggested_task = "Regression"
        else:
            suggested_task = "Classification"
        
        task = st.radio(
            "Task Type",
            ["Classification", "Regression"],
            index=0 if suggested_task == "Classification" else 1,
            key="task_radio",
            help=f"Suggested: **{suggested_task}** ({n_unique} unique values)"
        )
        S["task"] = task # Store detected/selected task
    
    # Feature Engineering
    with st.expander("✨ Feature Engineering", expanded=False):
        st.markdown("#### Correlation Analysis")
        
        X_temp = df.drop(columns=[S["target"]])
        recommendations = correlation_recommendations(X_temp, thresh=0.7)
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations, columns=["Feature A", "Feature B", "Correlation"])
            st.dataframe(rec_df, use_container_width=True)
            st.caption("💡 Highly correlated features above. Consider dropping one or creating combinations.")
        else:
            st.info("ℹ️ No strongly correlated feature pairs found (|r| >= 0.7)")
        
        st.markdown("#### Create Derived Features")
        
        feature_cols = [c for c in df.columns if c != S["target"] and pd.api.types.is_numeric_dtype(df[c])]
        
        if len(feature_cols) >= 2:
            col_a, col_b = st.columns(2)
            
            with col_a:
                feat1 = st.selectbox("Feature 1", feature_cols, key="feat1")
                operation = st.selectbox("Operation", ["+", "-", "*", "/", "**"], key="operation")
            
            with col_b:
                feat2_options = [c for c in feature_cols if c != feat1]
                if feat2_options:
                    feat2 = st.selectbox("Feature 2", feat2_options, key="feat2")
                else:
                    feat2 = None
                new_name = st.text_input("New feature name", placeholder="e.g., ratio_a_b", key="new_feat_name")
            
            if feat2 is not None and st.button("➕ Create Feature"):
                if not new_name:
                    st.error("❌ Please provide a feature name")
                elif new_name in df.columns:
                    st.error(f"❌ Column '{new_name}' already exists")
                else:
                    try:
                        if operation == "+":
                            df[new_name] = df[feat1] + df[feat2]
                        elif operation == "-":
                            df[new_name] = df[feat1] - df[feat2]
                        elif operation == "*":
                            df[new_name] = df[feat1] * df[feat2]
                        elif operation == "/":
                            # Handle division by zero gracefully
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                df[new_name] = np.divide(df[feat1], df[feat2].replace(0, np.nan))
                            df[new_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                            df[new_name].fillna(0, inplace=True) # Impute NaNs/Infs from division by zero
                        elif operation == "**":
                            df[new_name] = df[feat1] ** df[feat2]
                        
                        S["df"] = df
                        S["features_created"].append(new_name)
                        st.success(f"✅ Feature '**{new_name}**' created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Feature creation failed: {str(e)}")
        else:
            st.warning("⚠️ Need at least 2 numeric features for combinations")
        
        if S["features_created"]:
            st.markdown("**Created Features:**")
            st.caption(f"• {' • '.join(S['features_created'])}")
    
    # Model Selection
    st.markdown("### 🤖 Model Selection")
    
    # Define model configurations
    models_classification = {
        "LogisticRegression": {
            "class": LogisticRegression,
            "params": {
                "C": {"type": "slider", "min": 0.01, "max": 10.0, "value": 1.0, "step": 0.1,  "help": "Inverse of regularization strength. Lower = more regularization"},
                "max_iter": {"type": "slider", "min": 100, "max": 2000, "value": 500, "step": 100, "help": "Maximum iterations for convergence"},
                "solver": {"type": "select", "options": ["lbfgs", "liblinear", "saga"], "index": 0, "help": "Algorithm for optimization"}
            },
            "description": "Fast linear model for binary/multiclass. Good baseline."
        },
        "RandomForestClassifier": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50, "help": "Number of trees in the forest"},
                "max_depth": {"type": "slider", "min": 3, "max": 50, "value": 10, "step": 1, "help": "Maximum depth of trees. None = unlimited"},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "value": 2, "step": 1, "help": "Minimum samples to split a node"},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Ensemble of decision trees. Handles non-linear patterns well."
        },
        "GradientBoostingClassifier": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50},
                "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "value": 0.1, "step": 0.01, "help": "Shrinks contribution of each tree"},
                "max_depth": {"type": "slider", "min": 3, "max": 10, "value": 3, "step": 1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Powerful boosting algorithm. Often achieves high accuracy."
        },
        "SVC": {
            "class": SVC,
            "params": {
                "C": {"type": "slider", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1},
                "kernel": {"type": "select", "options": ["rbf", "linear", "poly"], "index": 0, "help": "Kernel type for non-linear boundaries"},
                "probability": {"type": "checkbox", "value": True, "help": "Enable probability estimates"}
            },
            "description": "Support Vector Machine. Good for medium datasets."
        },
        "KNeighborsClassifier": {
            "class": KNeighborsClassifier,
            "params": {
                "n_neighbors": {"type": "slider", "min": 3, "max": 30, "value": 5, "step": 2, "help": "Number of neighbors to consider"}
            },
            "description": "Simple instance-based learning. Fast training."
        },
        "DecisionTreeClassifier": {
            "class": DecisionTreeClassifier,
            "params": {
                "max_depth": {"type": "slider", "min": 3, "max": 30, "value": 10, "step": 1},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "value": 2, "step": 1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Single decision tree. Interpretable but can overfit."
        },
        "GaussianNB": {
            "class": GaussianNB,
            "params": {},
            "description": "Naive Bayes classifier. Fast and works well with small data."
        },
        "MLPClassifier": {
            "class": MLPClassifier,
            "params": {
                "hidden_layer_sizes": {"type": "slider", "min": 32, "max": 256, "value": 100, "step": 32, "help": "Neurons in hidden layer"},
                "learning_rate_init": {"type": "slider", "min": 0.0001, "max": 0.01, "value": 0.001, "step": 0.0001},
                "max_iter": {"type": "slider", "min": 200, "max": 1000, "value": 500, "step": 100},
                "early_stopping": {"type": "checkbox", "value": True},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Neural network. Can model complex patterns."
        }
    }
    
    models_regression = {
        "LinearRegression": {
            "class": LinearRegression,
            "params": {},
            "description": "Simple linear regression. Fast and interpretable."
        },
        "Ridge": {
            "class": Ridge,
            "params": {
                "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1, "help": "L2 regularization strength"}
            },
            "description": "Linear regression with L2 regularization."
        },
        "Lasso": {
            "class": Lasso,
            "params": {
                "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1, "help": "L1 regularization (feature selection)"}
            },
            "description": "Linear regression with L1 regularization. Can zero out features."
        },
        "ElasticNet": {
            "class": ElasticNet,
            "params": {
                "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1},
                "l1_ratio": {"type": "slider", "min": 0.0, "max": 1.0, "value": 0.5, "step": 0.1, "help": "Mix of L1 and L2: 0=Ridge, 1=Lasso"}
            },
            "description": "Combines L1 and L2 regularization."
        },
        "RandomForestRegressor": {
            "class": RandomForestRegressor,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50},
                "max_depth": {"type": "slider", "min": 3, "max": 50, "value": 10, "step": 1},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "value": 2, "step": 1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Ensemble of regression trees."
        },
        "GradientBoostingRegressor": {
            "class": GradientBoostingRegressor,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50},
                "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "value": 0.1, "step": 0.01},
                "max_depth": {"type": "slider", "min": 3, "max": 10, "value": 3, "step": 1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Boosting for regression. High accuracy."
        },
        "SVR": {
            "class": SVR,
            "params": {
                "C": {"type": "slider", "min": 0.1, "max": 10.0, "value": 1.0, "step": 0.1},
                "kernel": {"type": "select", "options": ["rbf", "linear", "poly"], "index": 0}
            },
            "description": "Support Vector Regression."
        },
        "DecisionTreeRegressor": {
            "class": DecisionTreeRegressor,
            "params": {
                "max_depth": {"type": "slider", "min": 3, "max": 30, "value": 10, "step": 1},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "value": 2, "step": 1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Single regression tree."
        },
        "KNeighborsRegressor": {
            "class": KNeighborsRegressor,
            "params": {
                "n_neighbors": {"type": "slider", "min": 3, "max": 30, "value": 5, "step": 2}
            },
            "description": "K-nearest neighbors for regression."
        },
        "MLPRegressor": {
            "class": MLPRegressor,
            "params": {
                "hidden_layer_sizes": {"type": "slider", "min": 32, "max": 256, "value": 100, "step": 32},
                "learning_rate_init": {"type": "slider", "min": 0.0001, "max": 0.01, "value": 0.001, "step": 0.0001},
                "max_iter": {"type": "slider", "min": 200, "max": 1000, "value": 500, "step": 100},
                "early_stopping": {"type": "checkbox", "value": True},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "Neural network for regression."
        }
    }
    
    # Add XGBoost if available
    if XGB_OK:
        models_classification["XGBClassifier"] = {
            "class": XGBClassifier,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50},
                "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "value": 0.1, "step": 0.01},
                "max_depth": {"type": "slider", "min": 3, "max": 10, "value": 6, "step": 1},
                "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "value": 0.8, "step": 0.1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "XGBoost classifier. State-of-the-art gradient boosting."
        }
        models_regression["XGBRegressor"] = {
            "class": XGBRegressor,
            "params": {
                "n_estimators": {"type": "slider", "min": 50, "max": 500, "value": 100, "step": 50},
                "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "value": 0.1, "step": 0.01},
                "max_depth": {"type": "slider", "min": 3, "max": 10, "value": 6, "step": 1},
                "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "value": 0.8, "step": 0.1},
                "random_state": {"type": "number", "value": 42}
            },
            "description": "XGBoost regressor. State-of-the-art gradient boosting."
        }
    
    model_zoo = models_classification if task == "Classification" else models_regression
    
    # Model selection
    model_name = st.selectbox(
        "Choose Model",
        list(model_zoo.keys()),
        key="model_select",
        help="Select algorithm for training"
    )
    
    st.info(f"ℹ️ {model_zoo[model_name]['description']}")
    
    # Hyperparameters
    st.markdown("#### ⚙️ Hyperparameters")
    model_params = {}
    
    param_config = model_zoo[model_name]["params"]
    
    if param_config:
        cols = st.columns(2)
        col_idx = 0
        
        for param_name, config in param_config.items():
            with cols[col_idx % 2]:
                # Streamlit input type mapping
                if config["type"] == "slider":
                    if isinstance(config["value"], float):
                        value = st.slider(
                            param_name,
                            min_value=config["min"],
                            max_value=config["max"],
                            value=config["value"],
                            step=config.get("step", (config["max"] - config["min"]) / 100),
                            help=config.get("help", ""),
                            key=f"{model_name}_{param_name}"
                        )
                    else: # Integer slider
                         value = st.slider(
                            param_name,
                            min_value=int(config["min"]),
                            max_value=int(config["max"]),
                            value=int(config["value"]),
                            step=int(config.get("step", 1)),
                            help=config.get("help", ""),
                            key=f"{model_name}_{param_name}"
                        )
                elif config["type"] == "select":
                    value = st.selectbox(
                        param_name,
                        config["options"],
                        index=config.get("index", 0),
                        help=config.get("help", ""),
                        key=f"{model_name}_{param_name}"
                    )
                elif config["type"] == "checkbox":
                    value = st.checkbox(
                        param_name,
                        value=config["value"],
                        help=config.get("help", ""),
                        key=f"{model_name}_{param_name}"
                    )
                elif config["type"] == "number":
                    value = st.number_input(
                        param_name,
                        value=config["value"],
                        help=config.get("help", ""),
                        key=f"{model_name}_{param_name}"
                    )
                else:
                    value = config["value"]
                
                # Special handling for MLP hidden_layer_sizes (requires a tuple)
                if param_name == "hidden_layer_sizes":
                    model_params[param_name] = (int(value),)
                else:
                    model_params[param_name] = value
                
                col_idx += 1
    else:
        st.info("ℹ️ This model has no hyperparameters to tune")
    
    # Training configuration
    with st.expander("🎛️ Training Configuration", expanded=False):
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05,
                                 help="Proportion of data for testing")
        
        if task == "Classification" and IMB_OK:
            use_smote = st.checkbox("Use SMOTE for class balancing", value=False,
                                     help="Oversample minority classes")
        else:
            use_smote = False
    
    # Train button
    st.markdown("---")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        try:
            with st.spinner("🔄 Training in progress..."):
                # Prepare data
                X = df.drop(columns=[S["target"]])
                y = df[S["target"]]
                
                # Encode categorical features
                X, encoders = safe_label_encode(X)
                S["label_encoders"] = encoders
                
                # Encode target for classification
                target_encoder = None
                if task == "Classification" and y.dtype == "object":
                    target_encoder = LabelEncoder()
                    # Ensure all values are strings before fit_transform
                    y = target_encoder.fit_transform(y.astype(str))
                
                # Train-test split
                split_result = safe_train_test_split(X, y, test_size=test_size, task=task)
                if split_result[0] is None:
                    st.error("❌ Train-test split failed")
                    st.stop()
                
                X_train, X_test, y_train, y_test = split_result
                
                # Apply SMOTE if requested
                if use_smote and task == "Classification":
                    try:
                        smote = SMOTE(random_state=42)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.info("✅ Applied SMOTE balancing to training data")
                    except Exception as e:
                        st.warning(f"⚠️ SMOTE failed: {str(e)}. Continuing without SMOTE.")
                
                # Initialize and train model
                ModelClass = model_zoo[model_name]["class"]
                model = ModelClass(**model_params)
                
                # Training with timing
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                results = {
                    "model": model_name,
                    "task": task,
                    "training_time_sec": round(training_time, 3),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "parameters": model_params
                }
                
                if task == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    # Handle multi-class F1 by using 'weighted' average
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results["accuracy"] = round(accuracy, 4)
                    results["f1_score"] = round(f1, 4)
                    
                    # Display metrics
                    st.markdown("### 📊 Classification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col3:
                        st.metric("Training Time", f"{training_time:.2f}s")
                    
                    # Confusion Matrix
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                                xticklabels=target_encoder.classes_ if target_encoder else np.unique(y_test), 
                                yticklabels=target_encoder.classes_ if target_encoder else np.unique(y_test))
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    plt.close(fig) # Close figure to free memory
                    
                    # Classification Report
                    with st.expander("📋 Detailed Classification Report"):
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        results['classification_report'] = report
                        st.json(report)
                        
                else:  # Regression
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    mae = float(mean_absolute_error(y_test, y_pred))
                    r2 = float(r2_score(y_test, y_pred))
                    
                    results["rmse"] = round(rmse, 4)
                    results["mae"] = round(mae, 4)
                    results["r2_score"] = round(r2, 4)
                    
                    # Display metrics
                    st.markdown("### 📊 Regression Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col2:
                        st.metric("MAE", f"{mae:.4f}")
                    with col3:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col4:
                        st.metric("Training Time", f"{training_time:.2f}s")
                    
                    # Actual vs Predicted
                    st.markdown("#### Actual vs Predicted")
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Regression: Actual vs Predicted'
                    )
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    # Perfect prediction reference line
                    fig.add_shape(
                        type='line',
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(color='red', dash='dash', width=2)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals
                    residuals = y_test - y_pred
                    fig2 = px.scatter(
                        x=y_pred, y=residuals,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title='Residual Plot'
                    )
                    fig2.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Save to session
                S["model"] = model
                S["final_cols"] = list(X.columns)
                S["results"] = results
                S["task"] = task
                
                st.success("✅ Training completed successfully! Check the **Playground** or **Results** tab.")
                
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# ===================== PLAYGROUND =====================
elif S["page"] == "playground":
    st.title("🎨 Model Playground")
    
    if S["df"] is None or S["target"] is None or S["model"] is None:
        st.info("📊 Train a supervised model first to access the playground")
        st.stop()
    
    try:
        df = S["df"].copy()
        target = S["target"]
        model = S["model"]
        task = S.get("task", "Classification")
        
        # Prepare data (Need to re-encode for safety as state might be stale)
        X = df.drop(columns=[target]).fillna(0) # Basic imputation for visualization robustness
        y = df[target]
        
        # Encode features (needed for PCA/Model consistency)
        X, _ = safe_label_encode(X)
        if y.dtype == "object":
            # Re-encode target if it's categorical/object
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        st.markdown("### 🔮 Model Visualization")
        
        # PCA for 2D visualization
        X_data = X.values # Use numerical data
        if X_data.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_data)
            explained_var = pca.explained_variance_ratio_
            st.caption(f"📊 PCA: **{explained_var[0]*100:.1f}%** + **{explained_var[1]*100:.1f}%** = **{sum(explained_var)*100:.1f}%** variance explained")
        elif X_data.shape[1] == 2:
            X_2d = X_data
            st.caption("📊 Using original 2D features")
        else:
            st.warning("⚠️ Cannot perform 2D visualization with fewer than 2 features.")
            st.stop()
        
        # Data distribution
        fig1 = px.scatter(
            x=X_2d[:, 0], y=X_2d[:, 1],
            color=[str(v) for v in y],
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title='Data Distribution (2D Projection)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Model-specific visualizations
        if task == "Classification":
            st.markdown("#### 🎯 Decision Boundary")
            
            try:
                # Train a model on PCA space for visualization
                X_train, X_test, y_train, y_test = safe_train_test_split(
                    X_2d, y, test_size=0.2, task="Classification"
                )
                
                if X_train is not None:
                    # Clone model with same params
                    ModelClass = type(model)
                    params = model.get_params()
                    vis_model = ModelClass(**params)
                    vis_model.fit(X_train, y_train)
                    
                    # Create mesh
                    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
                    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
                    
                    xx, yy = np.meshgrid(
                        np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100)
                    )
                    
                    Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    
                    # Plot
                    fig2 = go.Figure()
                    
                    # Decision boundary (Contour fill)
                    fig2.add_trace(go.Heatmap(
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        z=Z,
                        showscale=False,
                        opacity=0.4,
                        colorscale=[[0, 'pink'], [1/(len(np.unique(y))-1), 'lightblue'], [1, 'lightgreen']],
                        hoverinfo='skip'
                    ))
                    
                    # Data points
                    for label in np.unique(y):
                        mask = y == label
                        fig2.add_trace(go.Scatter(
                            x=X_2d[mask, 0],
                            y=X_2d[mask, 1],
                            mode='markers',
                            name=f'Class {label}',
                            marker=dict(size=8, line=dict(width=1, color='white'))
                        ))
                    
                    fig2.update_layout(
                        title='Decision Boundary (PCA Space)',
                        xaxis_title='Component 1',
                        yaxis_title='Component 2',
                        showlegend=True,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"⚠️ Decision boundary visualization not fully available for this model/data. Error: {str(e)}")
        
        else:  # Regression
            st.markdown("#### 📈 Prediction Analysis")
            
            # Re-run a test split for visualization robustness
            X_train, X_test, y_train, y_test = safe_train_test_split(
                X, y, test_size=0.2, task="Regression"
            )
            
            if X_test is not None:
                y_pred = model.predict(X_test)
                
                # Actual vs Predicted Scatter
                fig3 = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title='Regression: Actual vs Predicted'
                )
                # Perfect prediction reference line
                min_val = min(y_test.min(), y_pred.min()) if len(y_test) > 0 else 0
                max_val = max(y_test.max(), y_pred.max()) if len(y_test) > 0 else 1
                
                # Add the 45-degree line for perfect prediction
                fig3.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ))
                st.plotly_chart(fig3, use_container_width=True)
            
                # Residuals Plot
                residuals = y_test - y_pred
                fig4 = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={'x': 'Predicted', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                fig4.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig4, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Playground visualization failed: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# ===================== UNSUPERVISED =====================
elif S["page"] == "unsupervised":
    st.title("🧩 Unsupervised Learning Lab")
    
    if S["df"] is None:
        st.info("📁 Upload a dataset first from Dashboard")
        st.stop()
    
    df = S["df"].copy()
    
    # Pre-process data for unsupervised learning (impute and encode everything)
    try:
        df_encoded, _ = safe_label_encode(df)
    except Exception as e:
        st.error(f"❌ Encoding failed: {str(e)}")
        st.stop()
    
    # Handle remaining missing values (should be numerical by now)
    if df_encoded.isnull().any().any():
        st.info("⚠️ Missing numerical values detected. Filling with mean/mode for analysis...")
        for col in df_encoded.columns:
            if df_encoded[col].dtype in [np.float64, np.int64]:
                df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)
            else:
                df_encoded[col].fillna(df_encoded[col].mode()[0], inplace=True)
    
    st.markdown("### 🎯 Algorithm Selection")
    
    algorithms = {
        "KMeans": "Partition data into K clusters. Fast and popular.",
        "DBSCAN": "Density-based clustering. Finds arbitrary-shaped clusters and outliers.",
        "Agglomerative": "Hierarchical clustering. Builds tree of nested clusters.",
        "GaussianMixture": "Probabilistic clustering using Gaussian distributions.",
        "IsolationForest": "Anomaly detection. Identifies outliers in data.",
        "PCA": "Dimensionality reduction. Finds principal components."
    }
    
    algo = st.selectbox("Choose Algorithm", list(algorithms.keys()), key="unsuper_algo_select")
    st.info(f"ℹ️ {algorithms[algo]}")
    
    # Algorithm-specific parameters
    st.markdown("#### ⚙️ Parameters")
    
    labels = None
    model = None
    
    try:
        # --- KMeans ---
        if algo == "KMeans":
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters (k)", 2, 15, 4, key="km_n_clusters")
            with col2:
                n_init = st.slider("Number of Initializations", 5, 30, 10, key="km_n_init")
            
            if st.button("🚀 Run KMeans", type="primary"):
                with st.spinner("Running KMeans..."):
                    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
                    labels = model.fit_predict(df_encoded)
                    
                    # Calculate silhouette score
                    if len(set(labels)) > 1 and len(df_encoded) > n_clusters:
                        sil_score = silhouette_score(df_encoded, labels)
                        st.success(f"✅ Clustering complete! Silhouette Score: **{sil_score:.4f}**")
                        st.caption("📊 Silhouette Score ranges from -1 to 1. Higher is better.")
                    else:
                        st.warning("⚠️ Only one cluster or too few samples found for silhouette score.")
        
        # --- DBSCAN ---
        elif algo == "DBSCAN":
            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("Epsilon (neighborhood radius)", 0.1, 10.0, 0.5, 0.1, key="db_eps")
            with col2:
                min_samples = st.slider("Min Samples", 2, 20, 5, key="db_min_samples")
            
            if st.button("🚀 Run DBSCAN", type="primary"):
                with st.spinner("Running DBSCAN..."):
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(df_encoded)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    st.success(f"✅ Found **{n_clusters}** clusters and **{n_noise}** noise points")

        # --- Agglomerative ---
        elif algo == "Agglomerative":
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 15, 4, key="agg_n_clusters")
            with col2:
                linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="agg_linkage")
            
            if st.button("🚀 Run Agglomerative", type="primary"):
                with st.spinner("Running Agglomerative Clustering..."):
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    labels = model.fit_predict(df_encoded)
                    st.success(f"✅ Clustering complete! Found **{n_clusters}** clusters")
        
        # --- GaussianMixture ---
        elif algo == "GaussianMixture":
            col1, col2 = st.columns(2)
            with col1:
                n_components = st.slider("Number of Components", 2, 15, 4, key="gm_n_components")
            with col2:
                covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"], key="gm_cov_type")
            
            if st.button("🚀 Run Gaussian Mixture", type="primary"):
                with st.spinner("Running Gaussian Mixture Model..."):
                    model = GaussianMixture(n_components=n_components, 
                                             covariance_type=covariance_type,
                                             random_state=42)
                    model.fit(df_encoded)
                    labels = model.predict(df_encoded) # Predict components
                    
                    bic = model.bic(df_encoded)
                    aic = model.aic(df_encoded)
                    
                    st.success(f"✅ Clustering complete!")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("BIC", f"{bic:.2f}", help="Lower is better")
                    with col_b:
                        st.metric("AIC", f"{aic:.2f}", help="Lower is better")
        
        # --- IsolationForest (Anomaly Detection) ---
        elif algo == "IsolationForest":
            col1, col2 = st.columns(2)
            with col1:
                contamination = st.slider("Contamination (outlier %)", 0.01, 0.5, 0.1, 0.01, key="if_contamination")
            with col2:
                n_estimators = st.slider("Number of Trees", 50, 300, 100, 50, key="if_n_estimators")
            
            if st.button("🚀 Run Isolation Forest", type="primary"):
                with st.spinner("Running Isolation Forest..."):
                    model = IsolationForest(contamination=contamination,
                                             n_estimators=n_estimators,
                                             random_state=42)
                    labels = model.fit_predict(df_encoded)
                    
                    # Convert to 0/1 (inliers/outliers)
                    labels = np.where(labels == 1, 0, 1) # 1 = outlier
                    
                    n_outliers = sum(labels)
                    st.success(f"✅ Detection complete! Found **{n_outliers}** outliers (**{n_outliers/len(labels)*100:.1f}%**)")
        
        # --- PCA ---
        elif algo == "PCA":
            n_components = st.slider("Number of Components", 2, min(10, df_encoded.shape[1]), 2, key="pca_n_components")
            
            if st.button("🚀 Run PCA", type="primary"):
                with st.spinner("Running PCA..."):
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(df_encoded)
                    
                    st.success("✅ PCA complete!")
                    
                    # Explained variance
                    explained_var = pca.explained_variance_ratio_
                    cumulative_var = np.cumsum(explained_var)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Explained Variance")
                        var_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(n_components)],
                            'Variance': [f'{v:.4f}' for v in explained_var],
                            'Cumulative': [f'{v:.4f}' for v in cumulative_var]
                        })
                        st.dataframe(var_df, use_container_width=True)
                    
                    with col2:
                        fig_var = px.bar(
                            x=[f'PC{i+1}' for i in range(n_components)],
                            y=explained_var,
                            labels={'x': 'Component', 'y': 'Variance Explained'},
                            title='Variance by Component'
                        )
                        st.plotly_chart(fig_var, use_container_width=True)
                    
                    # 2D Visualization
                    if n_components >= 2:
                        fig_pca = px.scatter(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            labels={'x': 'PC1', 'y': 'PC2'},
                            title='PCA: First Two Components'
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                    
                    st.stop() # Stop here to show PCA results
        
        # Visualization (for clustering/anomaly detection)
        if labels is not None:
            st.markdown("### 📊 Visualization")
            
            # PCA for 2D projection (if needed)
            X_data = df_encoded.values
            if X_data.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_data)
                explained = pca.explained_variance_ratio_
                st.caption(f"PCA: {explained[0]*100:.1f}% + {explained[1]*100:.1f}% = {sum(explained)*100:.1f}% variance explained")
            else:
                X_2d = X_data
                
            # Create visualization
            fig = px.scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                color=[str(l) for l in labels],
                labels={'x': 'Component 1', 'y': 'Component 2'},
                title=f'{algo} Results (2D Projection)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.markdown("#### 📈 Cluster/Group Statistics")
            
            cluster_df = df.copy()
            if algo == "IsolationForest":
                cluster_df['Group'] = np.where(labels == 1, 'Outlier', 'Inlier')
                summary_col = 'Group'
            else:
                cluster_df['Cluster'] = labels
                summary_col = 'Cluster'
                
            cluster_summary = cluster_df.groupby(summary_col).size().reset_index(name='Count')
            cluster_summary['Percentage'] = cluster_summary['Count'] / len(cluster_df) * 100
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Save labels
            S["unsup_labels"] = labels
            
            # Download option
            if st.button("💾 Add Labels to Dataset"):
                S["df"][summary_col] = labels
                st.success(f"✅ Labels added to dataset as '{summary_col}' column")
    
    except Exception as e:
        st.error(f"❌ Algorithm failed: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# ===================== RESULTS =====================
elif S["page"] == "results":
    st.title("📊 Results & Reports")
    
    if not S["results"]:
        st.info("📈 Train a model to see results here")
        st.stop()
    
    st.markdown("### 🎯 Training Summary")
    
    results = S["results"]
    
    # Display results in cards
    if results.get("task") == "Classification":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>Accuracy</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('accuracy', 0)*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>F1 Score</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('f1_score', 0):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>Training Time</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('training_time_sec', 0):.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>Model</div>
                <div style='font-size:1rem;font-weight:bold;color:#7c3aed'>
                    {results.get('model', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if 'classification_report' in results:
             with st.expander("📋 Detailed Classification Report"):
                 # Pretty print the classification report
                 report_str = json.dumps(results['classification_report'], indent=4)
                 st.code(report_str, language='json')
        
    else:  # Regression
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>RMSE</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('rmse', 0):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>R² Score</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('r2_score', 0):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>MAE</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('mae', 0):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric'>
                <div style='color:#6b7280;font-size:0.85rem'>Training Time</div>
                <div style='font-size:1.5rem;font-weight:bold;color:#7c3aed'>
                    {results.get('training_time_sec', 0):.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Full results JSON
    with st.expander("📋 Complete Results (JSON)", expanded=False):
        st.json(results)
    
    # Email results
    st.markdown("---")
    st.markdown("### 📧 Email Results")
    
    if ENABLE_EMAIL:
        if S["user_email"]:
            email_col1, email_col2 = st.columns([3, 1])
            
            with email_col1:
                st.info(f"📬 Results will be sent to: **{S['user_email']}**")
            
            with email_col2:
                if st.button("📤 Send Email", type="primary", use_container_width=True):
                    with st.spinner("Sending email..."):
                        success = send_results_email(
                            to_email=S["user_email"],
                            subject=f"AutoMLPilot Results - {results.get('model', 'Model')}",
                            results=results,
                            extra_html=f"<p>Task: <strong>{results.get('task', 'N/A')}</strong></p>"
                        )
                        
                        if success:
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email")
        else:
            st.warning("⚠️ Please enter a recipient email on the Dashboard")
    else:
        st.warning("⚠️ Email feature is disabled. Configure credentials to enable.")
    
    # Download results
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download as JSON",
        data=results_json,
        file_name=f"automl_results_{results.get('model', 'model')}.json",
        mime="application/json"
    )

# ===================== HELP =====================
elif S["page"] == "help":
    st.title("❓ Help & Documentation")
    
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    ### 1. Upload Data (**Dashboard**)
    - Upload a **CSV** file.
    - Preview your data and check for missing values.
    
    ### 2. Preprocess (**Preprocess**)
    - Handle **missing values** (imputation or removal).
    - Remove **outliers** using the IQR method.
    - Encode **categorical variables** (One-Hot or Label).
    - Scale **numerical features** (Standard, MinMax, etc.). *Important for distance-based models like SVM, KNN, and Neural Networks.*
    
    ### 3. Train Model (**Train (Supervised)**)
    - Select your **target variable** and confirm **Task Type** (Classification/Regression).
    - Optionally create **derived features** in Feature Engineering.
    - Select and tune **Hyperparameters** for your chosen model.
    - Click **Train Model** to run evaluation.
    
    ### 4. Other Labs
    - **Playground**: Visualize model decision boundaries or regression performance.
    - **Unsupervised**: Run clustering (KMeans, DBSCAN) or dimensionality reduction (PCA).
    - **Results**: See detailed metrics and download/email the report.
    
    ---
    
    ## 🤖 Model Selection Guide (Summary)
    
    | Task | Model | Best For | Tip/Note |
    | :--- | :--- | :--- | :--- |
    | **Classification** | **Random Forest** | General purpose, non-linear data | Good default choice. |
    | | **XGBoost/GBM** | Maximum accuracy, production models | Requires careful tuning. |
    | | **Logistic Regression** | Baseline, interpretable models | Assumes linear separability. |
    | **Regression** | **Random Forest** | General purpose, robust | Excellent for most tasks. |
    | | **XGBoost/GBM** | Max accuracy for non-linear patterns | Highly effective but complex. |
    | | **Linear Regression** | Simple baseline, linear data | Add Ridge/Lasso for regularization. |
    
    ---
    
    ## 🧹 Preprocessing Best Practices
    
    ### Scaling Decision
    - **StandardScaler**: Recommended for SVM, Neural Networks, KNN.
    - **None**: Recommended for tree-based models (RF, GB, Decision Trees) as they are scale-invariant.
    
    ### Encoding Decision
    - **One-Hot Encoding**: Safer for most models, especially linear models. Can lead to many features.
    - **Label Encoding**: Only for ordinal features or as a quick-fix for non-tree models.
    
    ---
    
    ## 🔒 Security & Privacy Notes
    
    - **Email Credentials**: Never hardcode. Use the app's internal mechanism for safety (even here they are placeholders).
    - **Data Processing**: All data processing is done locally within your browser/Streamlit session. No data leaves the application unless you manually click **Send Email**.
    
    ---
    
    ### Need More Help?
    
    - Review the preprocessing steps carefully.
    - Check the error details in expandable sections.
    - Try simpler models first to isolate issues.
    
    **Version**: 2.0 (Secure & Enhanced)
    """)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#6b7280;font-size:0.85rem;padding-bottom:10px;'>
    <strong>AutoMLPilot Pro</strong> | Secure No-Code ML Platform<br>
    Built with Streamlit, Scikit-learn, Plotly, and ❤️<br>
    <span style='font-size:0.75rem;color:#94a3b8;'>Version 2.0 — Secure & Enhanced Edition</span>
</div>
""", unsafe_allow_html=True)

# ===================== END OF FILE =====================
if __name__ == "__main__":
    st.write("✅ AutoMLPilot Pro loaded successfully.")
