import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess
import json
import os
import sys

# ==================== CREDENTIALS SECTION ====================
# ENTER YOUR CREDENTIALS HERE:

SMTP_SENDER_EMAIL = "rhreddy4748@gmail.com"  # Your Gmail address
SMTP_APP_PASSWORD = "zujpoggswfcpxwjs"  # Gmail App Password (not regular password)
SMTP_RECIPIENT_EMAIL = "ragiharshithreddy@email.com"  # Email to receive notifications

# =============================================================

# CSS Styling
GLASSY_CSS = """
<style>
    .block-container {padding: 1rem 2rem;}
    h1 {color: #E6E6FA; font-weight: 800; text-shadow: 1px 1px 3px rgba(0,0,0,0.4);}
    h2, h3 {color: #DDA0DD; padding-bottom: 5px;}
    .stCard, [data-testid="stExpander"], .glass-card {
        background: rgba(255,255,255,0.15);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.4);
        padding: 15px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .stCard:hover, [data-testid="stExpander"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 10px;
        border-left: 5px solid #FFD700;
    }
    [data-testid="stMetricValue"] {font-size: 28px; color: #7B68EE; font-weight: bold;}
    .main {
        background: url("https://images.unsplash.com/photo-1542831371-29b0f74f9d13?fit=crop&w=1200&h=800&q=80") center/cover no-repeat fixed;
    }
</style>
"""

st.set_page_config(page_title="No-Code ML Explorer", layout="wide")
st.markdown(GLASSY_CSS, unsafe_allow_html=True)

# Caching
@st.cache_data(ttl=3600)
def load_data(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner="Generating profile report...")
def generate_profile(df):
    return ProfileReport(df, explorative=True, minimal=True, 
                        correlations={"auto": {"calculate": False}}).to_html()

# Email Function
def send_email(results, model_name, task_type):
    if not SMTP_SENDER_EMAIL or not SMTP_APP_PASSWORD or not SMTP_RECIPIENT_EMAIL:
        return False, "Email credentials not configured"
    if "@" not in SMTP_SENDER_EMAIL or len(SMTP_APP_PASSWORD) < 10:
        return False, "Invalid email credentials"
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"✅ ML Training Complete: {model_name}"
        msg["From"] = SMTP_SENDER_EMAIL
        msg["To"] = SMTP_RECIPIENT_EMAIL
        
        html_body = f"""
        <html><body>
            <h2 style="color:#4B0082;">ML Training Complete</h2>
            <p><b>{model_name} ({task_type})</b> training finished.</p>
            <h3 style="color:#DDA0DD;">Results:</h3>
            <pre style="background:#f4f4f4;border:1px solid #ddd;padding:10px;border-radius:5px;">
{json.dumps(results, indent=2)}</pre>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, ssl.create_default_context()) as server:
            server.login(SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_SENDER_EMAIL, SMTP_RECIPIENT_EMAIL, msg.as_string())
        return True, "Email sent!"
    except Exception as e:
        return False, f"Email failed: {str(e)[:100]}"

# Session State
if 'current_df' not in st.session_state:
    st.session_state.update({
        'logged_in': False, 'current_df': None, 'original_df': None,
        'target_col': None, 'feature_cols': [], 'model': None,
        'scaler': None, 'le_target': None, 'page': 'home',
        'local_results': {}, 'kaggle_results': {}
    })

# Login
def login_page():
    st.title("🔒 Login (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Sign in (Simulated)", use_container_width=True, type="primary"):
            st.session_state.logged_in = True
            st.session_state.page = 'upload'
            st.rerun()
    with col2:
        if st.button("➡ Continue as Guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.page = 'upload'
            st.rerun()

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Navbar
PAGES = {
    'upload': "📁 Upload", 'preprocess': "🧹 Preprocess",
    'explore': "📊 Explore", 'train': "🧠 Train", 'predict': "🔮 Predict"
}

cols = st.columns(len(PAGES))
for i, (key, title) in enumerate(PAGES.items()):
    if cols[i].button(f"{'✅' if key in ['upload','preprocess'] and st.session_state.current_df else '➡'} {title}", key=f"nav_{key}"):
        st.session_state.page = key
        st.rerun()

st.title("✨ No-Code ML Explorer")

# Sidebar Upload
with st.sidebar:
    st.title("📂 File Management")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded:
        try:
            new_df = load_data(uploaded)
            if st.session_state.original_df is None or not st.session_state.original_df.equals(new_df):
                st.session_state.update({
                    'original_df': new_df, 'current_df': new_df.copy(),
                    'local_results': {}, 'kaggle_results': {},
                    'model': None, 'page': 'preprocess'
                })
                st.success("File uploaded!")
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()
    
    if st.session_state.current_df is not None:
        st.markdown("---")
        st.subheader("Data Status")
        st.info(f"Rows: {st.session_state.current_df.shape[0]}, Cols: {st.session_state.current_df.shape[1]}")
        csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download", csv, "processed.csv", "text/csv")

if st.session_state.current_df is None:
    st.info("👆 Upload a CSV to begin")
    st.stop()

df = st.session_state.current_df.copy()

# ==================== PREPROCESS PAGE ====================
if st.session_state.page == 'preprocess':
    st.header("🧹 Preprocessing Pipeline")
    st.info(f"Shape: {df.shape}")
    
    # Drop Columns
    with st.expander("1️⃣ Drop Columns", expanded=True):
        cols_drop = st.multiselect("Select columns to drop:", df.columns.tolist())
        if st.button("Apply Drop", type="primary"):
            if cols_drop:
                df.drop(columns=cols_drop, inplace=True)
                st.session_state.current_df = df
                st.success(f"Dropped {len(cols_drop)} columns")
                st.rerun()
    
    # Missing Values
    with st.expander("2️⃣ Missing Values", expanded=True):
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if missing_cols.empty:
            st.info("No missing values!")
        else:
            st.dataframe(missing_cols)
            method = st.selectbox("Method", ["Drop Rows", "Mean/Mode Imputation"])
            if st.button("Apply", key="missing"):
                if method == "Drop Rows":
                    before = len(df)
                    df.dropna(inplace=True)
                    st.success(f"Dropped {before - len(df)} rows")
                else:
                    for col in missing_cols.index:
                        df[col].fillna(df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0], inplace=True)
                    st.success("Imputed missing values")
                st.session_state.current_df = df
                st.rerun()
    
    # Select Target
    with st.expander("3️⃣ Select Target", expanded=True):
        target = st.selectbox("Target Column (Y)", df.columns.tolist())
        if st.button("Confirm", type="primary"):
            st.session_state.target_col = target
            st.session_state.feature_cols = [c for c in df.columns if c != target]
            st.success(f"Target: {target}")
            st.session_state.page = 'explore'
            st.rerun()

# ==================== EXPLORE PAGE ====================
elif st.session_state.page == 'explore':
    st.header("📊 Data Exploration")
    
    with st.expander("📦 Data Preview", expanded=True):
        st.dataframe(df.head())
    
    with st.expander("✨ EDA Report", expanded=False):
        try:
            html(generate_profile(df), height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Report failed: {e}")
    
    st.markdown("---")
    st.subheader("📈 Chart Builder")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        chart_type = st.selectbox("Chart", ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"])
        x = st.selectbox("X-axis", df.columns, index=0)
        y = st.selectbox("Y-axis", [None] + [c for c in df.columns if c != x])
        color = st.selectbox("Color", [None] + df.select_dtypes(include=['object']).columns.tolist())
        title = st.text_input("Title", f"{chart_type} Chart")
    
    with col2:
        if st.button("▶ Generate Chart", use_container_width=True, type="primary"):
            try:
                fig_map = {
                    "Scatter": px.scatter, "Line": px.line, "Bar": px.bar,
                    "Histogram": px.histogram, "Box": px.box, "Pie": px.pie
                }
                kwargs = {'data_frame': df, 'x': x, 'color': color, 'title': title}
                if y and chart_type != "Histogram": kwargs['y'] = y
                if chart_type == "Pie": kwargs = {'data_frame': df, 'names': x, 'values': y, 'title': title}
                
                fig = fig_map[chart_type](**kwargs)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart failed: {e}")

# ==================== TRAIN PAGE ====================
elif st.session_state.page == 'train':
    st.header("🧠 Train Model")
    
    if not st.session_state.target_col:
        st.warning("Complete preprocessing first")
        st.stop()
    
    target_col = st.session_state.target_col
    feature_cols = st.session_state.feature_cols
    
    # Config
    with st.expander("⚙ Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            task = st.selectbox("Task", ["Classification", "Regression"])
        with col2:
            models = {
                "Classification": ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"],
                "Regression": ["LinearRegression", "DecisionTreeRegressor", "XGBoostRegressor"]
            }
            model_choice = st.selectbox("Algorithm", models[task])
    
    # Preprocessing
    with st.expander("🧹 Preprocessing", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            scale = st.checkbox("Scale Features", value=True)
            encoding = st.selectbox("Encoding", ["One-Hot", "Label", "Ordinal"])
        with col2:
            split = st.slider("Train Split", 0.5, 0.95, 0.8)
    
    # Hyperparameters
    hyperparams = {}
    with st.expander("🛠 Hyperparameters", expanded=False):
        if "Logistic" in model_choice:
            hyperparams = {"C": st.number_input("C", 0.01, 10.0, 1.0), "max_iter": st.number_input("Max Iter", 50, 1000, 200), "solver": 'liblinear'}
        elif "RandomForest" in model_choice:
            hyperparams = {"n_estimators": st.slider("Trees", 10, 500, 100), "max_depth": st.slider("Depth", 1, 50, 10), "random_state": 42}
        elif "XGBoost" in model_choice:
            hyperparams = {"learning_rate": st.slider("LR", 0.01, 0.5, 0.1), "n_estimators": st.slider("Estimators", 50, 500, 100), "random_state": 42}
            if "Classifier" in model_choice:
                hyperparams.update({"use_label_encoder": False, "eval_metric": 'logloss'})
        elif "DecisionTree" in model_choice:
            hyperparams = {"max_depth": st.slider("Depth", 1, 50, 10), "random_state": 42}
    
    # Execution
    with st.expander("🎯 Execution", expanded=True):
        exec_target = st.radio("Target", ["Run Locally", "Run on Kaggle (Sim)"])
        
        st.markdown("---")
        st.subheader("📧 Email Notification")
        
        # Show current credentials status
        creds_valid = "@" in SMTP_SENDER_EMAIL and len(SMTP_APP_PASSWORD) > 10
        if creds_valid:
            st.success(f"✅ Email configured: {SMTP_SENDER_EMAIL} → {SMTP_RECIPIENT_EMAIL}")
        else:
            st.warning("⚠️ Email not configured. Edit credentials at top of script.")
    
    # Train Button
    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        with st.status("Training...", expanded=True) as status:
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            # Encoding
            if cat_cols:
                status.write(f"Encoding ({encoding})...")
                if encoding == "One-Hot":
                    X = pd.get_dummies(X, drop_first=True)
                elif encoding == "Label":
                    for col in cat_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                else:
                    for col in cat_cols:
                        X[col] = X[col].map({v: i for i, v in enumerate(X[col].unique())}).fillna(-1)
            
            # Target encoding
            le_target = None
            if y.dtype == 'object' and task == 'Classification':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
            
            # Scaling
            scaler = None
            if scale:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            final_cols = X.columns.tolist()
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
            
            results = {"Execution": exec_target.split()[0]}
            
            # Local Training
            if "Locally" in exec_target:
                status.write(f"Training {model_choice}...")
                model_map = {
                    "LogisticRegression": LogisticRegression,
                    "RandomForestClassifier": RandomForestClassifier,
                    "XGBoostClassifier": XGBClassifier,
                    "LinearRegression": LinearRegression,
                    "DecisionTreeRegressor": DecisionTreeRegressor,
                    "XGBoostRegressor": XGBRegressor
                }
                model = model_map[model_choice](**hyperparams)
                
                start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start
                
                st.session_state.update({
                    'model': model, 'scaler': scaler, 'le_target': le_target,
                    'feature_cols_input': feature_cols, 'final_feature_cols': final_cols,
                    'model_type': task, 'encoding_method': encoding, 'cat_cols': cat_cols,
                    'target_classes': le_target.classes_ if le_target else None
                })
                
                y_pred = model.predict(X_test)
                results["Training Time (s)"] = round(train_time, 4)
                
                # Metrics
                st.markdown("---")
                st.subheader("💡 Results")
                
                if task == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    results["Accuracy"] = round(acc, 4)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{acc*100:.2f}%")
                    
                    if len(np.unique(y_test)) > 1:
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        f1 = report.get('weighted avg', {}).get('f1-score', 0)
                        col2.metric("F1-Score", f"{f1:.4f}")
                        results["F1-Score"] = round(f1, 4)
                        
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
                        st.pyplot(fig)
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    results.update({"RMSE": round(rmse, 4), "R2": round(r2, 4)})
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{rmse:.4f}")
                    col2.metric("R²", f"{r2:.4f}")
                    col3.metric("Time", f"{train_time:.2f}s")
                
                st.session_state.local_results = results
            
            # Kaggle Sim
            else:
                status.write("Simulating Kaggle...")
                st.session_state.model = None
                time.sleep(2)
                results.update({"Accuracy": 0.89, "F1-Score": 0.87, "Training Time (s)": 1.5})
                st.session_state.kaggle_results = results
                st.info("Kaggle simulation complete")
            
            # Send Email
            if creds_valid:
                status.write("Sending email...")
                success, msg = send_email(results, model_choice, task)
                if success:
                    status.write(f"✅ {msg}")
                else:
                    status.warning(f"⚠️ {msg}")
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
    
    # Comparison
    if st.session_state.local_results or st.session_state.kaggle_results:
        st.markdown("---")
        st.subheader("📊 Comparison")
        comp = {}
        if st.session_state.local_results: comp["Local"] = st.session_state.local_results
        if st.session_state.kaggle_results: comp["Kaggle"] = st.session_state.kaggle_results
        st.dataframe(pd.DataFrame(comp).transpose())

# ==================== PREDICT PAGE ====================
elif st.session_state.page == 'predict':
    st.header("🔮 Live Prediction")
    
    if not st.session_state.model:
        st.warning("Train a local model first")
        st.stop()
    
    model = st.session_state.model
    scaler = st.session_state.scaler
    le_target = st.session_state.le_target
    feature_cols = st.session_state.feature_cols_input
    final_cols = st.session_state.final_feature_cols
    
    with st.form("pred_form"):
        input_data = {}
        cols = st.columns(3)
        
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                val = df[col].iloc[0]
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_data[col] = st.number_input(f"🔢 {col}", value=float(val))
                else:
                    opts = list(df[col].astype(str).unique())
                    input_data[col] = st.selectbox(f"🏷 {col}", opts)
        
        submit = st.form_submit_button("Submit", type="primary")
    
    if submit:
        try:
            new_df = pd.DataFrame([input_data])
            
            # Encode
            if st.session_state.encoding_method == "One-Hot":
                new_df = pd.get_dummies(new_df).reindex(columns=final_cols, fill_value=0)
            
            # Scale
            if scaler:
                new_df = pd.DataFrame(scaler.transform(new_df[final_cols]), columns=final_cols)
            
            pred = model.predict(new_df)
            
            st.markdown("### ✅ Prediction")
            if st.session_state.model_type == "Classification":
                label = le_target.inverse_transform(pred.astype(int))[0] if le_target else pred[0]
                st.success(f"🎯 Class: **{label}**")
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(new_df)[0]
                    st.dataframe(pd.DataFrame({
                        'Class': st.session_state.target_classes,
                        'Probability': proba
                    }).sort_values('Probability', ascending=False))
            else:
                st.success(f"📈 Value: **{pred[0]:.4f}**")
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")
