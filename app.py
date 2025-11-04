# AutoMLPilot Pro — colorful no‑code ML lab with playgrounds
# -----------------------------------------------------------
# - Supervised (classification/regression) + Unsupervised (clustering/anomaly/dim‑red)
# - Rich preprocessing: imputation, encoding, scaling, outliers, balancing (SMOTE*)
# - Feature engineering: recommendations from correlation + arithmetic creator
# - Per‑model help + parameter tooltips
# - Visual playgrounds for every model
# - Email results to the user from owner Gmail (configure constants below)
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
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder,
    OneHotEncoder
)
# Fix: Import MLP models
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Optional libs
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

import smtplib, ssl, json, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===================== OWNER EMAIL (sender) =====================
# WARNING: Storing credentials directly in the source code is a security risk.
# For a production application, use Streamlit Secrets or environment variables.
OWNER_GMAIL = "rhreddy4748@gmail.com"           # Gmail address (sender)
OWNER_APP_PASSWORD = "zujpoggswfcpxwjs"        # Gmail App password
OWNER_ALIAS = "noreply@automlpilot.com"         # Displayed From address
SENDER_NAME = "AutoMLPilot"

# ===================== PAGE CONFIG & THEME =====================
st.set_page_config(page_title="AutoMLPilot Pro", page_icon="✨", layout="wide")

THEME = """
<style>
  :root { --bg1: #ffe5f0; --bg2: #e6e9ff; --card: rgba(255,255,255,0.65); --border: rgba(255,255,255,0.35); }
  .main { background: radial-gradient(1200px 600px at 10% 10%, var(--bg1), transparent),
                          radial-gradient(900px 500px at 90% 20%, var(--bg2), transparent),
                          linear-gradient(120deg,#f9fafb,#eef2ff); }
  .block-container { padding: 1rem 2rem; }
  h1,h2,h3,h4 { color:#0f172a; }
  .topbar { position: sticky; top:0; z-index:1000; backdrop-filter: blur(12px);
            background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0.6));
            border-bottom:1px solid var(--border); padding:.6rem 1rem; border-radius: 14px; }
  .chip { display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe; font-size:.8rem; }
  .card { background: var(--card); border:1px solid var(--border); border-radius:20px; box-shadow: 0 12px 35px rgba(31,41,55,.12); padding:16px; }
  .metric { background: rgba(255,255,255,0.75); border-left:4px solid #8b5cf6; border-radius:14px; padding:12px; }
  .pillbtn button { border-radius:999px !important; }
  .small { color:#475569; font-size:.85rem; }
  .tooltip { color:#6b7280; font-size:.85rem; }
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)

# ===================== HELPERS =====================
def send_results_email(to_email: str, subject: str, results: dict, extra_html: str = ""):
    """Safely send email with results."""
    if not to_email or "@" not in to_email:
        st.warning("Please enter a valid email.")
        return False
    # Check for credentials before attempting
    if not OWNER_GMAIL or not OWNER_APP_PASSWORD:
        st.error("Email credentials are not configured.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{OWNER_ALIAS}>"
        msg["To"] = to_email
        html_body = f"""
        <html><body style='font-family:Inter,system-ui'>
          <h2 style='color:#7c3aed;margin:0'>Your AutoMLPilot Results</h2>
          {extra_html}
          <pre style='background:#0b1220;color:#e5e7eb;border:1px solid #1f2937;padding:12px;border-radius:10px'>
{json.dumps(results, indent=2)}
          </pre>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
            s.login(OWNER_GMAIL, OWNER_APP_PASSWORD)
            # Use to_email for recipient
            s.sendmail(OWNER_GMAIL, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        # Log the specific error (e.g., authentication) without leaking sensitive data
        return False

@st.cache_data(show_spinner=False)
def profile_html(df: pd.DataFrame) -> str:
    """Generate ydata-profiling report HTML."""
    if not YDATA_OK:
        return "<p class='small'>Install **ydata-profiling** to enable full EDA.</p>"
    try:
        pr = ProfileReport(df, explorative=True, minimal=True)
        return pr.to_html()
    except Exception as e:
        st.error(f"Failed to generate profile report: {e}")
        return "<p class='small'>Error generating EDA report.</p>"

def correlation_recommendations(df: pd.DataFrame, thresh=0.85):
    """Suggest highly correlated numeric feature pairs for feature engineering."""
    num = df.select_dtypes(include=[np.number])
    recs = []
    if num.shape[1] < 2:
        return recs
    try:
        corr = num.corr()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j <= i:
                    continue
                v = corr.iloc[i, j]
                # Check for NaN correlations (can happen with near-constant columns)
                if not np.isnan(v) and abs(v) >= thresh:
                    recs.append((c1, c2, float(v)))
        return sorted(recs, key=lambda x: -abs(x[2]))[:20]
    except Exception as e:
        st.warning(f"Error generating correlation recommendations: {e}")
        return []

# ===================== SESSION =====================
if "S" not in st.session_state:
    st.session_state.S = {
        "page": "dashboard",
        "df": None,
        "target": None,
        "task": "Classification",
        "user_email": "",
        "corr_pairs": [],
        "features_created": [],
        # Preprocessing settings to be applied LATER inside the pipeline
        "imputer_strategy": "None",
        "outlier_remove": False,
        "encoding": "None", # One-Hot or Label
        "scaler_name": "None",
        "variance_threshold": 0.0,
        "smote_enabled": False,
        # training artifacts
        "model": None,
        "preprocessor_pipeline": None, # Store the fitted ColumnTransformer/Pipeline
        "final_cols": None,
        "results": {},
        # unsupervised
        "unsup_labels": None,
    }
S = st.session_state.S

# ===================== TOP BAR =====================
with st.container():
    st.markdown("""
    <div class='topbar'>
      <div style='display:flex;justify-content:space-between;align-items:center'>
        <div style='display:flex;gap:.6rem;align-items:center'>
          <span>🌈</span>
          <strong>AutoMLPilot Pro</strong>
          <span class='chip'>No‑Code AI Lab</span>
        </div>
        <div style='display:flex;gap:8px;align-items:center'>
          <span class='chip'>Playground</span>
          <span class='chip'>EDA</span>
          <span class='chip'>Email Reports</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ===================== SIDEBAR NAV =====================
with st.sidebar:
    st.subheader("Navigation")
    pg = st.radio("", ["dashboard","preprocess","train","playground","unsupervised","results","help"],
                      format_func=lambda x: {
                          "dashboard":"📁 Dashboard",
                          "preprocess":"🧹 Preprocess Config", # Renamed for clarity
                          "train":"🧠 Train (Supervised)",
                          "playground":"🎨 Playground (Supervised)",
                          "unsupervised":"🧩 Unsupervised",
                          "results":"📊 Results",
                          "help":"❓ Help"
                      }[x])
    S["page"] = pg

# ===================== DASHBOARD =====================
if S["page"] == "dashboard":
    st.title("Dashboard")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown("### Upload Dataset")
        up = st.file_uploader("CSV only", type=["csv"]) 
        if up is not None:
            try:
                # Use a cached version if already loaded to avoid re-reading on every rerun
                if S["df"] is None or S["df_name"] != up.name:
                    df = pd.read_csv(up)
                    # Reset all settings when a new dataset is loaded
                    S.update({
                        "df": df,
                        "df_name": up.name,
                        "target": None,
                        "model": None,
                        "preprocessor_pipeline": None,
                        "results": {},
                        "features_created": []
                    })
                    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        if S["df"] is not None:
            st.markdown("### Preview")
            st.dataframe(S["df"].head())
    with c2:
        st.markdown("### Email for Reports")
        S["user_email"] = st.text_input("User email (recipient)", value=S["user_email"], placeholder="you@example.com")
        st.info("Results will be sent from AutoMLPilot (owner Gmail).")
    st.markdown("---")
    st.markdown("### One‑click EDA Report")
    if S["df"] is not None:
        if st.button("Generate EDA", type="primary"):
            with st.spinner("Generating detailed EDA report..."):
                report_html = profile_html(S["df"])
                html(report_html, height=600, scrolling=True)

# ===================== PREPROCESS CONFIG =====================
# Now this page *configures* the preprocessing steps, which will be executed as a pipeline in the 'train' section to prevent data leakage.
elif S["page"] == "preprocess":
    st.title("Preprocessing Configuration")
    if S["df"] is None:
        st.info("Upload a dataset first.")
        st.stop()
    
    st.warning("⚠️ **Data Leakage Alert:** Preprocessing parameters (Imputation, Scaling, Encoding) are only **fitted** on the **training data** in the 'Train' tab to prevent data leakage. The configurations below set the steps for that pipeline.")

    df = S["df"].copy()

    with st.expander("1) Missing Values Strategy", expanded=True):
        S["imputer_strategy"] = st.selectbox("Imputation Strategy (Numeric/Categorical)", 
                                            ["None","Mean","Median","Most_frequent"],
                                            index=["None","Mean","Median","Most_frequent"].index(S["imputer_strategy"]))
        st.caption("Applied within the training pipeline to prevent leakage.")
    
    with st.expander("2) Outlier Removal (Manual, on original data)"):
        # Outlier removal is often done manually/exploratorily before train/test split.
        S["outlier_remove"] = st.checkbox("Apply IQR Outlier Removal on Full Dataset (Numeric only)", value=S["outlier_remove"])
        st.caption("This is an **exploratory step** and is applied immediately to the raw data *before* train/test split. Use with caution.")
    
    with st.expander("3) Encoding Strategy", expanded=True):
        S["encoding"] = st.selectbox("Categorical Encoding", ["None","One-Hot","Label"],
                                     index=["None","One-Hot","Label"].index(S["encoding"]))
        st.caption("One-Hot is recommended for linear models; Label is fast but implies an order (risky). Applied in the training pipeline.")

    with st.expander("4) Scaling/Normalization Strategy", expanded=False):
        S["scaler_name"] = st.selectbox("Scaler/Normalizer (Numeric only)", 
                                        ["None","Standard","MinMax","Robust","Normalize"],
                                        index=["None","Standard","MinMax","Robust","Normalize"].index(S["scaler_name"]))
        st.caption("Applied within the training pipeline to prevent leakage.")

    with st.expander("5) Feature Selection (VarianceThreshold)", expanded=False):
        remove_var = st.checkbox("Remove features with low variance (Numeric only)", S["variance_threshold"] > 0.0)
        S["variance_threshold"] = st.slider("Variance threshold", 0.0, 0.2, S["variance_threshold"] if remove_var else 0.0, 0.01)
        st.caption("Features with variance lower than the threshold will be dropped (Applied in the pipeline).")
    
    with st.expander("6) Balancing (SMOTE)", expanded=False):
        S["smote_enabled"] = st.checkbox("Enable SMOTE for Classification", value=S["smote_enabled"])
        if not IMB_OK:
            st.caption("Install **imbalanced-learn** to enable SMOTE.")
        else:
            st.caption("SMOTE is only applied to the **training data** when a classification task is selected.")

    # --- IMMEDIATE OUTLIER REMOVAL EXECUTION ---
    if S["outlier_remove"]:
        try:
            df_after_outliers = S["df"].copy()
            num_cols = df_after_outliers.select_dtypes(include=[np.number]).columns
            before = len(df_after_outliers)
            for col in num_cols:
                q1, q3 = df_after_outliers[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                df_after_outliers = df_after_outliers[(df_after_outliers[col] >= lower) & (df_after_outliers[col] <= upper)]
            
            if before > len(df_after_outliers):
                S["df"] = df_after_outliers
                st.success(f"Removed **{before - len(S['df'])}** rows as outliers from the dataset.")
            S["outlier_remove"] = True # Keep the state even after execution
        except Exception as e:
            st.error(f"Error during outlier removal: {e}")
            S["outlier_remove"] = False # Reset if failure
    else:
        st.caption("Outlier removal is disabled, full dataset retained.")

    st.success("Preprocessing configuration updated. Move to Train.")

# ===================== TRAIN (SUPERVISED) =====================
elif S["page"] == "train":
    st.title("Supervised Training")
    if S["df"] is None:
        st.info("Upload data first.")
        st.stop()
    df = S["df"].copy()

    # Target selection
    col_list = [None] + df.columns.tolist()
    # Find the correct index for the current target, safely defaulting to 0
    default_index = col_list.index(S["target"]) if S["target"] in col_list else 0
    S["target"] = st.selectbox("Target (y)", col_list, index=default_index)
    
    if S["target"] is None:
        st.info("Select a target feature to proceed with supervised training.")
        st.stop()
    
    # Task definition
    # Infer task based on unique values in target column
    unique_vals = df[S["target"]].nunique()
    if df[S["target"]].dtype in [np.number] and unique_vals > 20: # Heuristic for regression
        inferred_task = "Regression"
    else:
        inferred_task = "Classification" # Binary or multi-class
        
    S["task"] = st.radio("Task", ["Classification","Regression"], horizontal=True, index=["Classification","Regression"].index(inferred_task))
    task = S["task"]
    
    # Feature engineering suggestions
    with st.expander("✨ Feature Engineering", expanded=False):
        st.subheader("Correlation Recommendations")
        recs = correlation_recommendations(df.drop(columns=[S["target"]], errors='ignore'))
        if recs:
            st.dataframe(pd.DataFrame(recs, columns=["Feature A","Feature B","Correlation"]))
        else:
            st.caption("No strong correlation pairs found in numeric features (or less than 2 numeric features).")
            
        st.markdown("---")
        st.subheader("Arithmetic Feature Creator")
        cols = [c for c in df.columns if c != S["target"] and df[c].dtype in [np.number]]
        if not cols:
            st.caption("No numeric columns available for arithmetic feature creation.")
        else:
            new_name = st.text_input("New feature name", placeholder="e.g., a_div_b")
            c1, c2 = st.columns(2)
            with c1:
                f1 = st.selectbox("Feature 1", cols, key="fe_f1")
                op = st.selectbox("Operation", ["+","-","*","/"], key="fe_op")
            with c2:
                f2 = st.selectbox("Feature 2", [c for c in cols if c != f1], key="fe_f2")
                
            if st.button("Create Feature"):
                if not new_name:
                    st.error("Please enter a name for the new feature.")
                else:
                    try:
                        # Defensive copy to avoid Streamlit caching issues with in-place modification
                        temp_df = S["df"].copy()
                        if op=="/":
                            # Division by zero handling - replace 0 with a very small number or NaN
                            if (temp_df[f2] == 0).any():
                                st.warning(f"Feature '{f2}' contains zeros. Replacing them with NaN for safe division.")
                                temp_df[new_name] = temp_df[f1] / temp_df[f2].replace(0, np.nan)
                                # Imputation later in the pipeline will handle the resulting NaNs
                            else:
                                temp_df[new_name] = temp_df[f1] / temp_df[f2]
                        elif op=="+": temp_df[new_name] = temp_df[f1] + temp_df[f2]
                        elif op=="-": temp_df[new_name] = temp_df[f1] - temp_df[f2]
                        elif op=="*": temp_df[new_name] = temp_df[f1] * temp_df[f2]
                        
                        S["df"] = temp_df # Update the session state DataFrame
                        S["features_created"].append(new_name)
                        st.success(f"Feature '**{new_name}**' created and added to the dataset.")
                        # Rerun to update selectboxes with the new feature
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Feature creation failed: {e}")

    # Model zoo definition (No change needed here, just cleaner placement)
    st.markdown("### Choose Model & Params")
    # ... [Model zoo dictionary definitions remain the same] ...
    models_cls = {
        "LogisticRegression": (LogisticRegression, {
            "C": (st.slider, {"label":"C (inverse reg strength)", "min_value":0.01, "max_value":10.0, "value":1.0}),
            "max_iter": (st.slider, {"label":"Max Iter (epochs)", "min_value":50, "max_value":2000, "value":300}),
            "solver": (st.selectbox, {"label":"Solver", "options":["lbfgs","liblinear"], "index":1}),
        }),
        "RandomForestClassifier": (RandomForestClassifier, {
            "n_estimators": (st.slider, {"label":"Trees", "min_value":50, "max_value":800, "value":200}),
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":60, "value":None}),
            "random_state": (st.number_input, {"label":"Random State", "value":42}),
        }),
        "SVC": (SVC, {
            "C": (st.slider, {"label":"C", "min_value":0.01, "max_value":10.0, "value":1.0}),
            "kernel": (st.selectbox, {"label":"Kernel", "options":["rbf","linear","poly"], "index":0}),
            "probability": (st.checkbox, {"label":"Enable probability", "value":True}),
        }),
        "KNeighborsClassifier": (KNeighborsClassifier, {
            "n_neighbors": (st.slider, {"label":"Neighbors", "min_value":1, "max_value":50, "value":5}),
        }),
        "GaussianNB": (GaussianNB, {}),
        "DecisionTreeClassifier": (DecisionTreeClassifier, {
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":60, "value":10}),
        }),
        "GradientBoostingClassifier": (GradientBoostingClassifier, {
            "n_estimators": (st.slider, {"label":"Estimators", "min_value":50, "max_value":600, "value":200}),
            "learning_rate": (st.slider, {"label":"Learning rate", "min_value":0.01, "max_value":0.5, "value":0.1}),
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":8, "value":3}),
        }),
        "MLPClassifier": (MLPClassifier, {
            "hidden_layer_sizes": (st.slider, {"label":"Hidden units", "min_value":8, "max_value":512, "value":128}),
            "learning_rate_init": (st.slider, {"label":"LR", "min_value":1e-4, "max_value":1e-1, "value":1e-3, "format":"%.4f"}),
            "max_iter": (st.slider, {"label":"Epochs", "min_value":50, "max_value":2000, "value":300}),
            "early_stopping": (st.checkbox, {"label":"Early stopping", "value":True}),
        }),
    }
    if XGB_OK:
        models_cls["XGBClassifier"] = (XGBClassifier, {
            "n_estimators": (st.slider, {"label":"Estimators", "min_value":50, "max_value":1000, "value":300}),
            "learning_rate": (st.slider, {"label":"Learning rate", "min_value":0.01, "max_value":0.5, "value":0.1}),
            "subsample": (st.slider, {"label":"Subsample", "min_value":0.5, "max_value":1.0, "value":0.9}),
            "colsample_bytree": (st.slider, {"label":"Colsample by tree", "min_value":0.5, "max_value":1.0, "value":0.9}),
            "eval_metric": (st.selectbox, {"label":"Eval metric", "options":["logloss","auc"], "index":0}),
        })

    models_reg = {
        "LinearRegression": (LinearRegression, {}),
        "Ridge": (Ridge, {"alpha": (st.slider, {"label":"alpha", "min_value":0.0, "max_value":10.0, "value":1.0})}),
        "Lasso": (Lasso, {"alpha": (st.slider, {"label":"alpha", "min_value":0.0, "max_value":10.0, "value":1.0})}),
        "ElasticNet": (ElasticNet, {"alpha": (st.slider, {"label":"alpha", "min_value":0.0, "max_value":10.0, "value":1.0}),
                                     "l1_ratio": (st.slider, {"label":"l1_ratio", "min_value":0.0, "max_value":1.0, "value":0.5})}),
        "RandomForestRegressor": (RandomForestRegressor, {
            "n_estimators": (st.slider, {"label":"Trees", "min_value":50, "max_value":800, "value":200}),
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":60, "value":None}),
            "random_state": (st.number_input, {"label":"Random State", "value":42}),
        }),
        "SVR": (SVR, {"C": (st.slider, {"label":"C", "min_value":0.01, "max_value":10.0, "value":1.0}),
                        "kernel": (st.selectbox, {"label":"Kernel", "options":["rbf","linear","poly"], "index":0})}),
        "DecisionTreeRegressor": (DecisionTreeRegressor, {
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":60, "value":10}),
        }),
        "GradientBoostingRegressor": (GradientBoostingRegressor, {
            "n_estimators": (st.slider, {"label":"Estimators", "min_value":50, "max_value":600, "value":200}),
            "learning_rate": (st.slider, {"label":"Learning rate", "min_value":0.01, "max_value":0.5, "value":0.1}),
            "max_depth": (st.slider, {"label":"Max depth", "min_value":1, "max_value":8, "value":3}),
        }),
        "MLPRegressor": (MLPRegressor, {
            "hidden_layer_sizes": (st.slider, {"label":"Hidden units", "min_value":8, "max_value":512, "value":128}),
            "learning_rate_init": (st.slider, {"label":"LR", "min_value":1e-4, "max_value":1e-1, "value":1e-3, "format":"%.4f"}),
            "max_iter": (st.slider, {"label":"Epochs", "min_value":50, "max_value":2000, "value":300}),
            "early_stopping": (st.checkbox, {"label":"Early stopping", "value":True}),
        }),
    }
    if XGB_OK:
        models_reg["XGBRegressor"] = (XGBRegressor, {
            "n_estimators": (st.slider, {"label":"Estimators", "min_value":50, "max_value":1000, "value":300}),
            "learning_rate": (st.slider, {"label":"Learning rate", "min_value":0.01, "max_value":0.5, "value":0.1}),
            "subsample": (st.slider, {"label":"Subsample", "min_value":0.5, "max_value":1.0, "value":0.9}),
            "colsample_bytree": (st.slider, {"label":"Colsample by tree", "min_value":0.5, "max_value":1.0, "value":0.9}),
        })
    # ... [End of Model zoo dictionary definitions] ...

    zoo = models_cls if task=="Classification" else models_reg
    mname = st.selectbox("Model", list(zoo.keys()))

    # build kwargs via widgets with inline help
    params = {}
    with st.expander("Hyperparameters", expanded=True):
        for pname, (wfun, kwargs) in zoo[mname][1].items():
            label = kwargs.pop("label", pname)
            # Safely handle format for sliders/number inputs
            format_str = kwargs.pop("format", "%.2f") if wfun in [st.slider, st.number_input] else None
            
            if format_str:
                widget_val = wfun(label, format=format_str, **kwargs)
            else:
                widget_val = wfun(label, **kwargs)

            # adapt for MLP hidden_layer_sizes (int -> tuple)
            if pname == "hidden_layer_sizes":
                # Ensure it's a tuple of integers for the MLP model
                widget_val = (int(widget_val),)
            
            # Handle max_depth=None for tree models
            if pname == "max_depth" and widget_val == 60: # Using 60 as proxy for 'None' max value
                if wfun == st.slider: # Only apply if it came from the slider
                    widget_val = None
            
            params[pname] = widget_val
            st.caption(f"ℹ️ **{pname}**")

    # Train button logic
    if st.button("🚀 Train", type="primary"):
        try:
            # 1. Separate features (X) and target (y)
            X = df.drop(columns=[S["target"]], errors='ignore').copy()
            y = df[S["target"]].copy()

            # 2. Handle object/category types in X and y *before* preprocessor pipeline
            categorical_cols_X = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols_X = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Convert target to numeric if classification and object
            if task == "Classification" and y.dtype in ['object', 'category']:
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))
                S["label_encoders_y"] = le_y # Store for later use
            else:
                S["label_encoders_y"] = None

            # 3. Train-Test Split (CRITICAL for data leakage prevention)
            # Check for sufficient data
            if len(X) < 20:
                st.error("Dataset is too small for a 80/20 train-test split. Need at least 20 rows.")
                st.stop()

            stratify_y = y if task == "Classification" and unique_vals > 1 else None
            # Handle case where stratify has too few samples per class
            if stratify_y is not None:
                small_classes = y.value_counts()[y.value_counts() < 2].index.tolist()
                if small_classes:
                    st.warning(f"Classification target has classes with < 2 samples: {small_classes}. Disabling stratification.")
                    stratify_y = None

            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )
            
            # 4. Build Preprocessor Pipeline based on configuration (NO FITTING YET)
            numeric_steps = []
            
            # Imputation
            if S["imputer_strategy"] != "None":
                imputer_strat = S["imputer_strategy"].lower()
                if imputer_strat == "most_frequent":
                    numeric_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
                else:
                    numeric_steps.append(('imputer', SimpleImputer(strategy=imputer_strat)))

            # Variance Threshold (always needs to be before scaling if applied)
            if S["variance_threshold"] > 0.0:
                 numeric_steps.append(('var_thresh', VarianceThreshold(threshold=S["variance_threshold"])))
            
            # Scaling
            if S["scaler_name"] != "None":
                scaler_map = {
                    "Standard": StandardScaler(), "MinMax": MinMaxScaler(), 
                    "Robust": RobustScaler(), "Normalize": Normalizer()
                }
                numeric_steps.append(('scaler', scaler_map[S["scaler_name"]]))

            # Encoding
            if S["encoding"] == "One-Hot":
                # Use OneHotEncoder for categorical columns
                categorical_transformer = Pipeline(steps=[
                    ('imputer_cat', SimpleImputer(strategy='most_frequent')), # Impute categories
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=numeric_steps), numeric_cols_X),
                        ('cat', categorical_transformer, categorical_cols_X)
                    ],
                    remainder='passthrough' # Keep other columns (e.g., if you missed 'object' but not 'category')
                )
            elif S["encoding"] == "Label":
                 # LabelEncoder is complex in ColumnTransformer, easier to do it manually *before* the CT for a simple flow
                 # Note: Label encoding is often only suitable for tree models, as it introduces ordinality.
                for c in categorical_cols_X:
                    le = LabelEncoder()
                    X_train_df[c] = le.fit_transform(X_train_df[c].astype(str))
                    # Use fitted encoder to transform test data (handling unseen labels with .transform(labels) if possible)
                    try:
                        X_test_df[c] = le.transform(X_test_df[c].astype(str))
                    except ValueError:
                        st.warning(f"Unseen labels in test set for column '{c}'. Mapping unseen labels to -1.")
                        # Handle unseen labels by mapping them to -1 (or similar)
                        test_labels = X_test_df[c].astype(str)
                        known_labels = set(le.classes_)
                        X_test_df[c] = test_labels.apply(lambda x: le.transform([x])[0] if x in known_labels else -1)

                # Now that categorical columns are numeric (labeled), treat them as numeric for scaling/imputation
                numeric_cols_X = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Rebuild preprocessor with only numeric steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=numeric_steps), numeric_cols_X)
                    ],
                    remainder='passthrough'
                )
            else: # No Encoding
                # Only apply numeric steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=numeric_steps), numeric_cols_X)
                    ],
                    remainder='passthrough'
                )

            # 5. Full Pipeline: Preprocessor + Model
            Model = zoo[mname][0]
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier_regressor', Model(**params))
            ])

            # 6. SMOTE (Applied to training data only, after preprocessing)
            if task == "Classification" and S["smote_enabled"] and IMB_OK:
                # Need to transform X_train first to handle imputer/scaler
                X_train_processed = model_pipeline['preprocessor'].fit_transform(X_train_df)
                
                # SMOTE application
                sm = SMOTE(random_state=42)
                X_train_res, y_train_res = sm.fit_resample(X_train_processed, y_train)
                
                # Fit the final model only on the resampled, processed data
                t0 = time.time()
                model_pipeline['classifier_regressor'].fit(X_train_res, y_train_res)
                t = time.time() - t0
                
                # X_test needs to be transformed by the *fitted* preprocessor
                X_test_processed = model_pipeline['preprocessor'].transform(X_test_df)
                y_pred = model_pipeline['classifier_regressor'].predict(X_test_processed)
                
                st.info("SMOTE applied to training data.")

            else:
                # 7. Fit Pipeline (Preprocessor and Model)
                t0 = time.time()
                # Fit the entire pipeline on the original training data
                model_pipeline.fit(X_train_df, y_train)
                t = time.time() - t0

                # 8. Predict on the Test Data (Test data is transformed by the *fitted* preprocessor)
                y_pred = model_pipeline.predict(X_test_df)
                
                # Check for NaNs/Infs in prediction
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    st.error("Prediction generated NaN or Inf values. Training may have failed.")
                    st.stop()


            # 9. Store artifacts and results
            S["model"] = model_pipeline
            S["preprocessor_pipeline"] = model_pipeline['preprocessor']
            # Reconstruct the final column list after preprocessing (for visualization/playground)
            # This is complex, so we'll store the *original* X columns for the Playground PCA step.
            S["final_cols"] = list(X.columns)

            st.success("Training successful. Evaluating results...")

            # 10. Evaluation and Visualization
            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                # Handle binary vs multiclass f1
                average_f1 = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
                f1 = f1_score(y_test, y_pred, average=average_f1, zero_division=0)
                
                st.markdown(f"<div class='metric'><b>Accuracy</b> {acc*100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric'><b>F1 ({average_f1})</b> {f1:.4f}</div>", unsafe_allow_html=True)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(); 
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                S["results"] = {"task":task, "model":mname, "accuracy":acc, "f1":f1, "train_time":round(t,3), "params": params}
            else: # Regression
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
                
                st.markdown(f"<div class='metric'><b>RMSE</b> {rmse:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric'><b>R²</b> {r2:.4f}</div>", unsafe_allow_html=True)
                
                # Plot Actual vs Predicted (moved to Playground, just store results here)
                
                S["results"] = {"task":task, "model":mname, "rmse":rmse, "r2":r2, "train_time":round(t,3), "params": params}
            
            st.success("Training and Evaluation complete. See Results or try Playground.")

        except Exception as e:
            st.error(f"Training failed: {e}")
            # Ensure model and results are reset on catastrophic failure
            S["model"] = None
            S["results"] = {}


# ===================== PLAYGROUND (supervised) =====================
elif S["page"] == "playground":
    st.title("Model Playground (Supervised)")
    if S["df"] is None or S["target"] is None or S["model"] is None:
        st.info("Train a supervised model first.")
        st.stop()
        
    df = S["df"]; target = S["target"]
    # Get original features for test split
    X_original = df.drop(columns=[target], errors='ignore').copy()
    y_original = df[target].copy()
    
    task = S["results"].get("task","Classification")
    model_pipeline = S["model"]
    preprocessor = S["preprocessor_pipeline"]

    # 1. Prepare Data for Playground (Apply *fitted* preprocessor to test set)
    # The split is done here to ensure the Playground visualizations are on *unseen* data
    stratify_y = y_original if task == "Classification" and y_original.nunique() > 1 else None
    if stratify_y is not None:
        small_classes = y_original.value_counts()[y_original.value_counts() < 2].index.tolist()
        if small_classes:
            st.warning(f"Classes with < 2 samples in target ({small_classes}). Disabling stratification for split.")
            stratify_y = None
            
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42, stratify=stratify_y
    )

    # Convert y_test to numeric if classification and it was object
    if task == "Classification" and y_test_orig.dtype in ['object', 'category'] and S.get("label_encoders_y"):
        try:
            y_test = S["label_encoders_y"].transform(y_test_orig.astype(str))
        except ValueError:
            st.warning("Unseen labels in test set for target. Cannot visualize accurately.")
            y_test = y_test_orig.astype(str) # Fallback to original labels
    else:
        y_test = y_test_orig

    # Apply the fitted preprocessor from the training phase to the test data
    try:
        X_test_processed = preprocessor.transform(X_test_orig)
    except Exception as e:
        st.error(f"Failed to transform test data with fitted preprocessor: {e}. Cannot run playground.")
        st.stop()

    # 2. Reduce Processed Test Data to 2D for Visualization
    st.markdown("### Data Visualization (PCA on Test Set)")
    try:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X_test_processed) # Fit PCA *only* on test set for visualization
        st.caption(f"Visualizing **{len(X2)}** processed test samples in 2D using PCA.")
        
        # Ensure y_test is a string list for proper color mapping in Plotly
        y_test_str = [str(v) for v in y_test]
        
        fig = px.scatter(x=X2[:,0], y=X2[:,1], color=y_test_str, 
                         labels={'x':'PC1','y':'PC2', 'color':target}, 
                         title="Test Data (PCA Projection)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate PCA plot: {e}")

    # 3. Model-Specific Visualization
    st.markdown("### Model Insight")
    
    if task == "Classification":
        st.info("The Decision Boundary visualization requires training a new model *only* on the 2D PCA data, which may differ from the original model's performance.")
        
        # Train a new model on PCA space for boundary visualization (X_train_processed is from earlier)
        # Note: Need the *processed* training data from the main split.
        try:
            # Rerun the original split and transformation on the training set
            X_train_orig, _, y_train, _ = train_test_split(
                X_original, y_original, test_size=0.2, random_state=42, stratify=stratify_y
            )
            X_train_processed = preprocessor.transform(X_train_orig)
            
            # Convert y_train to numeric if classification and it was object
            if S.get("label_encoders_y"):
                y_train = S["label_encoders_y"].transform(y_train_orig.astype(str))
                
            pca_vis = PCA(n_components=2)
            X2_train = pca_vis.fit_transform(X_train_processed)
            X2_test = pca_vis.transform(X_test_processed) # Apply the same PCA to test

            # Get the underlying model from the pipeline
            model_core = model_pipeline['classifier_regressor']
            
            # Clone model params safely
            clone_params = model_core.get_params(deep=False)
            # Remove keys that might not be applicable or cause issues when cloning (e.g., random_state)
            if 'random_state' in clone_params: del clone_params['random_state']
            if 'max_iter' in clone_params: clone_params['max_iter'] = 300 # Lower iter for faster viz training
            
            clone = type(model_core)(**clone_params)
            clone.fit(X2_train, y_train)
            
            # Generate the grid for the decision boundary
            x_min, x_max = X2[:,0].min()-0.5, X2[:,0].max()+0.5
            y_min, y_max = X2[:,1].min()-0.5, X2[:,1].max()+0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            Z = clone.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            # Map numeric labels back to strings if possible for legend clarity
            class_labels = [str(v) for v in np.unique(y_test)]
            if S.get("label_encoders_y"):
                try:
                    class_labels = S["label_encoders_y"].inverse_transform(np.unique(y_test))
                except Exception:
                    pass # Keep numeric labels if inverse_transform fails
                    
            fig2 = go.Figure()
            # Boundary Contour
            fig2.add_trace(go.Contour(
                x=np.linspace(x_min,x_max,100), y=np.linspace(y_min,y_max,100), z=Z,
                showscale=False, contours_coloring='heatmap', opacity=0.3,
                name="Decision Boundary"
            ))
            # Test Data Points
            fig2.add_trace(go.Scatter(
                x=X2_test[:,0], y=X2_test[:,1], mode='markers',
                marker=dict(size=6, color=y_test), text=[str(v) for v in y_test_str],
                name="Test Data Points"
            ))
            fig2.update_layout(title="Decision Boundary (PCA space)", 
                               xaxis_title="PC1", yaxis_title="PC2")
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Decision Boundary visualization not available for this model/data combination: {e}")
            
    else: # Regression: Actual vs Predicted
        st.markdown("### Actual vs Predicted (Test Set)")
        try:
            y_pred = model_pipeline.predict(X_test_orig)
            
            # Check for NaNs/Infs in test prediction
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                st.error("Model predicted NaN/Inf values on test set. Cannot plot.")
                st.stop()
                
            fig3 = px.scatter(x=y_test, y=y_pred, 
                             labels={'x':'Actual','y':'Predicted'}, 
                             title=f'{model_pipeline["classifier_regressor"].__class__.__name__}: Actual vs Predicted')
            
            # Add y=x line
            min_val = float(np.min(y_test))
            max_val = float(np.max(y_test))
            fig3.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                           line=dict(color='red', width=2, dash='dash'))
            
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate Actual vs Predicted plot: {e}")


# ===================== UNSUPERVISED =====================
elif S["page"] == "unsupervised":
    st.title("Unsupervised Lab")
    if S["df"] is None:
        st.info("Upload a dataset first.")
        st.stop()
        
    df = S["df"].copy()
    
    # 1. Prepare Data for Unsupervised Learning
    st.caption("All non-numeric features are label encoded for unsupervised algorithms.")
    try:
        for c in df.select_dtypes(include=["object","category"]).columns:
            le = LabelEncoder()
            # Handle NaNs by converting to string, then imputing or dropping later
            df[c] = le.fit_transform(df[c].astype(str).fillna('NA_MISSING')) 
        
        # Impute NaNs in numeric columns (simple mean imputation for stability)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if df[num_cols].isnull().any().any():
            imputer = SimpleImputer(strategy='mean')
            df[num_cols] = imputer.fit_transform(df[num_cols])
            st.info("Numeric NaNs imputed with mean for stability.")
            
        # Standard Scale the data for distance-based algorithms
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    except Exception as e:
        st.error(f"Data preparation for unsupervised learning failed: {e}")
        st.stop()

    # 2. Algorithm Selection and Parameter Tuning
    algo = st.selectbox("Algorithm", ["KMeans","DBSCAN","Agglomerative","GaussianMixture","IsolationForest","PCA (2D)"])
    
    labels = None

    if algo=="KMeans":
        k = st.slider("k (clusters)", 2, 15, 4)
        try:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(df_scaled)
            sil = silhouette_score(df_scaled, labels) if len(np.unique(labels))>1 else np.nan
            st.caption(f"Silhouette Score (higher is better): **{sil:.4f}**")
        except Exception as e:
            st.error(f"KMeans failed: {e}")
            
    elif algo=="DBSCAN":
        eps = st.slider("eps (max distance)", 0.1, 5.0, 0.8) # Adjusted max for scaled data
        min_s = st.slider("min_samples", 3, 50, 5)
        try:
            model = DBSCAN(eps=eps, min_samples=min_s)
            labels = model.fit_predict(df_scaled)
            st.caption(f"Number of clusters found: **{len(np.unique(labels)) - (1 if -1 in labels else 0)}** (Cluster **-1** is noise)")
        except Exception as e:
            st.error(f"DBSCAN failed: {e}")
            
    elif algo=="Agglomerative":
        k = st.slider("clusters", 2, 15, 5)
        try:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(df_scaled)
            sil = silhouette_score(df_scaled, labels) if len(np.unique(labels))>1 else np.nan
            st.caption(f"Silhouette Score: **{sil:.4f}**")
        except Exception as e:
            st.error(f"Agglomerative Clustering failed: {e}")
            
    elif algo=="GaussianMixture":
        k = st.slider("components", 2, 15, 4)
        try:
            model = GaussianMixture(n_components=k, random_state=42)
            labels = model.fit_predict(df_scaled)
            # BIC can be used for model selection, but silhouette is better for visualization context
            st.caption(f"Converged: **{model.converged_}**")
        except Exception as e:
            st.error(f"Gaussian Mixture failed: {e}")
            
    elif algo=="IsolationForest":
        c = st.slider("contamination (anomaly fraction)", 0.01, 0.4, 0.05, format="%.2f")
        try:
            model = IsolationForest(contamination=c, random_state=42)
            # fit_predict returns 1 for inliers, -1 for outliers
            labels = model.fit_predict(df_scaled)
            st.caption(f"Anomaly Count (label **-1**): **{np.sum(labels == -1)}**")
        except Exception as e:
            st.error(f"Isolation Forest failed: {e}")
            
    elif algo=="PCA (2D)":  # PCA (2D) is just for visualization/dim-red, no labels to store
        st.markdown("### PCA Projection")
        try:
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(df_scaled)
            st.plotly_chart(px.scatter(x=X2[:,0], y=X2[:,1], 
                                       title="PCA 2D Projection (Scaled Data)",
                                       labels={'x':f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', 
                                               'y':f'PC2 ({pca.explained_variance_ratio_[1]:.2f})'}), 
                            use_container_width=True)
            st.stop()
        except Exception as e:
            st.error(f"PCA visualization failed: {e}")
            st.stop()

    # 3. Visualize Results (if labels were generated)
    if labels is not None:
        try:
            S["unsup_labels"] = labels
            
            st.markdown("### Cluster/Anomaly Visualization (PCA Projection)")
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(df_scaled)
            
            # Convert labels to string for categorical coloring, including -1 for DBSCAN/IF
            label_names = [str(v) for v in labels]
            
            fig = px.scatter(x=X2[:,0], y=X2[:,1], color=label_names, 
                             labels={'x':'PC1','y':'PC2', 'color':'Cluster/Anomaly'}, 
                             title=f"{algo} — PCA Projection of Scaled Data")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization of {algo} results failed: {e}")


# ===================== RESULTS & EMAIL =====================
elif S["page"] == "results":
    st.title("Results & Reports")
    if not S["results"]:
        st.info("Train a supervised model to see results.")
    else:
        st.markdown("### Supervised Model Results")
        
        # Display key metrics in a cleaner way
        c1, c2, c3 = st.columns(3)
        
        results = S["results"]
        task = results.get("task", "Unknown")
        
        with c1:
            st.metric("Model", results.get("model", "N/A"))
        with c2:
            st.metric("Task", task)
        with c3:
            st.metric("Train Time (s)", f"{results.get('train_time', 0):.3f}")
            
        st.markdown("---")

        if task == "Classification":
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", f"{results.get('accuracy', 0)*100:.2f}%")
            with c2:
                st.metric("F1 Score", f"{results.get('f1', 0):.4f}")
        elif task == "Regression":
            c1, c2 = st.columns(2)
            with c1:
                st.metric("RMSE", f"{results.get('rmse', 0):.4f}")
            with c2:
                st.metric("R² Score", f"{results.get('r2', 0):.4f}")

        st.markdown("### Full Results (JSON)")
        st.json(S["results"])
        
        # Email section
        st.markdown("---")
        st.markdown("### Send Report")
        if S["user_email"]:
            if st.button("📧 Send results to user", type="primary"):
                # Pass model and parameters for richer report
                email_html = f"<p>Model: **{results.get('model', 'N/A')}** ({task})</p>"
                ok = send_results_email(S["user_email"], "Your AutoMLPilot Results", S["results"], extra_html=email_html)
                st.success("Report Sent!") if ok else st.error("Email Failed.")
        else:
            st.warning("Enter recipient email on **Dashboard** to enable email reports.")

# ===================== HELP =====================
elif S["page"] == "help":
    st.title("Help & Tips")
    st.markdown("""
    ## 🧠 Modeling Guide

    **How to choose a model?**
    * **Logistic Regression / Linear Regression**: Fast baseline; good for interpretable linear relationships.
    * **Random Forest / Gradient Boosting**: Strong non-linear learners; handle mixed features (especially One-Hot encoding) well. Often a great starting point.
    * **SVM / SVR**: Robust for medium-sized datasets, especially when separation is complex. Requires **StandardScaler**.""")
