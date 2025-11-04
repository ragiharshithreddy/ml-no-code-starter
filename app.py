# AutoMLPilot Pro — colorful no‑code ML lab with playgrounds
# -----------------------------------------------------------
# - Supervised (classification/regression) + Unsupervised (clustering/anomaly/dim‑red)
# - Rich preprocessing: imputation, encoding, scaling, outliers, balancing (SMOTE*), feature selection
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
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
)
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
    if not to_email or "@" not in to_email:
        st.warning("Please enter a valid email.")
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
            s.sendmail(OWNER_GMAIL, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

@st.cache_data(show_spinner=False)
def profile_html(df: pd.DataFrame) -> str:
    if not YDATA_OK:
        return "<p class='small'>Install ydata-profiling to enable full EDA.</p>"
    pr = ProfileReport(df, explorative=True, minimal=True)
    return pr.to_html()

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
        # training artifacts
        "model": None,
        "final_cols": None,
        "label_encoders": {},
        "scaler_name": None,
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
                      "preprocess":"🧹 Preprocess",
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
                df = pd.read_csv(up)
                S["df"] = df
                S["target"] = None
                st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed: {e}")
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
            html(profile_html(S["df"]), height=600, scrolling=True)

# ===================== PREPROCESS =====================
elif S["page"] == "preprocess":
    st.title("Preprocessing Studio")
    if S["df"] is None:
        st.info("Upload a dataset first.")
        st.stop()
    df = S["df"].copy()

    with st.expander("1) Handle Missing Values", expanded=True):
        strategy = st.selectbox("Strategy", ["None","Mean","Median","Most_frequent"])
        if strategy != "None":
            num_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            if strategy in ("Mean","Median") and len(num_cols)==0:
                st.warning("No numeric columns for mean/median.")
            else:
                imputer_num = SimpleImputer(strategy=strategy.lower()) if strategy!="Most_frequent" else SimpleImputer(strategy="most_frequent")
                if len(num_cols):
                    df[num_cols] = imputer_num.fit_transform(df[num_cols])
                if len(cat_cols):
                    imputer_cat = SimpleImputer(strategy="most_frequent")
                    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
                st.success("Missing values imputed.")

    with st.expander("2) Outliers (IQR remove)"):
        if st.checkbox("Remove outliers using IQR (numeric only)"):
            num_cols = df.select_dtypes(include=[np.number]).columns
            before = len(df)
            for col in num_cols:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            st.success(f"Removed {before - len(df)} rows as outliers.")

    with st.expander("3) Encoding", expanded=True):
        enc = st.selectbox("Categorical encoding", ["None","One-Hot","Label"])
        if enc == "One-Hot":
            df = pd.get_dummies(df)
            st.info("Applied one‑hot encoding.")
        elif enc == "Label":
            for c in df.select_dtypes(include=["object","category"]).columns:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
            st.info("Applied label encoding.")

    with st.expander("4) Scaling", expanded=False):
        scale = st.selectbox("Scaler", ["None","Standard","MinMax","Robust","Normalize"])
        if scale != "None":
            scaler = {
                "Standard": StandardScaler(),
                "MinMax": MinMaxScaler(),
                "Robust": RobustScaler(),
                "Normalize": Normalizer(),
            }[scale]
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
            S["scaler_name"] = scale
            st.info(f"Applied {scale} scaling to numeric columns.")

    with st.expander("5) Feature Selection (VarianceThreshold)"):
        if st.checkbox("Remove near‑constant features"):
            thr = st.slider("Variance threshold", 0.0, 0.2, 0.0, 0.01)
            selector = VarianceThreshold(threshold=thr)
            arr = selector.fit_transform(df.select_dtypes(include=[np.number]))
            kept = df.select_dtypes(include=[np.number]).columns[selector.get_support(indices=True)]
            nonnum = df.select_dtypes(exclude=[np.number])
            df = pd.concat([pd.DataFrame(arr, columns=kept), nonnum.reset_index(drop=True)], axis=1)
            st.success(f"Kept {len(kept)} numeric features.")

    with st.expander("6) Balancing (SMOTE)"):
        if not IMB_OK:
            st.caption("Install imbalanced-learn to enable SMOTE.")
        else:
            st.caption("Applies when you choose a target in Train tab (classification only).")

    # Save
    S["df"] = df
    st.success("Preprocessing applied. Move to Train or Playground.")

# ===================== FEATURE ENGINEERING =====================
def correlation_recommendations(df: pd.DataFrame, thresh=0.85):
    num = df.select_dtypes(include=[np.number])
    recs = []
    if num.shape[1] < 2:
        return recs
    corr = num.corr()
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i: continue
            v = corr.iloc[i, j]
            if abs(v) >= thresh:
                recs.append((c1, c2, float(v)))
    return sorted(recs, key=lambda x: -abs(x[2]))[:20]

# ===================== TRAIN (SUPERVISED) =====================
elif S["page"] == "train":
    st.title("Supervised Training")
    if S["df"] is None:
        st.info("Upload data first.")
        st.stop()
    df = S["df"].copy()

    # Target selection
    S["target"] = st.selectbox("Target (y)", [None] + df.columns.tolist(), index=0 if S["target"] is None else 1 + list(df.columns).index(S["target"]))
    if S["target"] is None:
        st.stop()

    # Task
    task = st.radio("Task", ["Classification","Regression"], horizontal=True)

    # Feature engineering suggestions
    with st.expander("✨ Feature Recommendations (from correlation)", expanded=False):
        recs = correlation_recommendations(df.drop(columns=[S["target"]]))
        if recs:
            st.dataframe(pd.DataFrame(recs, columns=["Feature A","Feature B","Correlation"]))
        else:
            st.caption("No strong pairs found.")
        st.caption("Create derived features below:")
        cols = [c for c in df.columns if c != S["target"]]
        new_name = st.text_input("New feature name", placeholder="e.g., a_div_b")
        if cols:
            c1, c2 = st.columns(2)
            with c1:
                f1 = st.selectbox("Feature 1", cols)
                op = st.selectbox("Operation", ["+","-","*","/"])
            with c2:
                f2 = st.selectbox("Feature 2", [c for c in cols if c != f1])
            if st.button("Create feature"):
                try:
                    if op=="+": df[new_name] = df[f1] + df[f2]
                    if op=="-": df[new_name] = df[f1] - df[f2]
                    if op=="*": df[new_name] = df[f1] * df[f2]
                    if op=="/": df[new_name] = df[f1] / df[f2].replace(0,np.nan)
                    S["df"] = df
                    S["features_created"].append(new_name)
                    st.success(f"Feature '{new_name}' created.")
                except Exception as e:
                    st.error(e)

    # Model zoo
    st.markdown("### Choose Model & Params")
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
            "learning_rate_init": (st.slider, {"label":"LR", "min_value":1e-4, "max_value":1e-1, "value":1e-3}),
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
            "learning_rate_init": (st.slider, {"label":"LR", "min_value":1e-4, "max_value":1e-1, "value":1e-3}),
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

    zoo = models_cls if task=="Classification" else models_reg
    mname = st.selectbox("Model", list(zoo.keys()))

    # build kwargs via widgets with inline help
    params = {}
    with st.expander("Hyperparameters", expanded=True):
        for pname, (wfun, kwargs) in zoo[mname][1].items():
            label = kwargs.pop("label", pname)
            widget_val = wfun(label, **kwargs)
            # adapt for MLP hidden_layer_sizes (int -> tuple)
            if pname == "hidden_layer_sizes":
                widget_val = (int(widget_val),)
            params[pname] = widget_val
            st.caption(f"ℹ️ {pname}: {label}")

    # Train
    if st.button("🚀 Train", type="primary"):
        X = df.drop(columns=[S["target"]])
        y = df[S["target"]]
        # encode categoricals
        for c in X.select_dtypes(include=["object","category"]).columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
        if task=="Classification" and y.dtype=="object":
            y = LabelEncoder().fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                            stratify=y if task=="Classification" else None)
        Model = zoo[mname][0]
        model = Model(**params)
        t0 = time.time(); model.fit(X_train, y_train); t = time.time()-t0
        y_pred = model.predict(X_test)

        if task=="Classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            st.markdown(f"<div class='metric'><b>Accuracy</b> {acc*100:.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'><b>F1 (weighted)</b> {f1:.4f}</div>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax); st.pyplot(fig)
            S["results"] = {"task":task, "model":mname, "accuracy":acc, "f1":f1, "train_time":round(t,3)}
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            st.markdown(f"<div class='metric'><b>RMSE</b> {rmse:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'><b>R²</b> {r2:.4f}</div>", unsafe_allow_html=True)
            S["results"] = {"task":task, "model":mname, "rmse":rmse, "r2":r2, "train_time":round(t,3)}
        S["model"] = model
        S["final_cols"] = list(X.columns)
        st.success("Training complete. See Results or try Playground.")

# ===================== PLAYGROUND (supervised) =====================
elif S["page"] == "playground":
    st.title("Model Playground (Supervised)")
    if S["df"] is None or S["target"] is None or S["model"] is None:
        st.info("Train a supervised model first.")
        st.stop()
    df = S["df"]; target = S["target"]
    X = df.drop(columns=[target]); y = df[target]
    for c in X.select_dtypes(include=["object","category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    if y.dtype=="object": y = LabelEncoder().fit_transform(y.astype(str))

    # reduce to 2D for visualization
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    st.caption("Visualizing data in 2D using PCA.")
    fig = px.scatter(x=X2[:,0], y=X2[:,1], color=y.astype(str), labels={'x':'PC1','y':'PC2'}, title="Data (PCA)")
    st.plotly_chart(fig, use_container_width=True)

    # decision boundary (classification) or actual vs pred (regression)
    model = S["model"]
    task = S["results"].get("task","Classification")

    if task == "Classification":
        # train model on PCA space for boundary
        X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y)
        clone = type(model)(**getattr(model, 'get_params', lambda: {})())
        try:
            clone.fit(X_train, y_train)
            # grid
            x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
            y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            Z = clone.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig2 = go.Figure()
            fig2.add_trace(go.Contour(x=np.linspace(x_min,x_max,200), y=np.linspace(y_min,y_max,200), z=Z,
                                      showscale=False, contours_coloring='heatmap', opacity=0.3))
            fig2.add_trace(go.Scatter(x=X2[:,0], y=X2[:,1], mode='markers',
                                      marker=dict(size=6), text=[str(v) for v in y]))
            fig2.update_layout(title="Decision Boundary (PCA space)", xaxis_title="PC1", yaxis_title="PC2")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Boundary viz not available for this model: {e}")
    else:
        # regression: actual vs predicted
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = S["model"].predict(X_test)
        fig3 = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title='Actual vs Predicted')
        fig3.add_shape(type='line', x0=float(np.min(y_test)), y0=float(np.min(y_test)),
                       x1=float(np.max(y_test)), y1=float(np.max(y_test)))
        st.plotly_chart(fig3, use_container_width=True)

# ===================== UNSUPERVISED =====================
elif S["page"] == "unsupervised":
    st.title("Unsupervised Lab")
    if S["df"] is None:
        st.info("Upload a dataset first.")
        st.stop()
    df = S["df"].copy()
    # encode objects for algorithms
    for c in df.select_dtypes(include=["object","category"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    algo = st.selectbox("Algorithm", ["KMeans","DBSCAN","Agglomerative","GaussianMixture","IsolationForest","PCA (2D)"])
    params = {}
    if algo=="KMeans":
        k = st.slider("k (clusters)", 2, 15, 4)
        params["n_clusters"] = k
        model = KMeans(**params, n_init=10)
        labels = model.fit_predict(df)
        sil = silhouette_score(df, labels) if len(set(labels))>1 else np.nan
        st.caption(f"Silhouette: {sil:.4f}")
    elif algo=="DBSCAN":
        eps = st.slider("eps", 0.1, 10.0, 0.8)
        min_s = st.slider("min_samples", 3, 50, 5)
        model = DBSCAN(eps=eps, min_samples=min_s)
        labels = model.fit_predict(df)
    elif algo=="Agglomerative":
        k = st.slider("clusters", 2, 15, 5)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(df)
    elif algo=="GaussianMixture":
        k = st.slider("components", 2, 15, 4)
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(df)
    elif algo=="IsolationForest":
        c = st.slider("contamination", 0.01, 0.4, 0.05)
        model = IsolationForest(contamination=c, random_state=42)
        labels = model.fit_predict(df)
    else:  # PCA (2D)
        pca = PCA(n_components=2); X2 = pca.fit_transform(df)
        st.plotly_chart(px.scatter(x=X2[:,0], y=X2[:,1], title="PCA 2D"), use_container_width=True)
        st.stop()

    # visualize clusters in 2D
    pca = PCA(n_components=2); X2 = pca.fit_transform(df)
    fig = px.scatter(x=X2[:,0], y=X2[:,1], color=[str(v) for v in labels], title=f"{algo} — PCA Projection")
    st.plotly_chart(fig, use_container_width=True)
    S["unsup_labels"] = labels

# ===================== RESULTS & EMAIL =====================
elif S["page"] == "results":
    st.title("Results & Reports")
    if not S["results"]:
        st.info("Train a model to see results.")
    else:
        st.json(S["results"])
        if S["user_email"]:
            if st.button("📧 Send results to user", type="primary"):
                ok = send_results_email(S["user_email"], "Your AutoMLPilot Results", S["results"])
                st.success("Sent") if ok else st.error("Failed")
        else:
            st.warning("Enter recipient email on Dashboard.")

# ===================== HELP =====================
elif S["page"] == "help":
    st.title("Help & Tips")
    st.markdown("""
    **How to choose a model?**
    - **Logistic Regression**: fast baseline for classification; good with linear relationships.
    - **Random Forest / Gradient Boosting**: strong non‑linear learners; handle mixed features.
    - **SVM**: robust for medium‑sized datasets; try RBF kernel.
    - **KNN**: simple; useful when decision boundary is local.
    - **Neural MLP**: try when you want non‑linear fits without feature crafting.
    - **For regression**: start with Linear → Random Forest / GB → SVR / MLP.

    **Preprocessing recipes**
    - Use **One‑Hot** for categorical features with tree models (RF/GB) or linear models.
    - Use **StandardScaler** for SVM/MLP/SVR.
    - Remove extreme outliers with **IQR** when numeric ranges are skewed.

    **Playground** renders PCA to 2D and shows decision boundaries (classification) or Actual vs Predicted (regression).
    """)
