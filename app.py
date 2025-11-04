import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional external libs
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ==================== PAGE / THEME ====================
st.set_page_config(
    page_title="No-Code ML Explorer",
    layout="wide",
    page_icon="✨",
)

# -------------------- CSS: fixed navbar & glass UI --------------------
TEMPLATE_CSS = """
<style>
    :root { --glass: rgba(255,255,255,0.10); --border: rgba(255,255,255,0.18); }
    .block-container {padding: 0 1.2rem 2rem 1.2rem;}
    .main { background: linear-gradient(120deg, #0f172a, #1e293b) fixed; }
    h1,h2,h3,h4 { color: #e5e7eb; }
    .muted { color:#cbd5e1; }

    /* Fixed top bar */
    .topbar { position: sticky; top: 0; z-index: 1000; backdrop-filter: blur(10px);
              background: linear-gradient(90deg, rgba(2,6,23,0.75), rgba(15,23,42,0.75));
              border-bottom: 1px solid var(--border); padding: 0.75rem 0.8rem; }
    .topbar-inner { display:flex; align-items:center; justify-content:space-between; }
    .brand { display:flex; align-items:center; gap:.6rem; }
    .brand .title { font-weight:800; font-size:1.05rem; color:#f8fafc; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid var(--border); color:#e5e7eb; font-size:12px; }

    /* Left sidebar nav mimic */
    .side { position: sticky; top: 54px; }
    .nav { background: var(--glass); border:1px solid var(--border); border-radius:16px; padding:10px; }
    .nav a { display:block; padding:.55rem .8rem; margin:.2rem 0; border-radius:10px; text-decoration:none; color:#e5e7eb; }
    .nav a.active, .nav a:hover { background: rgba(139,92,246,0.18); }

    .card { background: var(--glass); border:1px solid var(--border); border-radius:18px; box-shadow: 0 8px 28px rgba(0,0,0,.25); padding:16px; }
    .metric { background: rgba(255,255,255,0.06); border-left:4px solid #8b5cf6; border-radius:12px; padding:10px; color:#e5e7eb; }

    /* File input align */
    .row { display:flex; gap:12px; align-items:center; }
</style>
"""

st.markdown(TEMPLATE_CSS, unsafe_allow_html=True)

# ==================== SIMPLE IN-APP LOGIN (for SMTP owner) ====================
DEFAULT_EMAIL = "rhreddy4748@gmail.com"  # You can change this default
DEFAULT_APP_PASSWORD = "zujpoggswfcpxwjs"  # Gmail App Password (avoid sharing publicly)

if "ui" not in st.session_state:
    st.session_state.ui = {
        "page": "dashboard",
        "authed": False,
        "email": DEFAULT_EMAIL,
        "app_password": DEFAULT_APP_PASSWORD,
        # data & model state
        "original_df": None,
        "df": None,
        "target": None,
        "features": [],
        "task": "Classification",
        "encoding": "One-Hot",
        "scaler": None,
        "encoders": {},
        "ohe_cols": None,
        "final_cols": None,
        "model": None,
        "target_encoder": None,
        "target_classes": None,
        "results": {},
    }
S = st.session_state.ui

# -------------------- Top bar --------------------
st.markdown(
    """
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <span>✨</span>
          <span class="title">No‑Code ML Explorer</span>
          <span class="pill">Dashboard</span>
        </div>
        <div class="brand">
          <span class="pill">Dark</span>
          <span class="pill">{email}</span>
        </div>
      </div>
    </div>
    """.format(email=S["email"] if S["authed"] else "guest@local"),
    unsafe_allow_html=True,
)

# ==================== SIDEBAR NAV ====================
with st.sidebar:
    st.markdown("<div class='side'>", unsafe_allow_html=True)
    st.markdown("### NAVIGATION")
    def nav_btn(name, key):
        active = (S["page"] == key)
        if st.button(("👉 " if active else "") + name, use_container_width=True):
            S["page"] = key
            st.experimental_rerun()
    nav_btn("Dashboard", "dashboard")
    nav_btn("Preprocess", "preprocess")
    nav_btn("Model Selection", "train")
    nav_btn("Results", "results")
    nav_btn("Settings", "settings")

    st.markdown("---")
    st.caption("Pro Version — Unlock features")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== AUTH CARD (Google-style simulated) ====================
if not S["authed"]:
    with st.container():
        st.markdown("### 🔐 Sign in to enable email notifications")
        with st.form("login_form"):
            st.text_input("Gmail address", value=S["email"], key="email_input")
            st.text_input("Gmail App password", type="password", value=S["app_password"], key="pwd_input")
            st.caption("We only use this to send training results via SMTP over SSL. Stored in session only.")
            if st.form_submit_button("Sign in", type="primary"):
                S["email"] = st.session_state.get("email_input", "")
                S["app_password"] = st.session_state.get("pwd_input", "")
                S["authed"] = True
                st.success("Signed in. You can change this later in Settings.")
    st.write("---")

# ==================== DASHBOARD ====================
if S["page"] == "dashboard":
    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.markdown("#### Upload Dataset")
        with st.container():
            upcol1, upcol2 = st.columns([4,1])
            with upcol1:
                uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
            with upcol2:
                st.write("\n")
                if st.button("Upload", type="primary", use_container_width=True):
                    if uploaded is None:
                        st.warning("Please choose a CSV file first.")
                    else:
                        try:
                            new_df = pd.read_csv(uploaded)
                            S["original_df"] = new_df
                            S["df"] = new_df.copy()
                            S["target"] = None
                            S["features"] = []
                            S["model"] = None
                            st.success("File uploaded!")
                            S["page"] = "preprocess"
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
        st.caption("ML Training: CSV. Preview-only: JSON & images (future). Storage: uploads/ → results/")

        t1, t2 = st.columns(2)
        with t1:
            S["task"] = st.selectbox("Task Type", ["Classification", "Regression"], index=["Classification","Regression"].index(S["task"]))
        with t2:
            target_placeholder = "e.g., label" if S["task"] == "Classification" else "e.g., price"
            if S["df"] is not None:
                S["target"] = st.selectbox("Target Column *", [None] + S["df"].columns.tolist(), index=0 if S["target"] is None else 1 + list(S["df"].columns).index(S["target"]))
            else:
                st.text_input("Target Column *", placeholder=target_placeholder, disabled=True)

        b1, b2 = st.columns([1,1])
        with b1:
            if st.button("🧹 Preprocess →", use_container_width=True):
                S["page"] = "preprocess"
                st.experimental_rerun()
        with b2:
            if st.button("⚡ Quick Train", use_container_width=True):
                if S["df"] is None or S["target"] is None:
                    st.warning("Upload data and pick target first.")
                else:
                    st.info("Running quick baseline (LogReg/Linear vs RandomForest)...")
                    # very small quick train
                    DF = S["df"].copy()
                    y = DF[S["target"]]
                    X = DF.drop(columns=[S["target"]])
                    # simple encoding
                    X = pd.get_dummies(X)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if S["task"] == "Classification":
                        model = RandomForestClassifier(n_estimators=150, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)
                    S["model"] = model
                    S["final_cols"] = X.columns.tolist()
                    S["features"] = list(DF.columns.drop(S["target"]))
                    st.success("Quick model trained. Go to Results or Predict tab.")
    with c2:
        st.markdown("#### Recent Datasets")
        if S["df"] is None:
            st.info("No datasets yet. Upload to get started!")
        else:
            st.write("• current.csv — ", S["df"].shape)

# ==================== PREPROCESS ====================
elif S["page"] == "preprocess":
    st.markdown("### 🧹 Preprocess")
    if S["df"] is None:
        st.info("Upload a dataset first.")
        st.stop()
    DF = S["df"].copy()

    with st.expander("Drop columns", expanded=True):
        cols_drop = st.multiselect("Columns to drop", DF.columns.tolist())
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply drop", type="primary") and cols_drop:
                DF.drop(columns=cols_drop, inplace=True, errors="ignore")
                S["df"] = DF
                st.experimental_rerun()
        with c2:
            if st.button("Reset to original"):
                S["df"] = S["original_df"].copy()
                st.experimental_rerun()

    with st.expander("Missing values", expanded=True):
        miss = DF.isna().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.info("No missing values detected.")
        else:
            st.dataframe(miss.rename("Missing"))
            how = st.selectbox("Strategy", ["Drop rows", "Mean/Mode impute"]) 
            if st.button("Fix missing"):
                if how == "Drop rows":
                    DF.dropna(inplace=True)
                else:
                    for c in miss.index:
                        if pd.api.types.is_numeric_dtype(DF[c]):
                            DF[c] = DF[c].fillna(DF[c].mean())
                        else:
                            DF[c] = DF[c].fillna(DF[c].mode().iloc[0])
                S["df"] = DF
                st.experimental_rerun()

    with st.expander("Target & features", expanded=True):
        if S["target"] is None:
            S["target"] = st.selectbox("Target (y)", DF.columns.tolist())
        else:
            S["target"] = st.selectbox("Target (y)", DF.columns.tolist(), index=DF.columns.get_loc(S["target"]))
        if st.button("Confirm target", type="primary"):
            S["features"] = [c for c in DF.columns if c != S["target"]]
            st.success(f"Target set to {S['target']}")
            S["page"] = "train"
            st.experimental_rerun()

# ==================== TRAIN ====================
elif S["page"] == "train":
    st.markdown("### 📦 Model Selection & Training")
    if S["df"] is None or S["target"] is None:
        st.warning("Please upload data and choose a target first.")
        st.stop()

    DF = S["df"].copy()
    target = S["target"]
    features = [c for c in DF.columns if c != target]

    with st.expander("Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            S["task"] = st.selectbox("Task", ["Classification", "Regression"], index=["Classification","Regression"].index(S["task"]))
        with c2:
            model_choices_c = [
                "LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier",
                "GaussianNB", "MLPClassifier", "GradientBoostingClassifier"
            ] + (["XGBClassifier"] if XGB_AVAILABLE else [])
            model_choices_r = [
                "LinearRegression", "RandomForestRegressor", "SVR", "KNeighborsRegressor",
                "DecisionTreeRegressor", "MLPRegressor", "GradientBoostingRegressor", "SGDRegressor"
            ] + (["XGBRegressor"] if XGB_AVAILABLE else [])
            model_name = st.selectbox("Algorithm", model_choices_c if S["task"]=="Classification" else model_choices_r)

    with st.expander("Preprocessing", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            do_scale = st.checkbox("Scale numeric features", value=True)
            S["encoding"] = st.selectbox("Categorical encoding", ["One-Hot", "Label", "Ordinal"], index=["One-Hot","Label","Ordinal"].index(S["encoding"]))
        with c2:
            split = st.slider("Train split", 0.5, 0.95, 0.8)

    with st.expander("Hyperparameters", expanded=True):
        params = {}
        if model_name == "LogisticRegression":
            params["C"] = st.number_input("C", 0.01, 10.0, 1.0)
            params["max_iter"] = st.number_input("Max iter", 50, 5000, 300)
            params["solver"] = "liblinear"
        elif model_name.startswith("RandomForest"):
            params["n_estimators"] = st.slider("n_estimators", 50, 800, 200)
            params["max_depth"] = st.slider("max_depth", 1, 60, 12)
            params["random_state"] = 42
        elif model_name in ("SVC", "SVR"):
            params["C"] = st.slider("C", 0.01, 10.0, 1.0)
            params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"]) 
        elif model_name.startswith("KNeighbors"):
            params["n_neighbors"] = st.slider("n_neighbors", 1, 35, 5)
        elif model_name == "GaussianNB":
            pass
        elif model_name == "DecisionTreeRegressor":
            params["max_depth"] = st.slider("max_depth", 1, 60, 10)
            params["random_state"] = 42
        elif model_name in ("MLPClassifier", "MLPRegressor"):
            params["hidden_layer_sizes"] = (st.slider("Hidden units", 16, 512, 128),)
            params["learning_rate_init"] = st.slider("Learning rate", 1e-4, 1e-1, 1e-3)
            params["max_iter"] = st.slider("Epochs (max_iter)", 50, 2000, 300)
            params["early_stopping"] = st.checkbox("Early stopping", value=True)
        elif model_name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
            params["n_estimators"] = st.slider("n_estimators", 50, 600, 200)
            params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1)
            params["max_depth"] = st.slider("max_depth", 1, 8, 3)
        elif model_name in ("XGBClassifier", "XGBRegressor") and XGB_AVAILABLE:
            params["n_estimators"] = st.slider("n_estimators", 50, 1000, 300)
            params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1)
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.9)
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.5, 1.0, 0.9)
            if model_name == "XGBClassifier":
                params["eval_metric"] = "logloss"

    with st.expander("Execution", expanded=True):
        exec_target = st.radio("Run on", ["Local", "Kaggle (sim)"])
        email_me = st.checkbox("Email me results when done", value=True if S["authed"] else False)

    # Train button
    if st.button("🚀 Train", type="primary", use_container_width=True):
        with st.status("Training...", expanded=True) as status:
            X = DF[features].copy()
            y = DF[target].copy()

            # Encode categoricals
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            encoders = {}
            ohe_cols = None
            if cat_cols:
                status.write(f"Encoding {len(cat_cols)} categorical features with {S['encoding']}...")
                if S["encoding"] == "One-Hot":
                    X = pd.get_dummies(X, drop_first=False)
                    ohe_cols = X.columns.tolist()
                else:
                    for c in cat_cols:
                        le = LabelEncoder()
                        X[c] = le.fit_transform(X[c].astype(str))
                        encoders[c] = le

            # Target encoding for classification strings
            target_encoder = None
            if S["task"] == "Classification" and y.dtype == "object":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

            # Scale
            scaler = None
            if do_scale:
                status.write("Scaling features...")
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            final_cols = X.columns.tolist()

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-split, random_state=42, stratify=y if S["task"]=="Classification" else None
            )

            # Model map
            model_map = {
                "LogisticRegression": LogisticRegression,
                "RandomForestClassifier": RandomForestClassifier,
                "SVC": SVC,
                "KNeighborsClassifier": KNeighborsClassifier,
                "GaussianNB": GaussianNB,
                "MLPClassifier": MLPClassifier,
                "GradientBoostingClassifier": GradientBoostingClassifier,
                "LinearRegression": LinearRegression,
                "RandomForestRegressor": RandomForestRegressor,
                "SVR": SVR,
                "KNeighborsRegressor": KNeighborsRegressor,
                "DecisionTreeRegressor": DecisionTreeRegressor,
                "MLPRegressor": MLPRegressor,
                "GradientBoostingRegressor": GradientBoostingRegressor,
                "SGDRegressor": SGDRegressor,
            }
            if XGB_AVAILABLE:
                model_map.update({"XGBClassifier": XGBClassifier, "XGBRegressor": XGBRegressor})

            Model = model_map[model_name]
            model = Model(**params)

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)

            results = {"Execution": exec_target.split()[0], "Training Time (s)": round(train_time, 3), "Model": model_name}

            st.markdown("---")
            st.markdown("#### Results")
            if S["task"] == "Classification":
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric'><b>Accuracy</b><br><h3>{acc*100:.2f}%</h3></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric'><b>F1 (weighted)</b><br><h3>{f1:.4f}</h3></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric'><b>Time</b><br><h3>{train_time:.2f}s</h3></div>", unsafe_allow_html=True)

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                ax.imshow(cm)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha='center', va='center', color='white')
                st.pyplot(fig, clear_figure=True, use_container_width=True)

                results.update({"Accuracy": round(acc, 4), "F1-Score": round(f1, 4)})
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric'><b>RMSE</b><br><h3>{rmse:.4f}</h3></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric'><b>R²</b><br><h3>{r2:.4f}</h3></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric'><b>Time</b><br><h3>{train_time:.2f}s</h3></div>", unsafe_allow_html=True)
                results.update({"RMSE": round(rmse, 4), "R2": round(r2, 4)})

            # Persist
            S.update({
                "model": model,
                "scaler": scaler,
                "encoders": encoders,
                "ohe_cols": ohe_cols,
                "final_cols": final_cols,
                "features": features,
                "target_encoder": target_encoder,
                "target_classes": list(target_encoder.classes_) if target_encoder is not None else None,
                "results": results,
            })

            # Email
            if email_me and S["authed"] and S["email"] and S["app_password"]:
                status.write("Sending email...")
                ok, message = (lambda res, m, t: (
                    (lambda sender, pwd, rcpt: (
                        (lambda: (False, "Credentials missing")) if not (sender and pwd) else (lambda: (
                            (lambda: (
                                (lambda server: (
                                    server.login(sender, pwd), server.sendmail(sender, sender, m.as_string()), server.quit(), True
                                ))(smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()))
                            ))()
                        ))()
                    ))(S["email"], S["app_password"], S["email"]))
                )({**results},
                  (lambda res: (lambda msg: (
                      msg.attach(MIMEText(f"""
<html><body style='font-family:Inter,system-ui;color:#0b1220'>
<h2 style='color:#8b5cf6;margin:0'>ML Training Complete</h2>
<p><b>Model</b>: {res.get('Model','')}</p>
<p><b>Execution</b>: {res.get('Execution','')} • <b>Time</b>: {res.get('Training Time (s)','')} s</p>
<pre style='background:#0b1220;color:#e5e7eb;border:1px solid #263043;padding:14px;border-radius:10px;'>{json.dumps(res, indent=2)}</pre>
</body></html>""", "html")), msg)[1]
                  )(MIMEMultipart("alternative"))), S["task"])  # type: ignore
                status.write("✅ Email sent" if ok else f"⚠️ {message}")

            status.update(label="✅ Complete", state="complete", expanded=False)

# ==================== RESULTS ====================
elif S["page"] == "results":
    st.markdown("### 📊 Results")
    if not S.get("results"):
        st.info("Train a model to see results.")
    else:
        st.json(S["results"])

# ==================== SETTINGS ====================
elif S["page"] == "settings":
    st.markdown("### ⚙️ Settings")
    with st.form("smtp_form"):
        st.text_input("Gmail address", value=S["email"], key="email_edit")
        st.text_input("Gmail App password", type="password", value=S["app_password"], key="pwd_edit")
        st.caption("These are used only to send emails to yourself via Gmail SMTP over SSL.")
        if st.form_submit_button("Save"):
            S["email"] = st.session_state.get("email_edit", "")
            S["app_password"] = st.session_state.get("pwd_edit", "")
            st.success("Saved.")

    if S.get("df") is not None:
        st.download_button(
            "⬇ Download current CSV",
            S["df"].to_csv(index=False).encode("utf-8"),
            file_name="processed.csv",
            mime="text/csv",
        )
