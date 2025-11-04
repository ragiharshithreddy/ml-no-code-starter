import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json

# ==================== PAGE / THEME ====================
st.set_page_config(
    page_title="No-Code ML Explorer",
    layout="wide",
    page_icon="✨",
)

# -------------------- Styles (glass + tidy UI) --------------------
GLASSY_CSS = """
<style>
    :root { --glass: rgba(255,255,255,0.12); }
    .block-container {padding: 1.25rem 2rem;}
    .main { background: linear-gradient(120deg, #0f172a, #1e293b) fixed; }
    h1, h2, h3, h4, h5 { color: #e5e7eb; }
    .muted { color:#cbd5e1; }
    .glass-card {
        background: var(--glass);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        box-shadow: 0 6px 28px rgba(0,0,0,0.25);
        backdrop-filter: blur(12px);
        padding: 16px;
        margin-bottom: 16px;
    }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); color:#e5e7eb; font-size:12px; }
    .metric-card {background: rgba(255,255,255,0.08); border-left:4px solid #8b5cf6; border-radius:12px; padding:12px;}
    .btn-primary button { background: linear-gradient(90deg,#8b5cf6,#6366f1)!important; color:white!important; border:none!important; }
    .btn-ghost button { background: transparent!important; color:#e5e7eb!important; border:1px solid rgba(255,255,255,0.25)!important; }
    .stDataFrame { background: rgba(255,255,255,0.06)!important; border-radius:12px; }
</style>
"""

st.markdown(GLASSY_CSS, unsafe_allow_html=True)

# ==================== HELPERS ====================
@st.cache_data(ttl=3600)
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def build_profile_html(df):
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(
            df,
            explorative=True,
            minimal=True,
            correlations={"auto": {"calculate": False}},
        )
        return profile.to_html()
    except Exception as e:
        return f"<p class='muted'>EDA report unavailable: {e}</p>"

# Email — secure by environment variables / st.secrets
# Set the following in .streamlit/secrets.toml or environment:
# SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD, SMTP_RECIPIENT_EMAIL

def get_email_creds():
    email = st.secrets.get("SMTP_SENDER_EMAIL", os.getenv("SMTP_SENDER_EMAIL", ""))
    pwd = st.secrets.get("SMTP_APP_PASSWORD", os.getenv("SMTP_APP_PASSWORD", ""))
    rcpt = st.secrets.get("SMTP_RECIPIENT_EMAIL", os.getenv("SMTP_RECIPIENT_EMAIL", ""))
    return email, pwd, rcpt


def send_email(results: dict, model_name: str, task_type: str):
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    sender, app_pwd, recipient = get_email_creds()
    if not sender or not app_pwd or not recipient:
        return False, "Email not configured (use st.secrets or env)."
    if "@" not in sender or len(app_pwd) < 12:
        return False, "Invalid email credentials."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"✅ ML Training Complete: {model_name}"
        msg["From"] = sender
        msg["To"] = recipient

        html_body = f"""
        <html><body style='font-family:Inter,system-ui'>
            <h2 style='color:#8b5cf6;margin-bottom:6px'>ML Training Complete</h2>
            <p><b>{model_name} ({task_type})</b> finished training.</p>
            <pre style="background:#0b1220;color:#e5e7eb;border:1px solid #263043;padding:14px;border-radius:10px;">
{json.dumps(results, indent=2)}
            </pre>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, app_pwd)
            server.sendmail(sender, recipient, msg.as_string())
        return True, "Email sent"
    except Exception as e:
        return False, f"Email failed: {e}"

# Session bootstrap
if "state" not in st.session_state:
    st.session_state.state = {
        "authed": False,
        "page": "upload",
        "original_df": None,
        "df": None,
        "target": None,
        "feature_cols": [],
        # training artifacts
        "model": None,
        "scaler": None,
        "encodings": {},   # per-column encoders for Label/Ordinal
        "ohe_cols": None,  # columns after one-hot
        "task": None,
        "target_encoder": None,
        "target_classes": None,
        # results
        "local_results": {},
        "kaggle_results": {},
    }
S = st.session_state.state

# ==================== TOP NAV ====================
with st.container():
    lcol, rcol = st.columns([0.7, 0.3])
    with lcol:
        st.markdown("<h1>✨ No‑Code ML Explorer</h1>", unsafe_allow_html=True)
        st.markdown("<span class='pill'>Upload → Explore → Train → Predict</span>", unsafe_allow_html=True)
    with rcol:
        if not S["authed"]:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔑 Sign in (demo)", use_container_width=True):
                    S["authed"] = True
                    st.rerun()
            with c2:
                if st.button("➡ Continue as guest", use_container_width=True):
                    S["authed"] = True
                    st.rerun()

if not S["authed"]:
    st.stop()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📂 Data & Settings")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            new_df = load_csv(up)
            if S["original_df"] is None or not S["original_df"].equals(new_df):
                S.update({
                    "original_df": new_df,
                    "df": new_df.copy(),
                    "target": None,
                    "feature_cols": [],
                    "model": None,
                    "scaler": None,
                    "encodings": {},
                    "ohe_cols": None,
                    "target_encoder": None,
                    "target_classes": None,
                    "local_results": {},
                    "kaggle_results": {},
                    "page": "preprocess",
                })
                st.success("File uploaded.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if S["df"] is not None:
        st.caption(f"Rows: {S['df'].shape[0]} • Cols: {S['df'].shape[1]}")
        st.download_button(
            "⬇ Download current CSV",
            S["df"].to_csv(index=False).encode("utf-8"),
            file_name="processed.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("### 📧 Email (optional)")
    em, pw, rcpt = get_email_creds()
    if em and pw and rcpt:
        st.success(f"Configured: {em} → {rcpt}")
    else:
        st.info("Add SMTP_* to st.secrets or env to enable notifications.")

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        options=["upload", "preprocess", "explore", "train", "predict"],
        format_func=lambda k: {
            "upload": "📁 Upload",
            "preprocess": "🧹 Preprocess",
            "explore": "📊 Explore",
            "train": "🧠 Train",
            "predict": "🔮 Predict",
        }[k],
        index=["upload", "preprocess", "explore", "train", "predict"].index(S["page"]),
    )
    S["page"] = page

# Guard for pages needing data
if S["df"] is None:
    st.info("👆 Upload a CSV to begin.")
    st.stop()

DF = S["df"].copy()

# ==================== PREPROCESS ====================
if S["page"] == "preprocess":
    st.markdown("### 🧹 Preprocessing Pipeline")
    with st.container():
        st.markdown("#### 1) Drop columns")
        drop_cols = st.multiselect("Choose columns to drop", DF.columns.tolist())
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply drop", type="primary") and drop_cols:
                DF.drop(columns=drop_cols, inplace=True, errors="ignore")
                S["df"] = DF
                st.success(f"Dropped {len(drop_cols)} columns.")
                st.rerun()
        with c2:
            if st.button("Reset to original", type="secondary"):
                S["df"] = S["original_df"].copy()
                st.experimental_rerun()

    with st.container():
        st.markdown("#### 2) Missing values")
        miss = DF.isna().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.info("No missing values detected.")
        else:
            st.dataframe(miss.rename("Missing"))
            how = st.selectbox("Strategy", ["Drop rows", "Mean/Mode impute"])
            if st.button("Fix missing"):
                if how == "Drop rows":
                    before = len(DF)
                    DF.dropna(inplace=True)
                    st.success(f"Dropped {before - len(DF)} rows with NA.")
                else:
                    for c in miss.index:
                        if pd.api.types.is_numeric_dtype(DF[c]):
                            DF[c] = DF[c].fillna(DF[c].mean())
                        else:
                            DF[c] = DF[c].fillna(DF[c].mode().iloc[0])
                    st.success("Imputed missing values.")
                S["df"] = DF
                st.rerun()

    with st.container():
        st.markdown("#### 3) Select target column")
        tgt = st.selectbox("Target (y)", DF.columns.tolist(), index=0 if S["target"] is None else DF.columns.get_loc(S["target"]))
        if st.button("Confirm target", type="primary"):
            S["target"] = tgt
            S["feature_cols"] = [c for c in DF.columns if c != tgt]
            st.success(f"Target set to: {tgt}")
            S["page"] = "explore"
            st.rerun()

# ==================== EXPLORE ====================
elif S["page"] == "explore":
    st.markdown("### 📊 Explore")

    with st.expander("📦 Data preview", expanded=True):
        st.dataframe(DF.head())

    with st.expander("✨ Auto EDA report", expanded=False):
        with st.spinner("Building EDA report..."):
            html(build_profile_html(DF), height=600, scrolling=True)

    st.markdown("---")
    st.markdown("#### 📈 Chart builder")
    col1, col2 = st.columns([1, 3])
    with col1:
        chart = st.selectbox("Chart", ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie"])
        x = st.selectbox("X", DF.columns)
        y = st.selectbox("Y", [None] + [c for c in DF.columns if c != x])
        color = st.selectbox("Color (categorical)", [None] + DF.select_dtypes(include=["object", "category"]).columns.tolist())
        title = st.text_input("Title", f"{chart} chart")
    with col2:
        if st.button("▶ Generate", type="primary", use_container_width=True):
            try:
                fn = {
                    "Scatter": px.scatter,
                    "Line": px.line,
                    "Bar": px.bar,
                    "Histogram": px.histogram,
                    "Box": px.box,
                    "Pie": px.pie,
                }[chart]
                kwargs = {"data_frame": DF, "title": title}
                if chart == "Pie":
                    if y is None:
                        st.warning("For Pie, choose a values column as Y.")
                    else:
                        fig = fn(data_frame=DF, names=x, values=y, title=title)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    kwargs.update({"x": x})
                    if y is not None and chart != "Histogram":
                        kwargs["y"] = y
                    if color is not None:
                        kwargs["color"] = color
                    fig = fn(**kwargs)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart failed: {e}")

# ==================== TRAIN ====================
elif S["page"] == "train":
    if not S["target"]:
        st.warning("Select a target on the Preprocess page first.")
        st.stop()

    st.markdown("### 🧠 Train a model")
    target = S["target"]
    features = [c for c in DF.columns if c != target]

    with st.expander("⚙️ Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            task = st.selectbox("Task", ["Classification", "Regression"], index=0 if S.get("task") is None else ["Classification","Regression"].index(S["task"]))
        with c2:
            model_choices = {
                "Classification": ["LogisticRegression", "RandomForestClassifier"] + (["XGBClassifier"] if XGB_AVAILABLE else []),
                "Regression": ["LinearRegression", "DecisionTreeRegressor"] + (["XGBRegressor"] if XGB_AVAILABLE else []),
            }
            model_name = st.selectbox("Algorithm", model_choices[task])

    with st.expander("🧹 Preprocessing", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            do_scale = st.checkbox("Scale numeric features", value=True)
            encoding = st.selectbox("Categorical encoding", ["One-Hot", "Label", "Ordinal"]) 
        with c2:
            split = st.slider("Train split", 0.5, 0.95, 0.8)

    with st.expander("🛠 Hyperparameters", expanded=False):
        params = {}
        if model_name == "LogisticRegression":
            C = st.number_input("C", 0.01, 10.0, 1.0)
            max_iter = st.number_input("Max iter", 50, 2000, 300)
            params = {"C": C, "max_iter": int(max_iter), "solver": "liblinear"}
        elif model_name == "RandomForestClassifier":
            trees = st.slider("n_estimators", 10, 500, 200)
            depth = st.slider("max_depth", 1, 50, 12)
            params = {"n_estimators": trees, "max_depth": depth, "random_state": 42}
        elif model_name == "DecisionTreeRegressor":
            depth = st.slider("max_depth", 1, 50, 10)
            params = {"max_depth": depth, "random_state": 42}
        elif model_name in ("XGBClassifier", "XGBRegressor") and XGB_AVAILABLE:
            lr = st.slider("learning_rate", 0.01, 0.5, 0.1)
            est = st.slider("n_estimators", 50, 600, 200)
            params = {"learning_rate": lr, "n_estimators": est, "random_state": 42}
            if model_name == "XGBClassifier":
                params.update({"eval_metric": "logloss"})

    with st.expander("🎯 Execution", expanded=True):
        exec_target = st.radio("Run on", ["Local", "Kaggle (sim)"])
        email_enabled = st.checkbox("Send email when done (if configured)")

    # Train button
    if st.button("🚀 Train", type="primary", use_container_width=True):
        with st.status("Training...", expanded=True) as st_status:
            X = DF[features].copy()
            y = DF[target].copy()

            # Encode categoricals
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            encoders = {}
            ohe_cols = None
            if cat_cols:
                st_status.write(f"Encoding {len(cat_cols)} categorical features with {encoding}...")
                if encoding == "One-Hot":
                    X = pd.get_dummies(X, drop_first=False)
                    ohe_cols = X.columns.tolist()
                elif encoding in ("Label", "Ordinal"):
                    for c in cat_cols:
                        le = LabelEncoder()
                        X[c] = le.fit_transform(X[c].astype(str))
                        encoders[c] = le
                else:
                    pass

            # Target encoding for classification with string labels
            target_encoder = None
            if task == "Classification" and y.dtype == "object":
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))

            # Scale numeric features
            scaler = None
            if do_scale:
                st_status.write("Scaling features...")
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            final_feature_cols = X.columns.tolist()

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - split, random_state=42, stratify=y if task == "Classification" else None
            )

            # Model
            st_status.write(f"Training {model_name}...")
            model_map = {
                "LogisticRegression": LogisticRegression,
                "RandomForestClassifier": RandomForestClassifier,
                "LinearRegression": LinearRegression,
                "DecisionTreeRegressor": DecisionTreeRegressor,
            }
            if XGB_AVAILABLE:
                model_map.update({"XGBClassifier": XGBClassifier, "XGBRegressor": XGBRegressor})

            model = model_map[model_name](**params)

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)

            results = {"Execution": exec_target.split()[0], "Training Time (s)": round(train_time, 3)}

            st.markdown("---")
            st.markdown("#### 💡 Results")

            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><b>Accuracy</b><br><h3>{acc*100:.2f}%</h3></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><b>F1 (weighted)</b><br><h3>{f1:.4f}</h3></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><b>Time</b><br><h3>{train_time:.2f}s</h3></div>", unsafe_allow_html=True)

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                im = ax.imshow(cm)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha='center', va='center')
                st.pyplot(fig, clear_figure=True, use_container_width=True)

                results.update({"Accuracy": round(acc, 4), "F1-Score": round(f1, 4)})
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><b>RMSE</b><br><h3>{rmse:.4f}</h3></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><b>R²</b><br><h3>{r2:.4f}</h3></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><b>Time</b><br><h3>{train_time:.2f}s</h3></div>", unsafe_allow_html=True)
                results.update({"RMSE": round(rmse, 4), "R2": round(r2, 4)})

            # Save artifacts to session
            S.update({
                "model": model,
                "scaler": scaler,
                "encodings": encoders,
                "ohe_cols": ohe_cols,
                "task": task,
                "feature_cols": features,
                "target": target,
                "target_encoder": target_encoder,
                "target_classes": list(target_encoder.classes_) if target_encoder is not None else None,
                "final_feature_cols": final_feature_cols,
            })

            # Fake Kaggle path (for comparison table parity)
            if exec_target.startswith("Kaggle"):
                S["kaggle_results"] = {**results}
            else:
                S["local_results"] = {**results}

            # Optional email
            if email_enabled:
                st_status.write("Sending email...")
                ok, msg = send_email(results, model_name, task)
                if ok:
                    st_status.write("✅ " + msg)
                else:
                    st_status.write("⚠️ " + msg)

            st_status.update(label="✅ Complete", state="complete", expanded=False)

    # Comparison table
    if S["local_results"] or S["kaggle_results"]:
        st.markdown("---")
        st.subheader("📊 Comparison")
        comp = {}
        if S["local_results"]: comp["Local"] = S["local_results"]
        if S["kaggle_results"]: comp["Kaggle"] = S["kaggle_results"]
        st.dataframe(pd.DataFrame(comp).T)

# ==================== PREDICT ====================
elif S["page"] == "predict":
    st.markdown("### 🔮 Live prediction")
    if S.get("model") is None:
        st.warning("Train a model first on the Train page.")
        st.stop()

    model = S["model"]
    scaler = S["scaler"]
    encoders = S["encodings"] or {}
    ohe_cols = S.get("ohe_cols")
    task = S["task"]
    features = S["feature_cols"]
    final_cols = S.get("final_feature_cols", features)

    with st.form("pred_form"):
        inputs = {}
        cols = st.columns(3)
        for i, c in enumerate(features):
            with cols[i % 3]:
                example_val = DF[c].iloc[0]
                if pd.api.types.is_numeric_dtype(DF[c]):
                    inputs[c] = st.number_input(f"🔢 {c}", value=float(example_val))
                else:
                    options = list(map(str, DF[c].astype(str).dropna().unique()))
                    inputs[c] = st.selectbox(f"🏷️ {c}", options, index=0)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        try:
            new = pd.DataFrame([inputs])

            # Apply same encoding as training
            if ohe_cols is not None:
                new = pd.get_dummies(new, drop_first=False)
                # align to training OHE columns
                new = new.reindex(columns=ohe_cols, fill_value=0)
            elif encoders:
                for c, le in encoders.items():
                    if c in new.columns:
                        # unseen labels → map to -1 safely
                        try:
                            new[c] = le.transform(new[c].astype(str))
                        except Exception:
                            mapped = new[c].astype(str).map({k: int(v) for k, v in zip(le.classes_, range(len(le.classes_)))})
                            new[c] = mapped.fillna(-1)

            # Ensure final column order
            if set(final_cols) == set(new.columns):
                new = new[final_cols]
            else:
                # align just in case
                new = new.reindex(columns=final_cols, fill_value=0)

            # Scale
            if scaler is not None:
                new = pd.DataFrame(scaler.transform(new), columns=new.columns)

            pred = model.predict(new)

            st.markdown("### ✅ Prediction")
            if task == "Classification":
                if S.get("target_encoder") is not None:
                    label = S["target_encoder"].inverse_transform(pred.astype(int))[0]
                else:
                    label = pred[0]
                st.success(f"🎯 Class: **{label}**")
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(new)[0]
                    classes = S.get("target_classes") or list(range(len(proba)))
                    st.dataframe(pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability", ascending=False))
            else:
                st.success(f"📈 Value: **{float(pred[0]):.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
