# AGENTS.md — working notes for this repo

## What the app is

Single-process Streamlit app (`app.py`, ~2500 lines) — AutoMLPilot Pro, a no-code ML
workbench. There is **no database, no API service and no frontend build step**: the
whole app is one Python script served by Streamlit.

## Running it in the Base44 sandbox

```bash
docker compose -f docker-compose.base44.yml up -d --build   # first run / after dependency changes
docker compose -f docker-compose.base44.yml up -d            # normal restart
docker compose -f docker-compose.base44.yml logs -f app
```

- The preview entry point is **http://localhost:3000** (Streamlit listens on 8501 inside
  the container).
- `Dockerfile.base44` installs only the dependencies; the source is bind-mounted and the
  app is started with `streamlit run app.py`, so source edits hot-apply (Streamlit
  `runOnSave` + `fileWatcherType poll`, because bind mounts do not deliver inotify events
  reliably).
- Health endpoint: `/_stcore/health` (returns `ok`).
- `torch`/`torchvision` are installed from the **CPU-only** PyTorch index; the default PyPI
  wheels would pull several GB of CUDA runtime that this app never uses (it only runs
  `distilgpt2` locally, CPU).

## Sandbox-only server flags

The compose `command` adds `--server.enableCORS false --server.enableXsrfProtection false`
**only when `BASE44_PREVIEW_MODE=1`**. The hosted preview reaches the container through a
proxy, so the browser's `Origin` host differs from the `Host` header the container sees and
Streamlit rejects the websocket with
`Rejecting WebSocket connection with disallowed Origin or Host header` (the UI then hangs on
"Connecting…"). This mirrors the flags the repo's own `.devcontainer/devcontainer.json`
already uses for local dev. With the variable unset, Streamlit keeps its stock CORS/XSRF
protections.

## Credentials

The app reads `SMTP_SENDER_EMAIL` / `SMTP_APP_PASSWORD` / `SMTP_SENDER_NAME` /
`SMTP_SENDER_ALIAS` from `st.secrets`, i.e. from the committed `.streamlit/secrets.toml`.
Without them the email feature is simply disabled (`ENABLE_EMAIL = False`) — the app still
boots. Streamlit also falls back to environment variables for secrets, but a key present in
`secrets.toml` wins over the environment.

## Behaviour worth knowing

- First page load takes ~20–30 s: the module-level imports (torch, transformers, xgboost,
  ydata-profiling, …) dominate.
- Optional features are feature-flagged by import success: `YDATA_OK`, `XGB_OK`, `IMB_OK`,
  `PYCARET_OK`, `TRANSFORMERS_OK`. `imbalanced-learn` and `pycaret` are **not** in
  `requirements.txt`; missing ones degrade gracefully instead of crashing.
- The "AI Smart Insights" / chat features download `distilgpt2` from Hugging Face on first
  use (network access required).
- Importing `pycaret` writes a `logs.log` into the working directory (pycaret hardcodes that
  filename, no env override). It is a runtime artifact and is gitignored.

## Tests

```bash
docker compose -f docker-compose.base44.yml exec -T app python -m unittest tests.test_correlation -v
```

`tests/test_correlation.py` mocks every third-party library (pandas, numpy, streamlit,
sklearn, plotly …) before importing `app.py`, so it runs fast and does not exercise the real
stack. The CI workflow in `.github/workflows/` only runs a papermill notebook and is
unrelated to the app.
