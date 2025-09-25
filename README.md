# No-Code ML Explorer — Starter


This repo is a starter template for a no-code/low-code ML exploration app using Streamlit and Colab.


## What you get
- Streamlit app (`app.py`) that uploads a CSV, previews data, generates EDA (ydata-profiling), shows quick charts, and provides an "Open in Colab" link.
- A Colab-friendly notebook template (`notebooks/training_template.ipynb`) that runs PyCaret for AutoML.
- An optional GitHub Actions workflow that runs the notebook with `papermill` on a schedule.


## Quick setup (no coding skills required)
1. Create a new GitHub repository and copy all files from this template into your repo.
2. In the repo settings, enable GitHub Pages if you want to publish static outputs (optional).
3. Deploy the Streamlit app:
- Go to https://streamlit.io/cloud and connect your GitHub account.
- Create a new app pointing to the repository and the file `app.py` in the main branch.
4. Upload a CSV using the Streamlit UI.
5. Click **Generate EDA report** to create a profiling HTML report.
6. To train models:
- Click **Generate Open-in-Colab link**.
- Open the notebook in Colab and run cells. Mount your Google Drive to load/save large files.


## To enable automated runs with GitHub Actions
