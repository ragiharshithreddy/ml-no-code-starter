# 🤖 AutoML Pilot

**AutoML Pilot** is an interactive, no-code machine learning web application built with [Streamlit](https://streamlit.io/). It empowers data scientists, analysts, and beginners alike to upload a dataset and automatically preprocess, train, compare, and export machine learning models — all without writing a single line of code.

🔗 **Live App:** [automlpilot.streamlit.app](https://automlpilot.streamlit.app/)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run Locally](#run-locally)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧭 Overview

AutoML Pilot automates the most time-consuming parts of a machine learning workflow. Simply upload your CSV dataset, select your target variable, and let the app handle everything — from data cleaning and feature engineering to model training, evaluation, and download.

Whether you are exploring machine learning for the first time or need a fast prototyping tool, AutoML Pilot gets you from raw data to a trained model in minutes.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Data Upload** | Upload any CSV file directly through the browser |
| 🔍 **Automated EDA** | Instant exploratory data analysis with visual summaries |
| 🧹 **Data Preprocessing** | Handles missing values, encoding, and scaling automatically |
| 🤖 **Model Training** | Trains and compares multiple ML algorithms in one click |
| 📊 **Model Comparison** | Side-by-side performance metrics to pick the best model |
| 💾 **Model Export** | Download your trained model as a `.pkl` file for deployment |
| 🌐 **No-Code Interface** | Fully interactive UI — no programming knowledge required |

---

## 🎬 Demo

> Try the live app here: **[https://automlpilot.streamlit.app](https://automlpilot.streamlit.app/)**

1. Upload your CSV dataset
2. Select your target column
3. Choose the task type (Classification or Regression)
4. Click **Run AutoML**
5. Compare model results and download your best model

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web application framework |
| [PyCaret](https://pycaret.org/) | Automated machine learning (AutoML) |
| [Pandas](https://pandas.pydata.org/) | Data manipulation and analysis |
| [Pandas Profiling](https://github.com/ydataai/ydata-profiling) | Automated exploratory data analysis |
| Python 3.9+ | Core programming language |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.9 or higher
- `pip` or `conda` package manager

### Installation

**Option 1 — Using Conda (Recommended)**

```bash
# Create a new virtual environment
conda create -n automlpilot python=3.9

# Activate the environment
conda activate automlpilot

# Install all dependencies
pip install -r requirements.txt
```

**Option 2 — Using pip**

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501l

## ⚙️ How It Works

AutoML Pilot follows a straightforward four-step pipeline:

**1. Upload & Preview**
Upload a CSV file and instantly preview your data along with a detailed profiling report showing distributions, missing values, correlations, and more.

**2. Preprocess**
The app automatically handles common data quality issues — null values are imputed, categorical columns are encoded, and numeric features are scaled, all behind the scenes.

**3. Train & Compare**
PyCaret trains multiple machine learning models simultaneously and ranks them by performance. Metrics like Accuracy, AUC, F1-Score, MAE, and RMSE are displayed in a clean comparison table.

**4. Export**
Select your best-performing model and download it as a `.pkl` file, ready to use in any Python environment or production pipeline.

---

## 🤝 Contributing

Contributions are welcome and appreciated! 
## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ using [Streamlit](https://streamlit.io/) &amp; [PyCaret](https://pycaret.org/)

⭐ If you find this project useful, give it a star!

</div>
