import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

st.set_page_config(page_title="No-Code ML Explorer", layout="wide")

st.title("🚀 No-Code ML Dataset Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Data Preview")
    st.write(df.head())

    # Profiling Report
    st.subheader("📊 Automated EDA Report")
    profile = ProfileReport(df, explorative=True)
    profile.to_file("report.html")
    with open("report.html", "r", encoding="utf-8") as f:
        html(f.read(), height=600, scrolling=True)

    # Interactive chart
    st.subheader("📈 Quick Chart")
    cols = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", options=cols)
    y_axis = st.selectbox("Select Y-axis", options=cols)
    if st.button("Plot Chart"):
        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)

    # Colab training button
    st.subheader("🤖 Train Model")
    st.markdown(
        """
        [Open Training Notebook in Colab](https://colab.research.google.com/github/<YOUR_USERNAME>/<YOUR_REPO>/blob/main/notebooks/training_template.ipynb)
        """
    )
else:
    st.info("👆 Please upload a CSV file to get started.")
