import streamlit as st


if uploaded_file is not None:
# Save uploaded file
file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
with open(file_path, "wb") as f:
f.write(uploaded_file.getbuffer())


st.success(f"Saved to {file_path}")


# Load dataframe
try:
df = pd.read_csv(file_path)
except Exception as e:
st.error(f"Could not read CSV: {e}")
st.stop()


st.header("Preview")
st.write(df.head())
st.write(df.dtypes)


st.header("Generate EDA (ydata-profiling)")
if st.button("Generate EDA report"):
with st.spinner("Generating report — may take a minute for large datasets..."):
profile = ProfileReport(df, title="Dataset profiling report", minimal=False)
out_html = os.path.join(UPLOAD_DIR, f"profile_{uploaded_file.name}.html")
profile.to_file(out_html)
st.success("Report generated")
# Display as HTML
with open(out_html, "r", encoding="utf-8") as f:
html = f.read()
st.components.v1.html(html, height=800, scrolling=True)


st.header("Quick Charts")
cols = df.columns.tolist()
if len(cols) >= 2:
x_col = st.selectbox("X column", cols, index=0)
y_col = st.selectbox("Y column", cols, index=1)
chart_type = st.selectbox("Chart type", ["line", "scatter", "bar"])
if st.button("Plot"):
try:
import plotly.express as px
if chart_type == "line":
fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type} of {y_col} vs {x_col}")
elif chart_type == "scatter":
fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type} of {y_col} vs {x_col}")
else:
fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type} of {y_col} vs {x_col}")
st.plotly_chart(fig, use_container_width=True)
except Exception as e:
st.error(f"Could not plot: {e}")


st.header("Open in Colab — Train Model")
st.markdown("This will open a preconfigured Colab notebook. In Colab you can mount Google Drive and run training using PyCaret.")


# The notebook path in the repo
repo_user = st.text_input("GitHub username or org (where you'll create the repo)", value="your-username")
repo_name = st.text_input("GitHub repo name (create a repo and paste files)", value="ml-no-code-starter")
notebook_path = st.text_input("Path to notebook in repo", value="notebooks/training_template.ipynb")


if st.button("Generate Open-in-Colab link"):
# Save a small parameters JSON to repo-style path for user reference
params_path = os.path.join(UPLOAD_DIR, "last_params.txt")
with open(params_path, "w") as f:
f.write(f"dataset_path: {file_path}\n")
st.info("Saved parameters locally in the app filesystem. To use the notebook from Colab, upload the CSV to your GitHub repo or to your Google Drive and update the notebook file path before running.")


# Create Colab link
colab_url = f"https://colab.research.google.com/github/{repo_user}/{repo_name}/blob/main/{notebook_path}"
st.markdown(f"[Open training notebook in Colab]({colab_url})")
st.write("Note: In Colab, you can mount Google Drive and copy the uploaded CSV there or load it directly from this app's uploads if you make the repo public and push the file.")


else:
st.info("Please upload a CSV to begin. If you want to try a sample dataset, upload one or create a small CSV locally.")
