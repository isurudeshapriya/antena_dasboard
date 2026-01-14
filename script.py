import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import os

# --- Config ---
st.set_page_config(page_title="Antenna & RRU Dashboard", layout="wide")
st.title("📡 Antenna & RRU Dashboard")

# --- Folder to store data ---
DATA_FOLDER = "saved_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# --- Session state for storing data ---
if "antenna_data" not in st.session_state:
    st.session_state.antenna_data = pd.DataFrame()
if "rru_data" not in st.session_state:
    st.session_state.rru_data = pd.DataFrame()
if "selected_type" not in st.session_state:
    st.session_state.selected_type = ""
if "selected_band" not in st.session_state:
    st.session_state.selected_band = ""
if "selected_project" not in st.session_state:
    st.session_state.selected_project = ""
if "selected_batch" not in st.session_state:
    st.session_state.selected_batch = ""
if "selected_band_count" not in st.session_state:
    st.session_state.selected_band_count = ""

# --- Load previous data if exists ---
antenna_file_path = os.path.join(DATA_FOLDER, "antenna.xlsx")
rru_file_path = os.path.join(DATA_FOLDER, "rru.xlsx")

def load_saved_data():
    if os.path.exists(antenna_file_path) and st.session_state.antenna_data.empty:
        df = pd.read_excel(antenna_file_path)
        df["Remaning_Count"] = df.get("Count_Start", 0) - df.get("Used_Count", 0)
        st.session_state.antenna_data = df
    if os.path.exists(rru_file_path) and st.session_state.rru_data.empty:
        df = pd.read_excel(rru_file_path)
        df["Remaning_Count"] = df.get("Count_Start", 0) - df.get("Used_Count", 0)
        st.session_state.rru_data = df

load_saved_data()

# --- File Upload ---
def handle_file_upload(uploaded_file, data_type):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df["Remaning_Count"] = df.get("Count_Start", 0) - df.get("Used_Count", 0)
        if data_type == "antenna":
            st.session_state.antenna_data = df
            df.to_excel(antenna_file_path, index=False)
        else:
            st.session_state.rru_data = df
            df.to_excel(rru_file_path, index=False)

st.file_uploader("Upload Antenna Excel File:", type=["xlsx"], key="antenna_file",
                 on_change=lambda: handle_file_upload(st.session_state.antenna_file, "antenna"))
st.file_uploader("Upload RRU Excel File:", type=["xlsx"], key="rru_file",
                 on_change=lambda: handle_file_upload(st.session_state.rru_file, "rru"))

# --- Clear Data ---
if st.button("🗑️ Clear All Data"):
    st.session_state.antenna_data = pd.DataFrame()
    st.session_state.rru_data = pd.DataFrame()
    st.session_state.selected_type = ""
    st.session_state.selected_band = ""
    st.session_state.selected_project = ""
    st.session_state.selected_batch = ""
    st.session_state.selected_band_count = ""
    for file in ["antenna.xlsx", "rru.xlsx"]:
        path = os.path.join(DATA_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)
    st.success("All data cleared!")

# --- Type Selection ---
types = []
if not st.session_state.antenna_data.empty:
    types.append("Antenna")
if not st.session_state.rru_data.empty:
    types.append("RRU")

st.session_state.selected_type = st.selectbox("Select Type:", [""] + types)

# --- Filter options ---
def get_filtered_data(df):
    data = df.copy()
    if st.session_state.selected_band:
        data = data[data["Bands"].astype(str) == st.session_state.selected_band]
    if st.session_state.selected_project:
        data = data[data["Project"].astype(str) == st.session_state.selected_project]
    if st.session_state.selected_batch:
        data = data[data["Batch"].astype(str) == st.session_state.selected_batch]
    return data

if st.session_state.selected_type:
    df = st.session_state.antenna_data if st.session_state.selected_type == "Antenna" else st.session_state.rru_data
    bands = [""] + sorted(df["Bands"].dropna().astype(str).unique().tolist())
    projects = [""] + sorted(df["Project"].dropna().astype(str).unique().tolist())
    batches = [""] + sorted(df["Batch"].dropna().astype(str).unique().tolist())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.selected_band = st.selectbox("Select Band:", bands)
    with col2:
        st.session_state.selected_project = st.selectbox("Select Project:", projects)
    with col3:
        st.session_state.selected_batch = st.selectbox("Select Batch:", batches)
    with col4:
        st.session_state.selected_band_count = st.selectbox("Band-wise Count:", bands)

# --- Update Remaining Count ---
def update_remaining_count(df):
    if not df.empty:
        df["Remaning_Count"] = df.get("Count_Start", 0) - df.get("Used_Count", 0)
    return df

# --- Model-wise counts ---
def get_model_counts(df):
    if df.empty:
        return pd.DataFrame()
    df_model = df.copy()
    df_model["Model"] = df_model.get("Model", df_model.get("Type", "Not Specified"))
    summary = df_model.groupby(["Model","Project"]).agg(
        totalStart=("Count_Start","sum"),
        totalRemaining=("Remaning_Count","sum")
    ).reset_index()
    return summary

# --- Horizontal Bar Chart for Batch Summary ---
def horizontal_bar_chart(model_summary, color="#636efa"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_summary["totalRemaining"],
        y=model_summary["Model"],
        orientation='h',
        text=model_summary["totalRemaining"],
        textposition="auto",
        marker_color=color
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20))
    return fig

# --- Display batch summary ---
if st.session_state.selected_batch and st.session_state.selected_type:
    filtered_df = get_filtered_data(df)
    filtered_df = update_remaining_count(filtered_df)

    model_summary = filtered_df.groupby(["Model", "Project"]).agg(
        totalStart=("Count_Start","sum"),
        totalRemaining=("Remaning_Count","sum")
    ).reset_index()

    st.subheader(f"Batch {st.session_state.selected_batch} - {st.session_state.selected_type} Summary")

    for idx, row in model_summary.iterrows():
        st.markdown("---")
        st.markdown(f"### Model: {row['Model']}  | Project: {row['Project']}")
        st.plotly_chart(horizontal_bar_chart(pd.DataFrame([row]), color="#8b5cf6"), use_container_width=True)
        st.markdown(f"**Start:** {row['totalStart']}  | **Remaining:** {row['totalRemaining']}")

# --- Display editable table ---
def display_table(df, data_type):
    if df.empty:
        st.info("No data to display")
        return df
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    # Save edits automatically
    if edited_df is not None:
        edited_df = update_remaining_count(edited_df)
        if data_type == "Antenna":
            st.session_state.antenna_data.update(edited_df)
            st.session_state.antenna_data.to_excel(antenna_file_path, index=False)
        else:
            st.session_state.rru_data.update(edited_df)
            st.session_state.rru_data.to_excel(rru_file_path, index=False)
    return edited_df

if st.session_state.selected_type:
    df_to_show = get_filtered_data(df)
    st.subheader(f"{st.session_state.selected_type} Data Table")
    display_table(df_to_show, st.session_state.selected_type)

# --- Total Counts ---
def get_total_counts(df, band=""):
    if df.empty:
        return 0,0
    data = df.copy()
    if band:
        data = data[data["Bands"].astype(str) == band]
    total_start = data["Count_Start"].sum()
    total_remaining = data["Remaning_Count"].sum()
    return total_start, total_remaining

antenna_start, antenna_remaining = get_total_counts(st.session_state.antenna_data, st.session_state.selected_band_count)
rru_start, rru_remaining = get_total_counts(st.session_state.rru_data, st.session_state.selected_band_count)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(horizontal_bar_chart(pd.DataFrame([{"Model":"Antenna","totalRemaining":antenna_remaining}]), color="#10b981"))
    st.markdown(f"**Antenna Total Start:** {antenna_start}")
with col2:
    st.plotly_chart(horizontal_bar_chart(pd.DataFrame([{"Model":"RRU","totalRemaining":rru_remaining}]), color="#3b82f6"))
    st.markdown(f"**RRU Total Start:** {rru_start}")

# --- Export to Excel ---
def to_excel_bytes(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, data in df_dict.items():
            data.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()

if st.button("📥 Export Updated Excel Files"):
    dfs = {}
    if not st.session_state.antenna_data.empty:
        dfs["Antenna"] = st.session_state.antenna_data
    if not st.session_state.rru_data.empty:
        dfs["RRU"] = st.session_state.rru_data
    if dfs:
        st.download_button("Download Excel", data=to_excel_bytes(dfs), file_name="Dashboard_Updated.xlsx")
