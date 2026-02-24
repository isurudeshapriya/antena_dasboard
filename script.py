import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Antenna & RRU Dashboard", layout="wide")
st.title("📡 Antenna & RRU Dashboard")

DATA_FOLDER = "saved_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

ANTENNA_FILE = os.path.join(DATA_FOLDER, "antenna.xlsx")
RRU_FILE = os.path.join(DATA_FOLDER, "rru.xlsx")

# =====================================================
# SESSION STATE INIT
# =====================================================
if "antenna_data" not in st.session_state:
    st.session_state.antenna_data = pd.DataFrame()

if "rru_data" not in st.session_state:
    st.session_state.rru_data = pd.DataFrame()

# =====================================================
# LOAD SAVED FILES
# =====================================================
def load_file(path):
    if os.path.exists(path):
        df = pd.read_excel(path)
        df["Count_Start"] = pd.to_numeric(df.get("Count_Start", 0), errors="coerce").fillna(0)
        df["Used_Count"] = pd.to_numeric(df.get("Used_Count", 0), errors="coerce").fillna(0)
        df["Remaning_Count"] = df["Count_Start"] - df["Used_Count"]

        # Ensure unique ID column exists
        if "ID" not in df.columns:
            df.insert(0, "ID", range(1, len(df) + 1))

        return df
    return pd.DataFrame()

if st.session_state.antenna_data.empty:
    st.session_state.antenna_data = load_file(ANTENNA_FILE)

if st.session_state.rru_data.empty:
    st.session_state.rru_data = load_file(RRU_FILE)

# =====================================================
# SAVE FUNCTION
# =====================================================
def save_file(df, path):
    df.to_excel(path, index=False)

# =====================================================
# CIRCULAR CHART
# =====================================================
def circular_progress(value, max_value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": color}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# =====================================================
# TYPE SELECT
# =====================================================
data_type = st.selectbox("Select Type", ["", "Antenna", "RRU"])

if data_type:
    master_df = st.session_state.antenna_data if data_type == "Antenna" else st.session_state.rru_data

    if master_df.empty:
        st.warning("No data available.")
    else:

        # =====================================================
        # FILTER SECTION (SAFE)
        # =====================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            band_filter = st.selectbox("Band",
                                       [""] + sorted(master_df["Bands"].astype(str).unique()))

        with col2:
            project_filter = st.selectbox("Project",
                                          [""] + sorted(master_df["Project"].astype(str).unique()))

        with col3:
            batch_filter = st.selectbox("Batch",
                                        [""] + sorted(master_df["Batch"].astype(str).unique()))

        filtered_df = master_df.copy()

        if band_filter:
            filtered_df = filtered_df[filtered_df["Bands"].astype(str) == band_filter]

        if project_filter:
            filtered_df = filtered_df[filtered_df["Project"].astype(str) == project_filter]

        if batch_filter:
            filtered_df = filtered_df[filtered_df["Batch"].astype(str) == batch_filter]

        # =====================================================
        # SAFE EDITOR (EDIT MASTER, NOT FILTERED)
        # =====================================================
        st.subheader("Editable Table")

        edited_master = st.data_editor(
            master_df,
            use_container_width=True,
            key=f"editor_{data_type}"
        )

        # Recalculate remaining
        edited_master["Count_Start"] = pd.to_numeric(
            edited_master["Count_Start"], errors="coerce").fillna(0)

        edited_master["Used_Count"] = pd.to_numeric(
            edited_master["Used_Count"], errors="coerce").fillna(0)

        edited_master["Remaning_Count"] = (
            edited_master["Count_Start"] - edited_master["Used_Count"]
        )

        # Save safely
        if data_type == "Antenna":
            st.session_state.antenna_data = edited_master
            save_file(edited_master, ANTENNA_FILE)
        else:
            st.session_state.rru_data = edited_master
            save_file(edited_master, RRU_FILE)

        # =====================================================
        # FILTERED VIEW (READ ONLY)
        # =====================================================
        st.subheader("Filtered View")
        st.dataframe(filtered_df, use_container_width=True)

        # =====================================================
        # SUMMARY
        # =====================================================
        total_start = int(filtered_df["Count_Start"].sum())
        total_remaining = int(filtered_df["Remaning_Count"].sum())

        st.subheader("Summary")

        colA, colB = st.columns(2)

        with colA:
            st.plotly_chart(
                circular_progress(total_remaining, max(1, total_start), "#10b981"),
                use_container_width=True
            )

        with colB:
            st.metric("Total Start", total_start)
            st.metric("Total Remaining", total_remaining)

# =====================================================
# EXPORT
# =====================================================
st.divider()

if st.button("Export Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if not st.session_state.antenna_data.empty:
            st.session_state.antenna_data.to_excel(writer, sheet_name="Antenna", index=False)
        if not st.session_state.rru_data.empty:
            st.session_state.rru_data.to_excel(writer, sheet_name="RRU", index=False)

    st.download_button(
        "Download File",
        data=output.getvalue(),
        file_name="Dashboard_Updated.xlsx"
    )

# =====================================================
# UPLOAD
# =====================================================
st.divider()
st.subheader("Upload Files")

colU1, colU2 = st.columns(2)

with colU1:
    fileA = st.file_uploader("Upload Antenna Excel", type=["xlsx"])
    if fileA:
        df = pd.read_excel(fileA)
        df.insert(0, "ID", range(1, len(df) + 1))
        st.session_state.antenna_data = df
        save_file(df, ANTENNA_FILE)
        st.success("Antenna Uploaded")
        st.rerun()

with colU2:
    fileR = st.file_uploader("Upload RRU Excel", type=["xlsx"])
    if fileR:
        df = pd.read_excel(fileR)
        df.insert(0, "ID", range(1, len(df) + 1))
        st.session_state.rru_data = df
        save_file(df, RRU_FILE)
        st.success("RRU Uploaded")
        st.rerun()

   
