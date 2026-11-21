import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import pickle
import io

st.set_page_config(page_title="AI Petrophysical Suite", layout="wide")
st.title("AI Petrophysical & Electrofacies Suite")

# ==========================================
# Session State Initialization
# ==========================================
defaults = {
    'master_df': None,
    'core_df': None,
    'track_props': {},
    'ai_summary': None,
    'perm_summary': None,
    'ef_summary': None,
    'layout_props': {
        'fig_height': 9,
        'track_width': 2.5,
        'title_fontsize': 10,
        'show_grid': True
    },
    'cutoff_params': {
        'vsh_max': 0.40,
        'phi_min': 0.10,
        'sw_max': 0.60,
        'k_min': 1.0,
        'tight_k_max': 1.0,
        'min_unit_thickness': 2.0
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def normalize_cols_and_depth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    for candidate in ['MD', 'DEPT', 'DEPTH', 'TVD']:
        if candidate in df.columns:
            df = df.rename(columns={candidate: 'MD'})
            break
    if 'MD' in df.columns:
        df['MD'] = pd.to_numeric(df['MD'], errors='coerce')
        df = df.dropna(subset=['MD'])
    return df


def assign_reservoir_units(df_well: pd.DataFrame, min_thick: float = 2.0) -> pd.Series:
    units = pd.Series("Non-Pay", index=df_well.index, dtype=object)
    if len(df_well) < 2:
        return units
    d_step = float(np.median(np.diff(np.sort(df_well['MD'].values))))
    if d_step <= 0 or np.isnan(d_step):
        d_step = 0.1524

    pay = df_well['PAY_FLAG'].fillna(0).astype(int).values
    unit_id = 0
    in_unit = False
    start_idx = 0
    labels = ["Non-Pay"] * len(df_well)

    for i, p in enumerate(pay):
        if p == 1 and not in_unit:
            in_unit = True
            start_idx = i
        elif p == 0 and in_unit:
            thick = (i - start_idx) * d_step
            if thick >= min_thick:
                unit_id += 1
                for j in range(start_idx, i):
                    labels[j] = f"Unit-{unit_id}"
            in_unit = False

    if in_unit:
        thick = (len(pay) - start_idx) * d_step
        if thick >= min_thick:
            unit_id += 1
            for j in range(start_idx, len(pay)):
                labels[j] = f"Unit-{unit_id}"

    return pd.Series(labels, index=df_well.index)


# ==========================================
# 1. Project Management (.zubair)
# ==========================================
st.sidebar.header("Project Management")

saved_project = st.sidebar.file_uploader("Resume Project (.zubair):", type=["zubair"])
if saved_project is not None:
    if st.sidebar.button("Load Project"):
        project_data = pickle.load(saved_project)
        st.session_state.master_df = project_data.get('master_df')
        st.session_state.core_df = project_data.get('core_df')
        st.session_state.track_props = project_data.get('track_props', {})
        st.session_state.ai_summary = project_data.get('ai_summary')
        st.session_state.perm_summary = project_data.get('perm_summary')
        st.session_state.ef_summary = project_data.get('ef_summary')
        if project_data.get('layout_props'):
            st.session_state.layout_props = project_data.get('layout_props')
        if project_data.get('cutoff_params'):
            st.session_state.cutoff_params = project_data.get('cutoff_params')
        st.sidebar.success("Project restored.")

if st.session_state.master_df is not None:
    project_snapshot = {
        'master_df': st.session_state.master_df,
        'core_df': st.session_state.core_df,
        'track_props': st.session_state.track_props,
        'ai_summary': st.session_state.ai_summary,
        'perm_summary': st.session_state.perm_summary,
        'ef_summary': st.session_state.ef_summary,
        'layout_props': st.session_state.layout_props,
        'cutoff_params': st.session_state.cutoff_params
    }
    st.sidebar.download_button(
        label="Save Project (.zubair)",
        data=pickle.dumps(project_snapshot),
        file_name="Petrophysics_AI_Suite.zubair",
        mime="application/octet-stream"
    )

st.sidebar.markdown("---")
st.sidebar.header("Data Input (LAS & Core)")
uploaded_files = st.sidebar.file_uploader("Upload LAS Files:", type=["las"], accept_multiple_files=True)
core_file = st.sidebar.file_uploader("Core Data File (CSV / XLSX):", type=["csv", "xlsx"])
formation_name = st.sidebar.text_input("Formation Name:", value="Zubair Formation")

if uploaded_files:
    if st.sidebar.button("Load Wells & Core"):
        all_wells_data = []
        for uploaded_file in uploaded_files:
            well_name = uploaded_file.name.split('.')[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                las = lasio.read(tmp_path)
                df = las.df().reset_index()
                df = normalize_cols_and_depth(df)
                df['WELL_NAME'] = well_name
                all_wells_data.append(df)
            except Exception as e:
                st.error(f"Error reading well {well_name}: {e}")

        if all_wells_data:
            new_df = pd.concat(all_wells_data, ignore_index=True)
            for col in ['VSH', 'PHID', 'SW', 'WASHOUT_FLAG', 'CORE_K', 'PERM_AI', 'PERM_UNC', 'TIGHT_FLAG', 'PAY_FLAG', 'ELECTROFACIES']:
                if col not in new_df.columns:
                    new_df[col] = np.nan

            parsed_core = None
            if core_file is not None:
                try:
                    cdf = pd.read_csv(core_file) if core_file.name.endswith('.csv') else pd.read_excel(core_file)
                    cdf = normalize_cols_and_depth(cdf)
                    k_cand = next((c for c in ['CORE_K', 'PERM', 'K', 'PERMEABILITY', 'KH', 'KL'] if c in cdf.columns), None)
                    if k_cand:
                        cdf = cdf.rename(columns={k_cand: 'CORE_K'})
                        cdf['CORE_K'] = pd.to_numeric(cdf['CORE_K'], errors='coerce')
                        cdf = cdf[cdf['CORE_K'] > 0].copy()
                        parsed_core = cdf

                        if 'WELL_NAME' in cdf.columns and 'MD' in cdf.columns:
                            cdf['WELL_NAME'] = cdf['WELL_NAME'].astype(str).str.strip()
                            for w_name, w_group in cdf.groupby('WELL_NAME'):
                                w_idx = new_df['WELL_NAME'] == w_name
                                if w_idx.any() and len(w_group) > 0:
                                    w_sorted = w_group.sort_values('MD')
                                    new_df.loc[w_idx, 'CORE_K'] = np.interp(
                                        new_df.loc[w_idx, 'MD'].values,
                                        w_sorted['MD'].values,
                                        w_sorted['CORE_K'].values,
                                        left=np.nan,
                                        right=np.nan
                                    )
                except Exception as e:
                    st.warning(f"Core file merge warning: {e}")

            st.session_state.master_df = new_df
            st.session_state.core_df = parsed_core
            st.sidebar.success("Datasets loaded.")

# ==========================================
# 2. Control Panel & AI Workflows
# ==========================================
if st.session_state.master_df is not None:
    master_df = st.session_state.master_df
    core_df = st.session_state.core_df

    st.sidebar.markdown("---")
    st.sidebar.header("Well & Depth Interval")

    wells_list = master_df['WELL_NAME'].unique().tolist()
    selected_well = st.sidebar.selectbox("Select Well:", options=wells_list)

    well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
    well_min_d = float(well_raw_df['MD'].min()) if not well_raw_df['MD'].isna().all() else 0.0
    well_max_d = float(well_raw_df['MD'].max()) if not well_raw_df['MD'].isna().all() else 1000.0

    d_col1, d_col2 = st.sidebar.columns(2)
    with d_col1:
        min_depth = st.number_input("Top Depth (m):", value=well_min_d)
    with d_col2:
        max_depth = st.number_input("Base Depth (m):", value=well_max_d)

    # ---- Base Petrophysics ----
    with st.sidebar.expander("1. Base Petrophysics (VSH, PHID, SW)", expanded=False):
        matrix_rho = st.number_input("Matrix Density (g/cc):", value=2.65
