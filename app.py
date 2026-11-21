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
                                        left=np.nan, right=np.nan
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
        matrix_rho = st.number_input("Matrix Density (g/cc):", value=2.65, step=0.01)
        fluid_rho = st.number_input("Fluid Density (g/cc):", value=1.00, step=0.05)
        rw_val = st.number_input("Formation Water Resistivity (Rw):", value=0.03, step=0.01)

        if st.button("Compute VSH, PHID, SW"):
            if 'GR' in master_df.columns:
                gr_clean = master_df['GR'].quantile(0.05)
                gr_shale = master_df['GR'].quantile(0.95)
                if gr_shale != gr_clean:
                    igr = (master_df['GR'] - gr_clean) / (gr_shale - gr_clean)
                    master_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            if 'RHOB' in master_df.columns and (matrix_rho - fluid_rho) != 0:
                master_df['PHID'] = ((matrix_rho - master_df['RHOB']) / (matrix_rho - fluid_rho)).clip(0, 0.45)

            res_col = next((r for r in ['ILD', 'LLD', 'RT', 'RILD', 'RESD'] if r in master_df.columns), None)
            if res_col and 'PHID' in master_df.columns:
                phid_safe = master_df['PHID'].replace(0, 0.001)
                rt_safe = master_df[res_col].replace(0, 1)
                sw = np.power((1.0 * rw_val) / (np.power(phid_safe, 2.0) * rt_safe), 0.5)
                master_df['SW'] = sw.clip(0, 1)

            st.session_state.master_df = master_df
            st.success("Base properties updated.")

    # ---- Stage 1: Log Prediction & Washout Correction ----
    with st.sidebar.expander("2. Stage 1: Log Prediction & Washout", expanded=True):
        all_available_now = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT', 'ELECTROFACIES']]
        target_log = st.selectbox("Target Log:", options=all_available_now)

        default_preds = [c for c in ['GR', 'NPHI', 'DT', 'RHOB', 'VSH'] if c in all_available_now and c != target_log]
        predictor_logs = st.multiselect(
            "Predictor Logs:",
            options=[c for c in all_available_now if c != target_log],
            default=default_preds
        )

        ai_algo_s1 = st.selectbox("Model (Stage 1):", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="algo_s1")
        cali_col = st.selectbox("Caliper Log (Washout):", options=["None"] + all_available_now)
        bit_size = st.number_input("Bit Size:", value=8.5, step=0.5)
        washout_delta = st.number_input("Washout Threshold (dCali):", value=1.5, step=0.25)

        replace_mode = st.selectbox("Merge Mode:", options=[
            "Smart Fill (Missing & Washout)",
            "Full Synthetic"
        ])

        if st.button("Run Stage 1"):
            if len(predictor_logs) < 1:
                st.error("Select at least one predictor log.")
            else:
                if cali_col != "None":
                    washout_mask = (master_df[cali_col] - bit_size) > washout_delta
                else:
                    washout_mask = pd.Series(False, index=master_df.index)
                master_df['WASHOUT_FLAG'] = washout_mask.astype(int)

                train_mask = (~washout_mask) & master_df[target_log].notna() & master_df[predictor_logs].notna().all(axis=1)
                pred_mask = master_df[predictor_logs].notna().all(axis=1)

                X_train = master_df.loc[train_mask, predictor_logs]
                y_train = master_df.loc[train_mask, target_log]
                X_pred = master_df.loc[pred_mask, predictor_logs]

                if len(X_train) < 20 or len(X_pred) == 0:
                    st.error("Insufficient training or input samples.")
                else:
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_pred_sc = scaler.transform(X_pred)

                    if ai_algo_s1 == "Random Forest Regressor":
                        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                        model.fit(X_train_sc, y_train)
                        y_hat = model.predict(X_pred_sc)
                        tree_preds = np.array([est.predict(X_pred_sc) for est in model.estimators_])
                        unc_vals = np.std(tree_preds, axis=0)
                    else:
                        model = GradientBoostingRegressor(n_estimators=150, random_state=42)
                        model.fit(X_train_sc, y_train)
                        y_hat = model.predict(X_pred_sc)
                        train_res_std = float(np.std(y_train - model.predict(X_train_sc)))
                        unc_vals = np.full(len(y_hat), train_res_std)

                    ai_col_name = f"{target_log}_AI"
                    unc_col_name = f"{target_log}_UNC"
                    merged_col_name = f"{target_log}_MERGED"

                    master_df[ai_col_name] = np.nan
                    master_df[unc_col_name] = np.nan
                    master_df.loc[pred_mask, ai_col_name] = y_hat
                    master_df.loc[pred_mask, unc_col_name] = unc_vals

                    if replace_mode == "Full Synthetic":
                        master_df[merged_col_name] = master_df[ai_col_name]
                    else:
                        needs_fix = master_df[target_log].isna() | washout_mask
                        master_df[merged_col_name] = np.where(needs_fix & master_df[ai_col_name].notna(), master_df[ai_col_name], master_df[target_log])

                    r2_train = model.score(X_train_sc, y_train)
                    st.session_state.master_df = master_df
                    st.session_state.ai_summary = {
                        'formation': formation_name,
                        'target': target_log,
                        'algo': ai_algo_s1,
                        'r2_train': round(float(r2_train), 4),
                        'washout_count': int(washout_mask.sum()),
                        'filled_count': int((master_df[target_log].isna() | washout_mask).sum()),
                        'mean_unc': round(float(np.nanmean(unc_vals)), 4)
                    }
                    st.success("Stage 1 completed.")

    # ---- Stage 2: Core Permeability, Cut-offs & Reservoir Units ----
    with st.sidebar.expander("3. Stage 2: Core Perm, Cut-offs & Units", expanded=True):
        train_source_mode = st.selectbox(
            "Core Training Source:",
            options=[
                "Core File (Reference Core Wells)",
                "Embedded Core Column in Loaded Wells"
            ]
        )

        available_las_feats = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT', 'ELECTROFACIES', 'PERM_AI', 'PERM_UNC'] and pd.api.types.is_numeric_dtype(master_df[c])]

        if train_source_mode == "Core File (Reference Core Wells)" and core_df is not None:
            core_num_cols = [c for c in core_df.columns if c not in ['MD', 'CORE_K'] and pd.api.types.is_numeric_dtype(core_df[c])]
            shared_feats = [c for c in core_num_cols if c in available_las_feats]
            feat_options = shared_feats if len(shared_feats) > 0 else available_las_feats
        else:
            feat_options = [c for c in available_las_feats if c != 'CORE_K']

        default_perm_feats = [c for c in ['PHID', 'VSH', 'GR', 'NPHI', 'RHOB'] if c in feat_options]
        perm_features = st.multiselect("Predictor Features:", options=feat_options, default=default_perm_feats)
        use_rolling_ctx = st.checkbox("Rolling Depth Context", value=True)
        rolling_win = st.slider("Rolling Window (pts):", min_value=3, max_value=21, value=7, step=2)
        ai_algo_s2 = st.selectbox("Model (Stage 2):", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="algo_s2")

        cp = st.session_state.cutoff_params
        vsh_max_in = st.number_input("Max VSH:", value=float(cp['vsh_max']), min_value=0.0, max_value=1.0, step=0.02)
        phi_min_in = st.number_input("Min PHID:", value=float(cp['phi_min']), min_value=0.0, max_value=0.5, step=0.01)
        sw_max_in = st.number_input("Max SW:", value=float(cp['sw_max']), min_value=0.0, max_value=1.0, step=0.02)
        k_min_in = st.number_input("Min Perm (mD):", value=float(cp['k_min']), min_value=0.001, step=0.5)
        tight_k_in = st.number_input("Tight Perm Max (mD):", value=float(cp['tight_k_max']), min_value=0.001, step=0.2)
        min_thick_in = st.number_input("Min Unit Thickness (m):", value=float(cp['min_unit_thickness']), min_value=0.5, step=0.5)

        st.session_state.cutoff_params = {
            'vsh_max': vsh_max_in,
            'phi_min': phi_min_in,
            'sw_max': sw_max_in,
            'k_min': k_min_in,
            'tight_k_max': tight_k_in,
            'min_unit_thickness': min_thick_in
        }

        if st.button("Run Stage 2"):
            if len(perm_features) < 1:
                st.error("Select at least one predictor feature.")
            else:
                work_df = master_df.copy()
                las_feat_cols = list(perm_features)

                if use_rolling_ctx:
                    for fc in perm_features:
                        rm_col = f"{fc}_RM{rolling_win}"
                        rs_col = f"{fc}_RS{rolling_win}"
                        work_df[rm_col] = work_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).mean())
                        work_df[rs_col] = work_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).std().fillna(0))
                        las_feat_cols.extend([rm_col, rs_col])

                if train_source_mode == "Core File (Reference Core Wells)":
                    if core_df is None or 'CORE_K' not in core_df.columns:
                        st.error("Upload a core dataset with permeability first.")
                        st.stop()
                    c_train_df = core_df.copy()
                    train_feat_cols = []
                    for fc in perm_features:
                        if fc in c_train_df.columns:
                            train_feat_cols.append(fc)
                            if use_rolling_ctx:
                                rm_c = f"{fc}_RM{rolling_win}"
                                rs_c = f"{fc}_RS{rolling_win}"
                                if 'WELL_NAME' in c_train_df.columns:
                                    c_train_df[rm_c] = c_train_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).mean())
                                    c_train_df[rs_c] = c_train_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).std().fillna(0))
                                else:
                                    c_train_df[rm_c] = c_train_df[fc].rolling(rolling_win, center=True, min_periods=1).mean()
                                    c_train_df[rs_c] = c_train_df[fc].rolling(rolling_win, center=True, min_periods=1).std().fillna(0)
                                train_feat_cols.extend([rm_c, rs_c])

                    if len(train_feat_cols) != len(las_feat_cols):
                        st.error("Selected features must exist in both core and LAS datasets.")
                        st.stop()

                    ck_vals = pd.to_numeric(c_train_df['CORE_K'], errors='coerce')
                    tr_mask = (ck_vals > 0) & c_train_df[train_feat_cols].notna().all(axis=1)
                    X_tr = c_train_df.loc[tr_mask, train_feat_cols]
                    y_tr = np.log10(ck_vals.loc[tr_mask].clip(lower=1e-4))
                    train_label = "Core File (Reference Wells)"
                else:
                    k_vals = pd.to_numeric(work_df['CORE_K'], errors='coerce')
                    tr_mask = (k_vals > 0) & work_df[las_feat_cols].notna().all(axis=1)
                    X_tr = work_df.loc[tr_mask, las_feat_cols]
                    y_tr = np.log10(k_vals.loc[tr_mask].clip(lower=1e-4))
                    train_label = "Embedded Core Column"

                pr_mask = work_df[las_feat_cols].notna().all(axis=1)
                X_pr = work_df.loc[pr_mask, las_feat_cols]

                if len(X_tr) < 5 or len(X_pr) == 0:
                    st.error("Insufficient core training samples or input features.")
                else:
                    scaler_k = StandardScaler()
                    X_tr_sc = scaler_k.fit_transform(X_tr)
                    X_pr_sc = scaler_k.transform(X_pr)

                    if ai_algo_s2 == "Random Forest Regressor":
                        k_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                        k_model.fit(X_tr_sc, y_tr)
                        log_k_hat = k_model.predict(X_pr_sc)
                        t_preds = np.array([est.predict(X_pr_sc) for est in k_model.estimators_])
                        log_k_std = np.std(t_preds, axis=0)
                    else:
                        k_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
                        k_model.fit(X_tr_sc, y_tr)
                        log_k_hat = k_model.predict(X_pr_sc)
                        res_std = float(np.std(y_tr - k_model.predict(X_tr_sc)))
                        log_k_std = np.full(len(log_k_hat), res_std)

                    k_pred_md = np.power(10.0, log_k_hat)
                    k_unc_md = np.power(10.0, log_k_hat + log_k_std) - np.power(10.0, log_k_hat - log_k_std)

                    work_df['PERM_AI'] = np.nan
                    work_df['PERM_UNC'] = np.nan
                    work_df.loc[pr_mask, 'PERM_AI'] = k_pred_md
                    work_df.loc[pr_mask, 'PERM_UNC'] = k_unc_md

                    work_df['TIGHT_FLAG'] = np.where(work_df['PERM_AI'].notna() & (work_df['PERM_AI'] <= tight_k_in), 1, 0)

                    vsh_ok = work_df['VSH'].fillna(0) <= vsh_max_in if 'VSH' in work_df.columns else pd.Series(True, index=work_df.index)
                    phi_ok = work_df['PHID'].fillna(0) >= phi_min_in if 'PHID' in work_df.columns else pd.Series(True, index=work_df.index)
                    sw_ok = work_df['SW'].fillna(0) <= sw_max_in if 'SW' in work_df.columns else pd.Series(True, index=work_df.index)
                    k_ok = work_df['PERM_AI'].fillna(0) >= k_min_in

                    work_df['PAY_FLAG'] = (vsh_ok & phi_ok & sw_ok & k_ok).astype(int)

                    work_df['RES_UNIT'] = "Non-Pay"
                    for w_name in work_df['WELL_NAME'].unique():
                        w_sub = work_df[work_df['WELL_NAME'] == w_name].sort_values('MD')
                        work_df.loc[w_sub.index, 'RES_UNIT'] = assign_reservoir_units(w_sub, min_thick=min_thick_in)

                    r2_k = k_model.score(X_tr_sc, y_tr)
                    st.session_state.master_df = work_df
                    st.session_state.perm_summary = {
                        'formation': formation_name,
                        'train_source': train_label,
                        'core_samples': int(len(X_tr)),
                        'algo': ai_algo_s2,
                        'r2_logk': round(float(r2_k), 4),
                        'tight_count': int(work_df['TIGHT_FLAG'].sum()),
                        'pay_count': int(work_df['PAY_FLAG'].sum()),
                        'units_found': int(work_df[work_df['RES_UNIT'] != 'Non-Pay']['RES_UNIT'].nunique())
                    }
                    st.success("Stage 2 completed.")

# ==========================================
# 3. Main Workspace Tabs
# ==========================================
if st.session_state.master_df is not None:
    master_df = st.session_state.master_df
    well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
    single_well_df = well_raw_df[(well_raw_df['MD'] >= min_depth) & (well_raw_df['MD'] <= max_depth)].sort_values(by='MD')
    all_plot_logs = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT', 'ELECTROFACIES'] and pd.api.types.is_numeric_dtype(master_df[c])]

    tab_ef, tab_s2, tab_ai, tab_tracks, tab_props, tab_export = st.tabs([
        "Electrofacies",
        "Core Perm & Units (Stage 2)",
        "Log Prediction & Uncertainty (Stage 1)",
        "Log Tracks",
        "Properties",
        "Export"
    ])

    with tab_ef:
        st.subheader(f"Electrofacies Classification — {selected_well} ({formation_name})")
        ef_c1, ef_c2 = st.columns([2, 1])
        with ef_c1:
            default_ef_logs = [c for c in ['GR', 'VSH', 'PHID', 'RHOB', 'NPHI', 'DT', 'PERM_AI'] if c in all_plot_logs]
            ef_features = st.multiselect("Electrofacies Input Logs:", options=all_plot_logs, default=default_ef_logs, key="ef_feats")
        with ef_c2:
            n_clusters = st.slider("Number of Clusters (EF):", min_value=2, max_value=8, value=5, key="ef_k")

        if st.button("Run Electrofacies Clustering"):
            if len(ef_features) < 2:
                st.error("Select at least 2 logs for clustering.")
            else:
                ef_df = master_df.copy()
                valid_ef = ef_df[ef_features].notna().all(axis=1)
                if valid_ef.sum() < n_clusters * 2:
                    st.error("Insufficient complete rows for Electrofacies clustering.")
                else:
                    sc_ef = StandardScaler()
                    X_ef = sc_ef.fit_transform(ef_df.loc[valid_ef, ef_features])
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels_ef = km.fit_predict(X_ef) + 1
                    ef_df['ELECTROFACIES'] = np.nan
                    ef_df.loc[valid_ef, 'ELECTROFACIES'] = labels_ef
                    st.session_state.master_df = ef_df
                    st.session_state.ef_summary = {
                        'n_clusters': int(n_clusters),
                        'features': ef_features,
                        'classified_pts': int(valid_ef.sum())
                    }
                    st.success("Electrofacies clustering completed.")

        if 'ELECTROFACIES' in single_well_df.columns and single_well_df['ELECTROFACIES'].notna().any() and len(single_well_df) > 0:
            fig_ef, ax_ef = plt.subplots(1, 1, figsize=(track_width * 1.4, fig_height))
            d_ef = single_well_df['MD'].values
            ef_vals = single_well_df['ELECTROFACIES'].fillna(0).values
            cmap = plt.get_cmap('tab10', max(int(ef_vals.max()) + 1, 2))
            for k_id in sorted(np.unique(ef_vals)):
                if k_id <= 0:
                    continue
                mask_k = ef_vals == k_id
                ax_ef.fill_betweenx(d_ef, 0, 1, where=mask_k, color=cmap(int(k_id) - 1), alpha=0.85, step="mid", label=f"EF-{int(k_id)}")
            ax_ef.set_xlim(0, 1)
            ax_ef.set_xticks([])
            ax_ef.set_ylim(d_ef.max(), d_ef.min())
            ax_ef.set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
            ax_ef.set_title("Electrofacies", fontsize=title_fontsize, fontweight="bold")
            ax_ef.legend(loc="best", fontsize=8)
            if show_grid:
                ax_ef.grid(True, axis="y", linestyle="--", alpha=0.4)

            plt.tight_layout()
            buf_ef = io.BytesIO()
            fig_ef.savefig(buf_ef, format="png", dpi=300, bbox_inches="tight")
            buf_ef.seek(0)
            st.pyplot(fig_ef)
            st.download_button("Download Electrofacies Plot (300 DPI)", data=buf_ef, file_name=f"{selected_well}_Electrofacies.png", mime="image/png")

            # Cluster statistics summary
            stat_cols = [c for c in ['GR', 'VSH', 'PHID', 'SW', 'PERM_AI'] if c in single_well_df.columns]
            if stat_cols and single_well_df['ELECTROFACIES'].notna().any():
                grp_ef = single_well_df[single_well_df['ELECTROFACIES'].notna()].groupby('ELECTROFACIES')[stat_cols].mean().round(3)
                st.dataframe(grp_ef, use_container_width=True)
        else:
            st.info("Select logs and cluster count, then click Run Electrofacies Clustering.")

    with tab_props:
        st.subheader("Curve & Layout Properties")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_height = st.slider("Figure Height:", min_value=5, max_value=20, value=9, key="p_h")
            track_width = st.slider("Track Width:", min_value=1.5, max_value=5.0, value=2.5, step=0.1, key="p_w")
        with p_col2:
            title_fontsize = st.slider("Title Font Size:", min_value=8, max_value=16, value=10, key="p_fs")
            show_grid = st.checkbox("Show Grid:", value=True, key="p_gr")

        st.markdown("---")
        if len(all_plot_logs) > 0:
            sel_curve = st.selectbox("Select Curve:", options=all_plot_logs, key="p_curve")
            curr_p = st.session_state.track_props.get(sel_curve, {'color': '#1f77b4', 'lw': 1.3, 'ls': '-'})

            c1, c2, c3 = st.columns(3)
            with c1:
                new_col = st.color_picker("Color:", value=curr_p['color'], key=f"col_{sel_curve}")
            with c2:
                new_lw = st.slider("Line Width:", min_value=0.5, max_value=5.0, value=float(curr_p['lw']), step=0.1, key=f"lw_{sel_curve}")
            with c3:
                ls_map = {'Solid (-)': '-', 'Dashed (--)': '--', 'Dotted (:)': ':', 'Dash-Dot (-.)': '-.'}
                ls_lbl = st.selectbox("Line Style:", options=list(ls_map.keys()), index=list(ls_map.values()).index(curr_p['ls']), key=f"ls_{sel_curve}")

            st.session_state.track_props[sel_curve] = {'color': new_col, 'lw': new_lw, 'ls': ls_map[ls_lbl]}

    with tab_s2:
        st.subheader(f"Core Permeability, Tight Intervals & Reservoir Units — {selected_well} ({formation_name})")
        if st.session_state.perm_summary is not None and len(single_well_df) > 0:
            ps = st.session_state.perm_summary
            st.caption(f"Training Source: {ps['train_source']} | Core Samples: {ps['core_samples']} | Model: {ps['algo']}")
            d_arr = single_well_df['MD'].values
            dz = float(np.median(np.diff(d_arr))) if len(d_arr) > 1 else 0.1524
            gross_thick = float(d_arr.max() - d_arr.min()) if len(d_arr) > 1 else 0.0
            pay_thick = float(single_well_df['PAY_FLAG'].sum() * dz)
            tight_thick = float(single_well_df['TIGHT_FLAG'].sum() * dz)
            ntg_pct = (pay_thick / gross_thick * 100.0) if gross_thick > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("log10(K) R2", ps['r2_logk'])
            m2.metric("Net Pay (m)", f"{pay_thick:.2f}")
            m3.metric("Net-to-Gross (%)", f"{ntg_pct:.1f}%")
            m4.metric("Tight Thickness (m)", f"{tight_thick:.2f}")

            fig_s2, (ax_k, ax_pay) = plt.subplots(1, 2, figsize=(9.5, fig_height), sharey=True)

            if 'PERM_AI' in single_well_df.columns and single_well_df['PERM_AI'].notna().any():
                k_plot = single_well_df['PERM_AI'].clip(lower=1e-3)
                kp = st.session_state.track_props.get('PERM_AI', {'color': '#2ca02c', 'lw': 1.4, 'ls': '-'})
                ax_k.semilogx(k_plot, d_arr, color=kp['color'], lw=kp['lw'], ls=kp['ls'], label="PERM_AI (mD)")
                if 'CORE_K' in single_well_df.columns and single_well_df['CORE_K'].notna().any():
                    ck_sub = single_well_df[single_well_df['CORE_K'] > 0]
                    ax_k.scatter(ck_sub['CORE_K'], ck_sub['MD'], color="#000000", s=14, zorder=5, label="Core K (mD)")
                ax_k.axvline(st.session_state.cutoff_params['tight_k_max'], color="#d62728", ls="--", lw=1.1, label="Tight Cutoff")
                ax_k.axvline(st.session_state.cutoff_params['k_min'], color="#1f77b4", ls=":", lw=1.2, label="Pay Cutoff")
                ax_k.set_title("Permeability (mD)", fontsize=title_fontsize, fontweight="bold")
                ax_k.set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
                ax_k.set_ylim(d_arr.max(), d_arr.min())
                ax_k.legend(loc="best", fontsize=7.5)
                if show_grid:
                    ax_k.grid(True, which="both", linestyle="--", alpha=0.4)

            pay_arr = single_well_df['PAY_FLAG'].fillna(0).values
            tight_arr = single_well_df['TIGHT_FLAG'].fillna(0).values
            ax_pay.fill_betweenx(d_arr, 0, pay_arr, color="#2ca02c", alpha=0.65, step="mid", label="Pay Zone")
            ax_pay.fill_betweenx(d_arr, 0, np.where((pay_arr == 0) & (tight_arr == 1), 0.5, 0), color="#8c564b", alpha=0.55, step="mid", label="Tight / Low-K")
            ax_pay.set_xlim(0, 1.1)
            ax_pay.set_xticks([0, 0.5, 1])
            ax_pay.set_xticklabels(["Non-Pay", "Tight", "Pay"], fontsize=8)
            ax_pay.set_title("Reservoir Pay & Tight Zones", fontsize=title_fontsize, fontweight="bold")
            ax_pay.legend(loc="best", fontsize=7.5)
            if show_grid:
                ax_pay.grid(True, linestyle="--", alpha=0.4)

            plt.tight_layout()
            buf_s2 = io.BytesIO()
            fig_s2.savefig(buf_s2, format="png", dpi=300, bbox_inches="tight")
            buf_s2.seek(0)
            st.pyplot(fig_s2)
            st.download_button("Download Perm & Units Plot (300 DPI)", data=buf_s2, file_name=f"{selected_well}_Perm_Pay_Units.png", mime="image/png")

            if 'RES_UNIT' in single_well_df.columns:
                unit_rows = []
                for u_name, u_grp in single_well_df[single_well_df['RES_UNIT'] != 'Non-Pay'].groupby('RES_UNIT'):
                    u_thick = float(len(u_grp) * dz)
                    avg_phi = float(u_grp['PHID'].mean()) if 'PHID' in u_grp.columns and u_grp['PHID'].notna().any() else np.nan
                    avg_vsh = float(u_grp['VSH'].mean()) if 'VSH' in u_grp.columns and u_grp['VSH'].notna().any() else np.nan
                    avg_sw = float(u_grp['SW'].mean()) if 'SW' in u_grp.columns and u_grp['SW'].notna().any() else np.nan
                    avg_k = float(u_grp['PERM_AI'].mean()) if 'PERM_AI' in u_grp.columns and u_grp['PERM_AI'].notna().any() else np.nan
                    unit_rows.append({
                        "Unit": u_name,
                        "Top (m)": round(float(u_grp['MD'].min()), 2),
                        "Base (m)": round(float(u_grp['MD'].max()), 2),
                        "Thickness (m)": round(u_thick, 2),
                        "Avg PHID": round(avg_phi, 3) if not np.isnan(avg_phi) else None,
                        "Avg VSH": round(avg_vsh, 3) if not np.isnan(avg_vsh) else None,
                        "Avg SW": round(avg_sw, 3) if not np.isnan(avg_sw) else None,
                        "Avg K (mD)": round(avg_k, 2) if not np.isnan(avg_k) else None
                    })
                if unit_rows:
                    st.markdown("**Reservoir Units Summary**")
                    st.dataframe(pd.DataFrame(unit_rows), use_container_width=True)
        else:
            st.info("Configure core training source and cut-offs, then click Run Stage 2.")

    with tab_ai:
        st.subheader(f"Log Reconstruction & Uncertainty — {selected_well} ({formation_name})")
        if st.session_state.ai_summary is not None:
            s = st.session_state.ai_summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Training R2", s['r2_train'])
            m2.metric("Washout Points", s['washout_count'])
            m3.metric("Filled / Corrected", s['filled_count'])
            m4.metric("Mean Uncertainty (±1s)", s['mean_unc'])

            t_name = s['target']
            ai_c = f"{t_name}_AI"
            mg_c = f"{t_name}_MERGED"
            uc_c = f"{t_name}_UNC"

            if ai_c in single_well_df.columns and len(single_well_df) > 0:
                fig_ai, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, fig_height), sharey=True)
                d_vals = single_well_df['MD'].values

                p_orig = st.session_state.track_props.get(t_name, {'color': '#1f77b4', 'lw': 1.2, 'ls': '-'})
                ax1.plot(single_well_df[t_name], d_vals, label=f"{t_name} (Original)", color=p_orig['color'], lw=p_orig['lw'], ls=p_orig['ls'])
                if mg_c in single_well_df.columns:
                    ax1.plot(single_well_df[mg_c], d_vals, label=f"{mg_c} (AI Corrected)", color="#d62728", lw=1.2, ls="--")

                if uc_c in single_well_df.columns and mg_c in single_well_df.columns:
                    y_m = single_well_df[mg_c].values
                    u_m = single_well_df[uc_c].values
                    ax1.fill_betweenx(d_vals, y_m - u_m, y_m + u_m, color="#d62728", alpha=0.18, label="Uncertainty (±1s)")

                ax1.set_title(f"{t_name} Reconstruction", fontsize=title_fontsize, fontweight="bold")
                ax1.set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
                ax1.set_ylim(d_vals.max(), d_vals.min())
                ax1.legend(loc="best", fontsize=8)
                if show_grid:
                    ax1.grid(True, linestyle="--", alpha=0.5)

                if uc_c in single_well_df.columns:
                    ax2.plot(single_well_df[uc_c], d_vals, color="#ff7f0e", lw=1.3)
                    ax2.fill_betweenx(d_vals, 0, single_well_df[uc_c].values, color="#ff7f0e", alpha=0.25)
                ax2.set_title("Uncertainty (±1s)", fontsize=title_fontsize, fontweight="bold")
                if show_grid:
                    ax2.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                buf_ai = io.BytesIO()
                fig_ai.savefig(buf_ai, format="png", dpi=300, bbox_inches="tight")
                buf_ai.seek(0)
                st.pyplot(fig_ai)
                st.download_button("Download Prediction Plot (300 DPI)", data=buf_ai, file_name=f"{selected_well}_{t_name}_AI_Uncertainty.png", mime="image/png")
        else:
            st.info("Configure target and predictor logs, then click Run Stage 1.")

    with tab_tracks:
        st.subheader(f"Well Log Tracks — {selected_well}")
        default_tr = [t for t in ['GR', 'VSH', 'PHID', 'SW', 'PERM_AI', 'ELECTROFACIES'] if t in all_plot_logs or t == 'ELECTROFACIES']
        chosen_tracks = st.multiselect("Select Tracks:", options=all_plot_logs + (['ELECTROFACIES'] if 'ELECTROFACIES' in master_df.columns else []), default=default_tr)

        if len(chosen_tracks) > 0 and len(single_well_df) > 0:
            fig_t, axes = plt.subplots(1, len(chosen_tracks), figsize=(track_width * len(chosen_tracks), fig_height), sharey=True)
            if len(chosen_tracks) == 1:
                axes = [axes]
            d_t = single_well_df['MD']
            for i, tr in enumerate(chosen_tracks):
                if tr == 'ELECTROFACIES':
                    ef_v = single_well_df['ELECTROFACIES'].fillna(0).values
                    cmap_t = plt.get_cmap('tab10', max(int(ef_v.max()) + 1, 2))
                    for k_id in sorted(np.unique(ef_v)):
                        if k_id > 0:
                            axes[i].fill_betweenx(d_t, 0, 1, where=(ef_v == k_id), color=cmap_t(int(k_id) - 1), alpha=0.85, step="mid")
                    axes[i].set_xlim(0, 1)
                    axes[i].set_xticks([])
                else:
                    tp = st.session_state.track_props.get(tr, {'color': '#1f77b4', 'lw': 1.3, 'ls': '-'})
                    axes[i].plot(single_well_df[tr], d_t, color=tp['color'], lw=tp['lw'], ls=tp['ls'])
                axes[i].set_title(tr, fontsize=title_fontsize, fontweight="bold")
                axes[i].set_ylim(d_t.max(), d_t.min())
                if show_grid:
                    axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[0].set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
            plt.tight_layout()

            buf_t = io.BytesIO()
            fig_t.savefig(buf_t, format="png", dpi=300, bbox_inches="tight")
            buf_t.seek(0)
            st.pyplot(fig_t)
            st.download_button("Download Tracks Plot (300 DPI)", data=buf_t, file_name=f"{selected_well}_Tracks.png", mime="image/png")

    with tab_export:
        st.subheader("Export Results")
        csv_bytes = master_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Master Table (CSV)", data=csv_bytes, file_name=f"{formation_name}_AI_Suite_Results.csv", mime="text/csv")
        st.dataframe(master_df.head(100), use_container_width=True)
