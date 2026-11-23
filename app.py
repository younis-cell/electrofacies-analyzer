import pickle
import tempfile
import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Petrophysical Suite", layout="wide")
st.title("AI Petrophysical & Electrofacies Suite")

DEFAULTS = {
    "master_df": None,
    "core_df": None,
    "track_props": {},
    "ai_summary": None,
    "perm_summary": None,
    "ef_summary": None,
    "layout_props": {
        "fig_height": 9,
        "track_width": 2.5,
        "title_fontsize": 10,
        "show_grid": True,
    },
    "cutoff_params": {
        "vsh_max": 0.40,
        "phi_min": 0.10,
        "sw_max": 0.60,
        "k_min": 1.0,
        "tight_k_max": 1.0,
        "min_unit_thickness": 2.0,
    },
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper().strip() for c in out.columns]
    for cand in ["MD", "DEPT", "DEPTH", "TVD"]:
        if cand in out.columns:
            out = out.rename(columns={cand: "MD"})
            break
    if "MD" in out.columns:
        out["MD"] = pd.to_numeric(out["MD"], errors="coerce")
        out = out.dropna(subset=["MD"])
    return out


def get_regressor(algo_name: str):
    if algo_name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(n_estimators=150, random_state=42)
    return RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)


def predict_with_uncertainty(model, X_train_sc, y_train, X_pred_sc):
    model.fit(X_train_sc, y_train)
    y_hat = model.predict(X_pred_sc)
    if hasattr(model, "estimators_") and isinstance(model.estimators_, (list, np.ndarray)):
        tree_preds = np.array([est.predict(X_pred_sc) for est in np.ravel(model.estimators_)])
        unc = np.std(tree_preds, axis=0)
    else:
        res_std = float(np.std(y_train - model.predict(X_train_sc)))
        unc = np.full(len(y_hat), res_std)
    return y_hat, unc


def assign_units(df_well: pd.DataFrame, min_thick: float) -> pd.Series:
    labels = ["Non-Pay"] * len(df_well)
    if len(df_well) < 2:
        return pd.Series(labels, index=df_well.index)
    d_sorted = np.sort(df_well["MD"].values)
    dz = float(np.median(np.diff(d_sorted)))
    if dz <= 0 or np.isnan(dz):
        dz = 0.1524
    pay = df_well["PAY_FLAG"].fillna(0).astype(int).values
    unit_id = 0
    start = None
    for i, flag in enumerate(pay):
        if flag == 1 and start is None:
            start = i
        elif flag == 0 and start is not None:
            if (i - start) * dz >= min_thick:
                unit_id += 1
                for j in range(start, i):
                    labels[j] = f"Unit-{unit_id}"
            start = None
    if start is not None and (len(pay) - start) * dz >= min_thick:
        unit_id += 1
        for j in range(start, len(pay)):
            labels[j] = f"Unit-{unit_id}"
    return pd.Series(labels, index=df_well.index)


def add_rolling_features(df: pd.DataFrame, cols: list, win: int, group_by_well: bool):
    out = df.copy()
    feat_cols = []
    for col in cols:
        if col not in out.columns:
            continue
        feat_cols.append(col)
        rm_col = f"{col}_RM{win}"
        rs_col = f"{col}_RS{win}"
        if group_by_well and "WELL_NAME" in out.columns:
            out[rm_col] = out.groupby("WELL_NAME")[col].transform(
                lambda s: s.rolling(win, center=True, min_periods=1).mean()
            )
            out[rs_col] = out.groupby("WELL_NAME")[col].transform(
                lambda s: s.rolling(win, center=True, min_periods=1).std().fillna(0)
            )
        else:
            out[rm_col] = out[col].rolling(win, center=True, min_periods=1).mean()
            out[rs_col] = out[col].rolling(win, center=True, min_periods=1).std().fillna(0)
        feat_cols.extend([rm_col, rs_col])
    return out, feat_cols


# ==========================================
# Sidebar: Project, Data Input & Well Selection
# ==========================================
st.sidebar.header("Project & Data")
saved_proj = st.sidebar.file_uploader("Resume Project (.zubair):", type=["zubair"])
if saved_proj is not None and st.sidebar.button("Load Project"):
    proj = pickle.load(saved_proj)
    for key in ["master_df", "core_df", "track_props", "ai_summary", "perm_summary", "ef_summary", "layout_props", "cutoff_params"]:
        if key in proj and proj[key] is not None:
            st.session_state[key] = proj[key]
    st.sidebar.success("Project restored.")

if st.session_state.master_df is not None:
    snap = {k: st.session_state[k] for k in ["master_df", "core_df", "track_props", "ai_summary", "perm_summary", "ef_summary", "layout_props", "cutoff_params"]}
    st.sidebar.download_button(
        label="Save Project (.zubair)",
        data=pickle.dumps(snap),
        file_name="Petrophysics_AI_Suite.zubair",
        mime="application/octet-stream",
    )

las_files = st.sidebar.file_uploader("Upload LAS Files:", type=["las"], accept_multiple_files=True)
core_upload = st.sidebar.file_uploader("Core File (CSV / XLSX):", type=["csv", "xlsx"])
formation_name = st.sidebar.text_input("Formation Name:", value="Zubair Formation")

if las_files and st.sidebar.button("Load Wells & Core"):
    wells_frames = []
    for f in las_files:
        w_name = f.name.rsplit(".", 1)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            las_obj = lasio.read(tmp_path)
            w_df = normalize_df(las_obj.df().reset_index())
            w_df["WELL_NAME"] = w_name
            wells_frames.append(w_df)
        except Exception as err:
            st.error(f"Error reading {w_name}: {err}")

    if wells_frames:
        mdf = pd.concat(wells_frames, ignore_index=True)
        for col in ["VSH", "PHID", "SW", "WASHOUT_FLAG", "CORE_K", "PERM_AI", "PERM_UNC", "TIGHT_FLAG", "PAY_FLAG", "ELECTROFACIES"]:
            if col not in mdf.columns:
                mdf[col] = np.nan
        if "RES_UNIT" not in mdf.columns:
            mdf["RES_UNIT"] = "Non-Pay"

        parsed_core = None
        if core_upload is not None:
            try:
                cdf = pd.read_csv(core_upload) if core_upload.name.lower().endswith(".csv") else pd.read_excel(core_upload)
                cdf = normalize_df(cdf)
                k_col = next((c for c in ["CORE_K", "PERM", "K", "PERMEABILITY", "KH", "KL"] if c in cdf.columns), None)
                if k_col:
                    cdf = cdf.rename(columns={k_col: "CORE_K"})
                    cdf["CORE_K"] = pd.to_numeric(cdf["CORE_K"], errors="coerce")
                    cdf = cdf[cdf["CORE_K"] > 0].copy()
                    parsed_core = cdf
                    if "WELL_NAME" in cdf.columns and "MD" in cdf.columns:
                        cdf["WELL_NAME"] = cdf["WELL_NAME"].astype(str).str.strip()
                        for w_n, grp in cdf.groupby("WELL_NAME"):
                            mask_w = mdf["WELL_NAME"] == w_n
                            if mask_w.any() and len(grp) > 0:
                                s_grp = grp.sort_values("MD")
                                mdf.loc[mask_w, "CORE_K"] = np.interp(
                                    mdf.loc[mask_w, "MD"].values,
                                    s_grp["MD"].values,
                                    s_grp["CORE_K"].values,
                                    left=np.nan,
                                    right=np.nan,
                                )
            except Exception as err:
                st.warning(f"Core file notice: {err}")

        st.session_state.master_df = mdf
        st.session_state.core_df = parsed_core
        st.sidebar.success("Data loaded.")

if st.session_state.master_df is None:
    st.info("Upload one or more LAS files in the sidebar and click **Load Wells & Core** to begin.")
    st.stop()

mdf = st.session_state.master_df
cdf = st.session_state.core_df

st.sidebar.markdown("---")
well_names = mdf["WELL_NAME"].unique().tolist()
selected_well = st.sidebar.selectbox("Active Well:", options=well_names)
w_sub_all = mdf[mdf["WELL_NAME"] == selected_well]
d_min_def = float(w_sub_all["MD"].min()) if not w_sub_all["MD"].isna().all() else 0.0
d_max_def = float(w_sub_all["MD"].max()) if not w_sub_all["MD"].isna().all() else 1000.0

c_d1, c_d2 = st.sidebar.columns(2)
min_depth = c_d1.number_input("Top (m):", value=d_min_def)
max_depth = c_d2.number_input("Base (m):", value=d_max_def)

# ==========================================
# Main Workspace: Dedicated Tabs
# ==========================================
tab_petro, tab_facies, tab_export = st.tabs([
    "1. Petrophysics & Stage 1 AI",
    "2. Electrofacies & Stage 2 Perm/Pay",
    "3. Logs, Crossplots & Export",
])

# ------------------------------------------
# Tab 1: Direct Petrophysics & Stage 1
# ------------------------------------------
with tab_petro:
    st.subheader("Direct Petrophysical Evaluation (VSH, PHID, SW)")
    c1, c2, c3, c4 = st.columns(4)
    rho_ma = c1.number_input("Matrix Density (g/cc):", value=2.65, step=0.01)
    rho_fl = c2.number_input("Fluid Density (g/cc):", value=1.00, step=0.05)
    rw_in = c3.number_input("Water Resistivity (Rw):", value=0.03, step=0.01)
    with c4:
        st.write("")
        st.write("")
        run_base_btn = st.button("Compute VSH, PHID, SW", type="primary")

    if run_base_btn:
        if "GR" in mdf.columns:
            gr_min, gr_max = mdf["GR"].quantile(0.05), mdf["GR"].quantile(0.95)
            if gr_max != gr_min:
                igr = (mdf["GR"] - gr_min) / (gr_max - gr_min)
                mdf["VSH"] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
        if "RHOB" in mdf.columns and (rho_ma - rho_fl) != 0:
            mdf["PHID"] = ((rho_ma - mdf["RHOB"]) / (rho_ma - rho_fl)).clip(0, 0.45)
        res_c = next((c for c in ["ILD", "LLD", "RT", "RILD", "RESD"] if c in mdf.columns), None)
        if res_c and "PHID" in mdf.columns:
            phi_s = mdf["PHID"].replace(0, 0.001)
            rt_s = mdf[res_c].replace(0, 1)
            mdf["SW"] = np.power(rw_in / (np.power(phi_s, 2.0) * rt_s), 0.5).clip(0, 1)
        st.session_state.master_df = mdf
        st.success("Base petrophysical properties updated.")

    st.divider()
    st.subheader("Stage 1: Missing Log & Washout Correction AI")
    skip_s1 = {"MD", "WELL_NAME", "WASHOUT_FLAG", "TIGHT_FLAG", "PAY_FLAG", "RES_UNIT", "ELECTROFACIES"}
    avail_s1 = [c for c in mdf.columns if c not in skip_s1]

    s1_c1, s1_c2, s1_c3 = st.columns(3)
    target_log = s1_c1.selectbox("Target Log:", options=avail_s1)
    pred_opts = [c for c in avail_s1 if c != target_log]
    pred_def = [c for c in ["GR", "NPHI", "DT", "RHOB", "VSH"] if c in pred_opts]
    predictor_logs = s1_c2.multiselect("Predictor Logs:", options=pred_opts, default=pred_def)
    algo_s1 = s1_c3.selectbox("Model (Stage 1):", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="s1_m")

    with st.expander("Washout & Merge Settings"):
        w1, w2, w3, w4 = st.columns(4)
        cali_sel = w1.selectbox("Caliper Log:", options=["None"] + avail_s1)
        bit_sz = w2.number_input("Bit Size:", value=8.5, step=0.5)
        dc_thr = w3.number_input("Washout Threshold (dCali):", value=1.5, step=0.25)
        merge_mode = w4.selectbox("Merge Mode:", options=["Smart Fill (Missing & Washout)", "Full Synthetic"])

    if st.button("Run Stage 1"):
        if not predictor_logs:
            st.error("Select at least one predictor log.")
        else:
            wo_mask = ((mdf[cali_sel] - bit_sz) > dc_thr) if cali_sel != "None" else pd.Series(False, index=mdf.index)
            mdf["WASHOUT_FLAG"] = wo_mask.astype(int)
            tr_m = (~wo_mask) & mdf[target_log].notna() & mdf[predictor_logs].notna().all(axis=1)
            pr_m = mdf[predictor_logs].notna().all(axis=1)
            if tr_m.sum() < 20 or pr_m.sum() == 0:
                st.error("Insufficient training or input samples.")
            else:
                sc1 = StandardScaler()
                X_tr1 = sc1.fit_transform(mdf.loc[tr_m, predictor_logs])
                X_pr1 = sc1.transform(mdf.loc[pr_m, predictor_logs])
                y_hat1, unc1 = predict_with_uncertainty(get_regressor(algo_s1), X_tr1, mdf.loc[tr_m, target_log], X_pr1)

                mdf[f"{target_log}_AI"] = np.nan
                mdf[f"{target_log}_UNC"] = np.nan
                mdf.loc[pr_m, f"{target_log}_AI"] = y_hat1
                mdf.loc[pr_m, f"{target_log}_UNC"] = unc1
                if merge_mode == "Full Synthetic":
                    mdf[f"{target_log}_MERGED"] = mdf[f"{target_log}_AI"]
                else:
                    fix_m = mdf[target_log].isna() | wo_mask
                    mdf[f"{target_log}_MERGED"] = np.where(fix_m & mdf[f"{target_log}_AI"].notna(), mdf[f"{target_log}_AI"], mdf[target_log])

                st.session_state.master_df = mdf
                st.session_state.ai_summary = {
                    "formation": formation_name,
                    "target": target_log,
                    "r2_train": round(float(get_regressor(algo_s1).fit(X_tr1, mdf.loc[tr_m, target_log]).score(X_tr1, mdf.loc[tr_m, target_log])), 4),
                    "washout_count": int(wo_mask.sum()),
                    "filled_count": int((mdf[target_log].isna() | wo_mask).sum()),
                    "mean_unc": round(float(np.nanmean(unc1)), 4),
                }
                st.success("Stage 1 completed.")

    if st.session_state.ai_summary is not None:
        st.json(st.session_state.ai_summary)

# ------------------------------------------
# Tab 2: Electrofacies & Stage 2 Perm/Pay
# ------------------------------------------
with tab_facies:
    st.subheader("Electrofacies Clustering (K-Means)")
    num_all = [c for c in mdf.columns if pd.api.types.is_numeric_dtype(mdf[c]) and c != "MD"]
    ef_def = [c for c in ["GR", "VSH", "PHID", "NPHI", "RHOB"] if c in num_all]
    ef_c1, ef_c2, ef_c3 = st.columns([2, 1, 1])
    ef_feats = ef_c1.multiselect("Electrofacies Input Curves:", options=num_all, default=ef_def)
    n_clusters = ef_c2.number_input("Number of Electrofacies:", min_value=2, max_value=12, value=5, step=1)
    with ef_c3:
        st.write("")
        st.write("")
        run_ef_btn = st.button("Classify Electrofacies")

    if run_ef_btn:
        if len(ef_feats) < 1:
            st.error("Select at least one numeric curve for electrofacies clustering.")
        else:
            valid_ef = mdf[ef_feats].notna().all(axis=1)
            if valid_ef.sum() < int(n_clusters):
                st.error("Not enough complete rows for the selected cluster count.")
            else:
                sc_ef = StandardScaler()
                X_ef = sc_ef.fit_transform(mdf.loc[valid_ef, ef_feats])
                km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
                mdf["ELECTROFACIES"] = np.nan
                mdf.loc[valid_ef, "ELECTROFACIES"] = km.fit_predict(X_ef) + 1
                st.session_state.master_df = mdf
                st.session_state.ef_summary = mdf.groupby("ELECTROFACIES", dropna=True)[ef_feats].mean().round(4).to_dict(orient="index")
                st.success("Electrofacies classification completed.")

    if st.session_state.ef_summary is not None:
        st.dataframe(pd.DataFrame.from_dict(st.session_state.ef_summary, orient="index"))

    st.divider()
    st.subheader("Stage 2: Core Permeability Prediction, Cutoffs & Reservoir Units")
    skip_s2 = {"MD", "WELL_NAME", "WASHOUT_FLAG", "TIGHT_FLAG", "PAY_FLAG", "RES_UNIT", "ELECTROFACIES", "PERM_AI", "PERM_UNC"}
    las_num_cols = [c for c in mdf.columns if c not in skip_s2 and pd.api.types.is_numeric_dtype(mdf[c])]

    s2_c1, s2_c2, s2_c3 = st.columns(3)
    core_mode = s2_c1.selectbox("Core Training Source:", options=["Core File (Reference Core Wells)", "Embedded Core Column"])
    if core_mode == "Core File (Reference Core Wells)" and cdf is not None:
        c_cols = [c for c in cdf.columns if c not in {"MD", "CORE_K"} and pd.api.types.is_numeric_dtype(cdf[c])]
        shared = [c for c in c_cols if c in las_num_cols]
        p_opts2 = shared if shared else las_num_cols
    else:
        p_opts2 = [c for c in las_num_cols if c != "CORE_K"]

    p_def2 = [c for c in ["PHID", "VSH", "GR", "NPHI", "RHOB"] if c in p_opts2]
    perm_feats = s2_c2.multiselect("Predictor Features:", options=p_opts2, default=p_def2)
    algo_s2 = s2_c3.selectbox("Model (Stage 2):", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="s2_m")

    cp = st.session_state.cutoff_params
    cut_c1, cut_c2, cut_c3, cut_c4, cut_c5, cut_c6 = st.columns(6)
    vsh_max = cut_c1.number_input("Max VSH:", value=float(cp["vsh_max"]), min_value=0.0, max_value=1.0, step=0.02)
    phi_min = cut_c2.number_input("Min PHID:", value=float(cp["phi_min"]), min_value=0.0, max_value=0.5, step=0.01)
    sw_max = cut_c3.number_input("Max SW:", value=float(cp["sw_max"]), min_value=0.0, max_value=1.0, step=0.02)
    k_min = cut_c4.number_input("Min Perm (mD):", value=float(cp["k_min"]), min_value=0.001, step=0.5)
    tight_k_max = cut_c5.number_input("Tight K Max (mD):", value=float(cp["tight_k_max"]), min_value=0.001, step=0.2)
    min_unit_th = cut_c6.number_input("Min Thick (m):", value=float(cp["min_unit_thickness"]), min_value=0.5, step=0.5)
    st.session_state.cutoff_params = {
        "vsh_max": vsh_max,
        "phi_min": phi_min,
        "sw_max": sw_max,
        "k_min": k_min,
        "tight_k_max": tight_k_max,
        "min_unit_thickness": min_unit_th,
    }

    with st.expander("Rolling Window Options"):
        use_roll = st.checkbox("Rolling Depth Context", value=True)
        roll_win = st.slider("Rolling Window (pts):", min_value=3, max_value=21, value=7, step=2)

    if st.button("Run Stage 2"):
        if not perm_feats:
            st.error("Select at least one predictor feature.")
        else:
            w_df, las_fcols = add_rolling_features(mdf, perm_feats, roll_win, True) if use_roll else (mdf.copy(), list(perm_feats))
            if core_mode == "Core File (Reference Core Wells)":
                if cdf is None or "CORE_K" not in cdf.columns:
                    st.error("Upload a core file with permeability first.")
                    st.stop()
                c_df, tr_fcols = add_rolling_features(cdf, perm_feats, roll_win, True) if use_roll else (cdf.copy(), [c for c in perm_feats if c in cdf.columns])
                if len(tr_fcols) != len(las_fcols):
                    st.error("Features must match across core and LAS files.")
                    st.stop()
                ck_s = pd.to_numeric(c_df["CORE_K"], errors="coerce")
                m_tr2 = (ck_s > 0) & c_df[tr_fcols].notna().all(axis=1)
                X_tr2 = c_df.loc[m_tr2, tr_fcols]
                y_tr2 = np.log10(ck_s.loc[m_tr2].clip(lower=1e-4))
            else:
                k_s = pd.to_numeric(w_df["CORE_K"], errors="coerce")
                m_tr2 = (k_s > 0) & w_df[las_fcols].notna().all(axis=1)
                X_tr2 = w_df.loc[m_tr2, las_fcols]
                y_tr2 = np.log10(k_s.loc[m_tr2].clip(lower=1e-4))

            pr_m2 = w_df[las_fcols].notna().all(axis=1)
            if len(X_tr2) < 10 or pr_m2.sum() == 0:
                st.error("Insufficient core training samples or predictor rows.")
            else:
                sc2 = StandardScaler()
                X_tr2_sc = sc2.fit_transform(X_tr2)
                X_pr2_sc = sc2.transform(w_df.loc[pr_m2, las_fcols])
                y_hat_log, unc_log = predict_with_uncertainty(get_regressor(algo_s2), X_tr2_sc, y_tr2, X_pr2_sc)

                perm_pred = np.power(10.0, y_hat_log)
                mdf["PERM_AI"] = np.nan
                mdf["PERM_UNC"] = np.nan
                mdf.loc[pr_m2, "PERM_AI"] = perm_pred
                mdf.loc[pr_m2, "PERM_UNC"] = unc_log

                eff_k = mdf["PERM_AI"].fillna(pd.to_numeric(mdf["CORE_K"], errors="coerce"))
                mdf["TIGHT_FLAG"] = ((eff_k > 0) & (eff_k <= tight_k_max)).astype(int)

                cond_pay = pd.Series(True, index=mdf.index)
                if "VSH" in mdf.columns and mdf["VSH"].notna().any():
                    cond_pay &= mdf["VSH"].fillna(1.0) <= vsh_max
                if "PHID" in mdf.columns and mdf["PHID"].notna().any():
                    cond_pay &= mdf["PHID"].fillna(0.0) >= phi_min
                if "SW" in mdf.columns and mdf["SW"].notna().any():
                    cond_pay &= mdf["SW"].fillna(1.0) <= sw_max
                cond_pay &= eff_k.fillna(0.0) >= k_min
                mdf["PAY_FLAG"] = cond_pay.astype(int)

                mdf["RES_UNIT"] = "Non-Pay"
                for w_n, idx in mdf.groupby("WELL_NAME").groups.items():
                    mdf.loc[idx, "RES_UNIT"] = assign_units(mdf.loc[idx], min_unit_th)

                st.session_state.master_df = mdf
                st.session_state.perm_summary = {
                    "training_samples": int(len(X_tr2)),
                    "pay_samples": int(mdf["PAY_FLAG"].sum()),
                    "tight_samples": int(mdf["TIGHT_FLAG"].sum()),
                    "reservoir_units": int(mdf[mdf["RES_UNIT"] != "Non-Pay"]["RES_UNIT"].nunique()),
                }
                st.success("Stage 2 completed.")

    if st.session_state.perm_summary is not None:
        st.json(st.session_state.perm_summary)

# ------------------------------------------
# Tab 3: Log Plot, Crossplot & Export
# ------------------------------------------
with tab_export:
    w_df_view = mdf[(mdf["WELL_NAME"] == selected_well) & (mdf["MD"] >= min_depth) & (mdf["MD"] <= max_depth)].sort_values("MD")
    st.subheader(f"Well: {selected_well} | Formation: {formation_name}")

    plot_cols = [c for c in ["GR", "VSH", "PHID", "SW", "PERM_AI", "ELECTROFACIES", "PAY_FLAG"] if c in w_df_view.columns and w_df_view[c].notna().any()]
    if len(w_df_view) > 0 and plot_cols:
        n_tr = len(plot_cols)
        fig, axes = plt.subplots(1, n_tr, sharey=True, figsize=(max(6, n_tr * 2.6), 9))
        if n_tr == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_cols):
            ax.plot(w_df_view[col], w_df_view["MD"], linewidth=1.1)
            ax.set_title(col, fontsize=10)
            ax.grid(True, alpha=0.35)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("MD (m)")
        st.pyplot(fig)
        plt.close(fig)

    xp_c1, xp_c2 = st.columns(2)
    num_view = [c for c in w_df_view.columns if pd.api.types.is_numeric_dtype(w_df_view[c]) and c != "MD"]
    if len(num_view) >= 2:
        x_ax = xp_c1.selectbox("X-Axis:", options=num_view, index=0)
        y_ax = xp_c2.selectbox("Y-Axis:", options=num_view, index=min(1, len(num_view) - 1))
        fig_xp, ax_xp = plt.subplots(figsize=(6, 4))
        c_vals = w_df_view["ELECTROFACIES"] if "ELECTROFACIES" in w_df_view.columns and w_df_view["ELECTROFACIES"].notna().any() else None
        sc = ax_xp.scatter(w_df_view[x_ax], w_df_view[y_ax], c=c_vals, cmap="tab10", s=14, alpha=0.8)
        ax_xp.set_xlabel(x_ax)
        ax_xp.set_ylabel(y_ax)
        ax_xp.grid(True, alpha=0.3)
        if c_vals is not None:
            plt.colorbar(sc, ax=ax_xp, label="ELECTROFACIES")
        st.pyplot(fig_xp)
        plt.close(fig_xp)

    st.dataframe(w_df_view.head(200))
    st.download_button(
        label="Download Interpreted Well CSV",
        data=w_df_view.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_well}_interpreted.csv",
        mime="text/csv",
    )
