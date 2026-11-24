import pickle
import tempfile
import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Electrofacies & Petrophysical Suite", layout="wide")
st.title("Electrofacies & Petrophysical Suite - test")

DEFAULT_FACIES_COLORS = [
    "#E63946",
    "#2A9D8F",
    "#E9C46A",
    "#457B9D",
    "#8D5B4C",
    "#9B5DE5",
    "#F77F00",
    "#3A86FF",
    "#06D6A0",
    "#FF006E",
    "#6C757D",
    "#AACC00",
]

CLUSTER_ALGORITHMS = [
    "K-Means",
    "Agglomerative (Hierarchical)",
    "Gaussian Mixture (GMM)",
    "MiniBatch K-Means",
]

DEFAULTS = {
    "master_df": None,
    "core_df": None,
    "facies_colors": {},
    "ai_summary": None,
    "perm_summary": None,
    "ef_summary": None,
    "cutoff_params": {
        "vsh_max": 0.40,
        "phi_min": 0.10,
        "sw_max": 0.60,
        "k_min": 1.0,
        "tight_k_max": 1.0,
        "min_unit_thickness": 2.0,
    },
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def get_facies_color(facies_id: int) -> str:
    fid = int(facies_id)
    idx = max(0, fid - 1)
    fallback = DEFAULT_FACIES_COLORS[idx % len(DEFAULT_FACIES_COLORS)]
    if not isinstance(st.session_state.facies_colors, dict):
        st.session_state.facies_colors = {}
    return st.session_state.facies_colors.get(fid, fallback)


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


def fit_predict_facies(algo_name: str, X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    k = int(n_clusters)
    if algo_name == "Agglomerative (Hierarchical)":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(X_scaled) + 1
    if algo_name == "Gaussian Mixture (GMM)":
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        model.fit(X_scaled)
        return model.predict(X_scaled) + 1
    if algo_name == "MiniBatch K-Means":
        model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        return model.fit_predict(X_scaled) + 1
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(X_scaled) + 1


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


def plot_facies_track(ax, df_well: pd.DataFrame):
    ax.set_title("ELECTROFACIES", fontsize=10, fontweight="bold")
    sub = df_well[["MD", "ELECTROFACIES"]].dropna().sort_values("MD")
    if sub.empty:
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.text(0.5, 0.5, "No Facies", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return

    md = sub["MD"].values
    ef = sub["ELECTROFACIES"].astype(int).values
    if len(md) > 1:
        dz = np.abs(np.diff(md))
        dz = np.append(dz, dz[-1])
        dz = np.where((dz <= 0) | np.isnan(dz), 0.1524, dz)
    else:
        dz = np.array([0.1524])

    for i in range(len(md)):
        fid = int(ef[i])
        ax.barh(
            y=md[i],
            width=1.0,
            height=dz[i],
            left=0.0,
            color=get_facies_color(fid),
            edgecolor="none",
            align="center",
        )

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    unique_f = sorted(np.unique(ef).tolist())
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=get_facies_color(int(u)), edgecolor="black", linewidth=0.4)
        for u in unique_f
    ]
    labels = [f"EF-{int(u)}" for u in unique_f]
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)


# ==========================================
# Sidebar: Project, Data Input & Well Selection
# ==========================================
st.sidebar.header("Project & Data")
saved_proj = st.sidebar.file_uploader("Resume Project (.zubair):", type=["zubair"])
if saved_proj is not None and st.sidebar.button("Load Project"):
    proj = pickle.load(saved_proj)
    for key in DEFAULTS:
        if key in proj and proj[key] is not None:
            st.session_state[key] = proj[key]
    st.sidebar.success("Project restored.")
    st.rerun()

if st.session_state.master_df is not None:
    snap = {k: st.session_state[k] for k in DEFAULTS}
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
        st.rerun()

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
w_df_view = mdf[(mdf["WELL_NAME"] == selected_well) & (mdf["MD"] >= min_depth) & (mdf["MD"] <= max_depth)].sort_values("MD")

num_all = [c for c in mdf.columns if pd.api.types.is_numeric_dtype(mdf[c]) and c != "MD"]

# ==========================================
# Top Summary Strip
# ==========================================
s1, s2, s3, s4 = st.columns(4)
s1.metric("Well", str(selected_well))
s2.metric("Formation", formation_name)
s3.metric("Electrofacies Count", int(w_df_view["ELECTROFACIES"].nunique(dropna=True)) if "ELECTROFACIES" in w_df_view else 0)
s4.metric("Core Samples", int(len(cdf)) if cdf is not None and len(cdf) > 0 else int(pd.to_numeric(w_sub_all["CORE_K"], errors="coerce").gt(0).sum()))

# ==========================================
# Dedicated Tabs
# ==========================================
tab_logs, tab_crossplot, tab_stats, tab_petro, tab_export = st.tabs([
    "1. Logs & Electrofacies Track",
    "2. Crossplot",
    "3. Statistical Plots",
    "4. Petrophysics & Permeability",
    "5. Data & Export",
])

# ------------------------------------------
# Tab 1: Electrofacies Classification & Log Track
# ------------------------------------------
with tab_logs:
    st.subheader("Electrofacies Classification")
    ef_def = [c for c in ["GR", "VSH", "PHID", "NPHI", "RHOB"] if c in num_all]

    ef_c1, ef_c2, ef_c3, ef_c4 = st.columns([2.2, 2.0, 1.0, 1.1])
    ef_feats = ef_c1.multiselect("Electrofacies Input Curves:", options=num_all, default=ef_def)
    ef_algo = ef_c2.selectbox("Clustering Algorithm:", options=CLUSTER_ALGORITHMS)
    n_clusters = ef_c3.number_input("Facies Count:", min_value=2, max_value=12, value=5, step=1)
    with ef_c4:
        st.write("")
        st.write("")
        run_ef_btn = st.button("Classify Electrofacies", type="primary")

    if run_ef_btn:
        if len(ef_feats) < 1:
            st.error("Select at least one numeric curve.")
        else:
            valid_ef = mdf[ef_feats].notna().all(axis=1)
            if int(valid_ef.sum()) < int(n_clusters):
                st.error("Not enough complete rows for the selected facies count. Check missing curves or compute VSH/PHID first.")
            else:
                sc_ef = StandardScaler()
                X_ef = sc_ef.fit_transform(mdf.loc[valid_ef, ef_feats])
                labels_ef = fit_predict_facies(ef_algo, X_ef, int(n_clusters))
                mdf["ELECTROFACIES"] = np.nan
                mdf.loc[valid_ef, "ELECTROFACIES"] = labels_ef
                for idx_k in range(1, int(n_clusters) + 1):
                    if idx_k not in st.session_state.facies_colors:
                        st.session_state.facies_colors[idx_k] = DEFAULT_FACIES_COLORS[(idx_k - 1) % len(DEFAULT_FACIES_COLORS)]
                st.session_state.master_df = mdf
                st.session_state.ef_summary = mdf.groupby("ELECTROFACIES", dropna=True)[ef_feats].mean().round(4).to_dict(orient="index")
                st.success(f"Electrofacies classified using {ef_algo}.")
                st.rerun()

    if mdf["ELECTROFACIES"].notna().any():
        active_f_ids = sorted(mdf["ELECTROFACIES"].dropna().astype(int).unique().tolist())
        with st.expander("Facies Colors Customization", expanded=True):
            color_cols = st.columns(min(6, max(1, len(active_f_ids))))
            for i, f_id in enumerate(active_f_ids):
                def_c = get_facies_
