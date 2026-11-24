import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def render_section_header(title: str, subtitle: str):
    st.markdown(f"### {title}")
    st.caption(subtitle)


def render_data_overview(mdf: pd.DataFrame, cdf: pd.DataFrame | None, selected_well: str):
    w_df = mdf[mdf["WELL_NAME"] == selected_well]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wells Loaded", int(mdf["WELL_NAME"].nunique()))
    c2.metric("Active Well Rows", int(len(w_df)))
    c3.metric("Core Samples", int(len(cdf)) if cdf is not None else int(w_df["CORE_K"].notna().sum()) if "CORE_K" in w_df else 0)
    c4.metric("Facies Classified", int(w_df["ELECTROFACIES"].notna().sum()) if "ELECTROFACIES" in w_df else 0)


def render_quick_log_viewer(w_df_view: pd.DataFrame, curves: list, facies_colors: dict | None = None, default_palette: list | None = None):
    if w_df_view.empty or not curves:
        st.info("No numeric curves available in the selected depth interval.")
        return

    n_cols = len(curves)
    fig, axes = plt.subplots(1, n_cols, sharey=True, figsize=(max(6, n_cols * 2.5), 8.5))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, curves):
        if col == "ELECTROFACIES":
            sub = w_df_view[["MD", "ELECTROFACIES"]].dropna().sort_values("MD")
            if not sub.empty:
                md = sub["MD"].values
                ef = sub["ELECTROFACIES"].astype(int).values
                dz = np.abs(np.diff(md))
                dz = np.append(dz, dz[-1] if len(dz) else 0.1524)
                dz = np.where((dz <= 0) | np.isnan(dz), 0.1524, dz)
                palette = default_palette or ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#8D5B4C"]
                f_map = facies_colors or {}
                for i in range(len(md)):
                    fid = int(ef[i])
                    clr = f_map.get(fid, palette[(fid - 1) % len(palette)])
                    ax.barh(y=md[i], width=1.0, height=dz[i], left=0.0, color=clr, edgecolor="none", align="center")
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title("ELECTROFACIES", fontsize=10)
        else:
            ax.plot(w_df_view[col], w_df_view["MD"], linewidth=1.1)
            ax.set_title(col, fontsize=10)
            ax.grid(True, alpha=0.3)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("MD (m)")
    st.pyplot(fig)
    plt.close(fig)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

CLUSTER_ALGOS = [
    "K-Means",
    "Agglomerative (Hierarchical)",
    "Gaussian Mixture (GMM)",
    "MiniBatch K-Means",
]


def ensure_facies_palette():
    if "facies_colors" not in st.session_state or not isinstance(st.session_state.facies_colors, dict):
        st.session_state.facies_colors = {}


def get_facies_color(facies_id: int) -> str:
    ensure_facies_palette()
    idx = max(0, int(facies_id) - 1)
    default_hex = DEFAULT_FACIES_COLORS[idx % len(DEFAULT_FACIES_COLORS)]
    return st.session_state.facies_colors.get(int(facies_id), default_hex)


def run_clustering_model(algo_name: str, X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
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


def render_researcher_banner(mdf: pd.DataFrame, cdf: pd.DataFrame | None, selected_well: str, min_depth: float, max_depth: float):
    w_all = mdf[mdf["WELL_NAME"] == selected_well]
    w_view = w_all[(w_all["MD"] >= min_depth) & (w_all["MD"] <= max_depth)]
    core_cnt = int(len(cdf)) if cdf is not None and len(cdf) > 0 else int(w_all["CORE_K"].notna().sum()) if "CORE_K" in w_all else 0
    facies_cnt = int(w_view["ELECTROFACIES"].nunique(dropna=True)) if "ELECTROFACIES" in w_view else 0
    pay_cnt = int(w_view["PAY_FLAG"].fillna(0).sum()) if "PAY_FLAG" in w_view else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Well", str(selected_well))
    m2.metric("Depth Interval (m)", f"{min_depth:.1f} - {max_depth:.1f}")
    m3.metric("Electrofacies Found", facies_cnt)
    m4.metric("Core Samples", core_cnt)
    m5.metric("Pay Samples", pay_cnt)


def render_facies_bar_track(ax, df_well: pd.DataFrame):
    sub = df_well[["MD", "ELECTROFACIES"]].dropna().sort_values("MD")
    ax.set_title("ELECTROFACIES", fontsize=10, fontweight="bold")
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
    unique_ids = sorted(np.unique(ef).tolist())
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=get_facies_color(int(u)), edgecolor="black", linewidth=0.4)
        for u in unique_ids
    ]
    labels = [f"EF-{int(u)}" for u in unique_ids]
    ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)


def render_electrofacies_tab(mdf: pd.DataFrame, w_df_view: pd.DataFrame, selected_well: str, formation_name: str):
    ensure_facies_palette()
    st.subheader(f"Electrofacies Classification & Log Track — {selected_well} ({formation_name})")

    num_cols = [c for c in mdf.columns if pd.api.types.is_numeric_dtype(mdf[c]) and c != "MD"]
    default_curves = [c for c in ["GR", "VSH", "PHID", "NPHI", "RHOB"] if c in num_cols]

    c1, c2, c3, c4 = st.columns([2.2, 2.0, 1.0, 1.1])
    selected_curves = c1.multiselect("Input Curves for Clustering:", options=num_cols, default=default_curves, key="ui_ef_curves")
    algo_name = c2.selectbox("Clustering Algorithm:", options=CLUSTER_ALGOS, key="ui_ef_algo")
    n_facies = c3.number_input("Facies Count:", min_value=2, max_value=12, value=5, step=1, key="ui_ef_k")
    with c4:
        st.write("")
        st.write("")
        run_clicked = st.button("Classify Facies", type="primary", key="ui_ef_run")

    if run_clicked:
        if not selected_curves:
            st.error("Please select at least one numeric log curve.")
        else:
            valid_mask = mdf[selected_curves].notna().all(axis=1)
            if int(valid_mask.sum()) < int(n_facies):
                st.error("Not enough complete rows across all selected curves. Remove sparse curves or compute VSH/PHID first.")
            else:
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(mdf.loc[valid_mask, selected_curves])
                labels = run_clustering_model(algo_name, X_sc, int(n_facies))
                mdf["ELECTROFACIES"] = np.nan
                mdf.loc[valid_mask, "ELECTROFACIES"] = labels
                for fid in range(1, int(n_facies) + 1):
                    if fid not in st.session_state.facies_colors:
                        st.session_state.facies_colors[fid] = get_facies_color(fid)
                st.session_state.master_df = mdf
                st.session_state.ef_summary = (
                    mdf.groupby("ELECTROFACIES", dropna=True)[selected_curves].mean().round(4).to_dict(orient="index")
                )
                st.success(f"Classified {int(n_facies)} electrofacies using {algo_name}.")
                st.rerun()

    if "ELECTROFACIES" in mdf.columns and mdf["ELECTROFACIES"].notna().any():
        active_ids = sorted(mdf["ELECTROFACIES"].dropna().astype(int).unique().tolist())
        with st.expander("Customize Facies Colors", expanded=True):
            p_cols = st.columns(min(6, max(1, len(active_ids))))
            for idx, fid in enumerate(active_ids):
                current_hex = get_facies_color(fid)
                st.session_state.facies_colors[fid] = p_cols[idx % len(p_cols)].color_picker(
                    f"EF-{fid} Color",
                    value=current_hex,
                    key=f"ui_fc_{fid}",
                )

    if st.session_state.get("ef_summary"):
        st.markdown("**Electrofacies Mean Properties Table**")
        st.dataframe(pd.DataFrame.from_dict(st.session_state["ef_summary"], orient="index"), use_container_width=True)

    st.divider()
    log_opts = [c for c in num_cols if c != "ELECTROFACIES"]
    def_logs = [c for c in ["GR", "VSH", "PHID", "SW", "PERM_AI", "PAY_FLAG"] if c in w_df_view.columns and w_df_view[c].notna().any()]

    t1, t2 = st.columns([3, 1])
    chosen_logs = t1.multiselect("Log Tracks:", options=log_opts, default=def_logs, key="ui_log_tracks")
    with t2:
        st.write("")
        st.write("")
        show_ef = st.checkbox("Show Facies Track", value=True, key="ui_show_ef_track")

    tracks = list(chosen_logs) + (["ELECTROFACIES"] if show_ef and "ELECTROFACIES" in w_df_view.columns else [])
    if not w_df_view.empty and tracks:
        fig, axes = plt.subplots(1, len(tracks), sharey=True, figsize=(max(6.5, len(tracks) * 2.5), 9.5))
        if len(tracks) == 1:
            axes = [axes]
        for ax, col in zip(axes, tracks):
            if col == "ELECTROFACIES":
                render_facies_bar_track(ax, w_df_view)
            else:
                ax.plot(w_df_view[col], w_df_view["MD"], linewidth=1.15)
                ax.set_title(col, fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.35)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("MD (m)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_crossplot_tab(w_df_view: pd.DataFrame, selected_well: str):
    ensure_facies_palette()
    st.subheader(f"Crossplot Analysis — {selected_well}")
    num_cols = [c for c in w_df_view.columns if pd.api.types.is_numeric_dtype(w_df_view[c]) and c != "MD"]
    if len(num_cols) < 2 or w_df_view.empty:
        st.info("Not enough numeric curves to generate a crossplot.")
        return

    c1, c2, c3 = st.columns(3)
    x_var = c1.selectbox("X-Axis:", options=num_cols, index=0, key="ui_xp_x")
    y_var = c2.selectbox("Y-Axis:", options=num_cols, index=min(1, len(num_cols) - 1), key="ui_xp_y")
    color_opts = ["None", "ELECTROFACIES"] + [c for c in num_cols if c != "ELECTROFACIES"]
    def_c = 1 if "ELECTROFACIES" in w_df_view.columns and w_df_view["ELECTROFACIES"].notna().any() else 0
    c_var = c3.selectbox("Color By:", options=color_opts, index=def_c, key="ui_xp_c")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if c_var == "ELECTROFACIES" and w_df_view["ELECTROFACIES"].notna().any():
        for fid, grp in w_df_view.groupby("ELECTROFACIES"):
            if pd.isna(fid):
                continue
            fid_int = int(fid)
            ax.scatter(grp[x_var], grp[y_var], c=get_facies_color(fid_int), label=f"EF-{fid_int}", s=18, alpha=0.85)
        ax.legend(fontsize=8)
    elif c_var != "None" and c_var in w_df_view.columns:
        sc = ax.scatter(w_df_view[x_var], w_df_view[y_var], c=w_df_view[c_var], cmap="viridis", s=18, alpha=0.85)
        plt.colorbar(sc, ax=ax, label=c_var)
    else:
        ax.scatter(w_df_view[x_var], w_df_view[y_var], s=18, alpha=0.85)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_statistics_tab(w_df_view: pd.DataFrame, selected_well: str):
    ensure_facies_palette()
    st.subheader(f"Statistical Plots (Histogram & Boxplot) — {selected_well}")
    num_cols = [c for c in w_df_view.columns if pd.api.types.is_numeric_dtype(w_df_view[c]) and c != "MD"]
    if not num_cols or w_df_view.empty:
        st.info("No numeric variables available for statistical plots.")
        return

    c1, c2, c3 = st.columns(3)
    var_sel = c1.selectbox("Variable:", options=num_cols, key="ui_st_var")
    has_ef = "ELECTROFACIES" in w_df_view.columns and w_df_view["ELECTROFACIES"].notna().any()
    group_ef = c2.checkbox("Group by Electrofacies", value=has_ef, key="ui_st_grp")
    bins = c3.slider("Histogram Bins:", min_value=10, max_value=80, value=30, key="ui_st_bins")

    col_h, col_b = st.columns(2)
    with col_h:
        st.markdown("**Histogram**")
        fig_h, ax_h = plt.subplots(figsize=(6, 4.2))
        if group_ef and has_ef:
            for fid, grp in w_df_view.groupby("ELECTROFACIES"):
                if pd.isna(fid):
                    continue
                fid_int = int(fid)
                vals = grp[var_sel].dropna().values
                if len(vals) > 0:
                    ax_h.hist(vals, bins=bins, alpha=0.55, color=get_facies_color(fid_int), label=f"EF-{fid_int}")
            ax_h.legend(fontsize=8)
        else:
            vals = w_df_view[var_sel].dropna().values
            if len(vals) > 0:
                ax_h.hist(vals, bins=bins, alpha=0.8)
        ax_h.set_xlabel(var_sel)
        ax_h.set_ylabel("Count")
        ax_h.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    with col_b:
        st.markdown("**Boxplot**")
        fig_b, ax_b = plt.subplots(figsize=(6, 4.2))
        if group_ef and has_ef:
            groups, labels, fids = [], [], []
            for fid, grp in w_df_view.groupby("ELECTROFACIES"):
                if pd.isna(fid):
                    continue
                vals = grp[var_sel].dropna().values
                if len(vals) > 0:
                    fid_int = int(fid)
                    groups.append(vals)
                    labels.append(f"EF-{fid_int}")
                    fids.append(fid_int)
            if groups:
                bp = ax_b.boxplot(groups, tick_labels=labels, patch_artist=True)
                for patch, fid_int in zip(bp["boxes"], fids):
                    patch.set_facecolor(get_facies_color(fid_int))
                    patch.set_alpha(0.75)
        else:
            vals = w_df_view[var_sel].dropna().values
            if len(vals) > 0:
                ax_b.boxplot(vals, tick_labels=[var_sel], patch_artist=True)
        ax_b.set_ylabel(var_sel)
        ax_b.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)


def render_permeability_guide(mdf: pd.DataFrame, cdf: pd.DataFrame | None):
    with st.expander("How Permeability (CORE_K vs. PERM_AI) & Cutoffs Work", expanded=False):
        st.markdown(
            """
            - **CORE_K (Measured Core Permeability):** Extracted from your core CSV/XLSX table (or embedded column) and aligned by `WELL_NAME` and `MD` in mD.
            - **PERM_AI (Predicted Continuous Permeability):** Trained on `log10(CORE_K)` using selected log predictors (such as `PHID`, `VSH`, `GR`, `NPHI`, `RHOB`) and optional rolling-window depth context, then converted back to mD with uncertainty (`PERM_UNC`).
            - **Cutoffs & Reservoir Units (`RES_UNIT`):** Combines `VSH`, `PHID`, `SW`, and effective permeability (`PERM_AI` filled with `CORE_K`) to flag pay (`PAY_FLAG`), tight intervals (`TIGHT_FLAG`), and contiguous reservoir units above minimum thickness.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows with CORE_K", int(pd.to_numeric(mdf["CORE_K"], errors="coerce").gt(0).sum()) if "CORE_K" in mdf else 0)
        c2.metric("Rows with PERM_AI", int(mdf["PERM_AI"].notna().sum()) if "PERM_AI" in mdf else 0)
        c3.metric("Core Reference Rows", int(len(cdf)) if cdf is not None else 0)