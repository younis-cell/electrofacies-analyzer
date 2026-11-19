import io
import json
import zipfile
import tempfile
import os

import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EFA | ElectroFacies Analyzer", layout="wide")
st.title("🛢️ EFA — ElectroFacies Analyzer")
st.caption("A formation-agnostic workflow for LAS quality control, petrophysical calculations and electrofacies analysis.")

# -------------------------- Session state --------------------------
DEFAULTS = {
    "data": None,
    "facies_names": {},
    "facies_colors": {},
    "track_props": {},
    "analysis_info": {},
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, dict) else value

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def numeric_frame(frame, columns):
    out = frame.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_las(upload):
    """Read LAS from a temporary file and normalize common depth naming."""
    suffix = os.path.splitext(upload.name)[1] or ".las"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getvalue())
            path = tmp.name
        las = lasio.read(path)
        frame = las.df().reset_index()
        if frame.empty:
            raise ValueError("The LAS file contains no log samples.")
        depth_candidates = ["MD", "DEPT", "DEPTH", "MDEP", "TDEP"]
        depth_col = next((c for c in frame.columns if str(c).strip().upper() in depth_candidates), frame.columns[0])
        frame = frame.rename(columns={depth_col: "MD"})
        frame.columns = [str(c).strip().upper() for c in frame.columns]
        if "MD" not in frame:
            raise ValueError("No depth column could be identified.")
        frame = numeric_frame(frame, [c for c in frame.columns if c != "WELL_NAME"])
        frame["WELL_NAME"] = os.path.splitext(os.path.basename(upload.name))[0]
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["MD"]).sort_values("MD")
        if frame.empty:
            raise ValueError("No valid numeric depth values were found.")
        return frame
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def calculate_petrophysics(data, gr_min, gr_max, rho_matrix, rho_fluid, rw, archie_a, archie_m, archie_n, rt_curve):
    result = data.copy()
    # Each well is treated independently: log ranges and calculations must not
    # be mixed across wells with different acquisition ranges.
    for well in result["WELL_NAME"].dropna().unique():
        mask = result["WELL_NAME"] == well
        part = result.loc[mask]
        if "GR" in result.columns:
            clean = gr_min if gr_min is not None else part["GR"].quantile(0.05)
            shale = gr_max if gr_max is not None else part["GR"].quantile(0.95)
            if pd.notna(clean) and pd.notna(shale) and shale > clean:
                igr = ((part["GR"] - clean) / (shale - clean)).clip(0, 1)
                result.loc[mask, "VSH"] = (0.083 * (np.power(2.0, 3.7 * igr) - 1.0)).clip(0, 1)
        if "RHOB" in result.columns and rho_matrix > rho_fluid:
            result.loc[mask, "PHID"] = ((rho_matrix - part["RHOB"]) / (rho_matrix - rho_fluid)).clip(0, 1)
        if rt_curve and rt_curve in result.columns and "PHID" in result.columns:
            phi = result.loc[mask, "PHID"].clip(lower=0.001)
            rt = result.loc[mask, rt_curve].where(result.loc[mask, rt_curve] > 0)
            # Archie is an ideal clean-formation model; clay-bearing formations
            # may require a shaly-sand model and independent calibration.
            sw = np.power((archie_a * rw) / (rt * np.power(phi, archie_m)), 1.0 / archie_n)
            result.loc[mask, "SW"] = sw.clip(0, 1)
    return result


def make_project_bytes(data):
    metadata = {
        "format": "efa-project",
        "version": 1,
        "facies_names": st.session_state.facies_names,
        "facies_colors": st.session_state.facies_colors,
        "track_props": st.session_state.track_props,
        "analysis_info": st.session_state.analysis_info,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.csv", data.to_csv(index=False))
        archive.writestr("project.json", json.dumps(metadata, ensure_ascii=False, default=str))
    return buffer.getvalue()


def read_project(upload):
    # ZIP/CSV/JSON only: never unpickle user-uploaded files.
    with zipfile.ZipFile(upload) as archive:
        names = set(archive.namelist())
        if not {"data.csv", "project.json"}.issubset(names):
            raise ValueError("Project archive must contain data.csv and project.json.")
        data = pd.read_csv(io.BytesIO(archive.read("data.csv")))
        metadata = json.loads(archive.read("project.json").decode("utf-8"))
    if "WELL_NAME" not in data or "MD" not in data:
        raise ValueError("Project data is missing WELL_NAME or MD.")
    return data, metadata


# -------------------------- Project controls --------------------------
with st.sidebar:
    st.header("📁 Project")
    project_upload = st.file_uploader("Open EFA project (.efa)", type=["efa"])
    if project_upload and st.button("Open project"):
        try:
            loaded_data, meta = read_project(project_upload)
            st.session_state.data = loaded_data
            for key in ["facies_names", "facies_colors", "track_props", "analysis_info"]:
                st.session_state[key] = meta.get(key, {})
            st.success("Project opened.")
        except Exception as exc:
            st.error(f"Could not open project: {exc}")

    las_uploads = st.file_uploader("Add LAS well files", type=["las"], accept_multiple_files=True)
    if las_uploads and st.button("Load LAS files"):
        frames, errors = [], []
        for item in las_uploads:
            try:
                frames.append(load_las(item))
            except Exception as exc:
                errors.append(f"{item.name}: {exc}")
        for error in errors:
            st.error(error)
        if frames:
            incoming = pd.concat(frames, ignore_index=True, sort=False)
            if st.session_state.data is not None:
                st.session_state.data = pd.concat([st.session_state.data, incoming], ignore_index=True, sort=False)
            else:
                st.session_state.data = incoming
            st.success(f"Loaded {len(frames)} LAS file(s).")

data = st.session_state.data
if data is None or data.empty:
    st.info("Upload one or more LAS files from the sidebar to begin.")
    st.markdown("**EFA is designed for different wells and formations.** Curve mnemonics and petrophysical assumptions should be checked against each dataset before interpretation.")
    st.stop()

data = data.copy()
for col in data.columns:
    if col not in ["WELL_NAME"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
st.session_state.data = data

# -------------------------- Data overview --------------------------
with st.expander("Dataset overview and quality checks", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Wells", data["WELL_NAME"].nunique())
    c2.metric("Samples", f"{len(data):,}")
    c3.metric("Log curves", len([c for c in data.columns if c not in ["MD", "WELL_NAME"]]))
    st.dataframe(data.groupby("WELL_NAME").agg(
        Samples=("MD", "count"), Min_depth=("MD", "min"), Max_depth=("MD", "max")
    ).reset_index(), use_container_width=True)
    missing = data.isna().mean().mul(100).sort_values(ascending=False).rename("Missing (%)").reset_index(names="Curve")
    st.dataframe(missing, use_container_width=True)

numeric_logs = [c for c in data.columns if c not in ["MD", "WELL_NAME", "ELECTROFACIES", "FACIES_NAME"] and pd.api.types.is_numeric_dtype(data[c])]
wells = data["WELL_NAME"].dropna().unique().tolist()
if not wells or not numeric_logs:
    st.error("The dataset needs at least one well and one numeric log curve.")
    st.stop()

# -------------------------- Sidebar analysis controls --------------------------
with st.sidebar:
    st.divider()
    st.header("⚙️ Analysis")
    selected_well = st.selectbox("Well for log display", wells)
    well_data = data[data["WELL_NAME"] == selected_well]
    depth_min = float(well_data["MD"].min())
    depth_max = float(well_data["MD"].max())
    min_depth = st.number_input("Top depth", value=depth_min)
    max_depth = st.number_input("Base depth", value=depth_max)
    st.subheader("Petrophysical parameters")
    gr_min_text = st.text_input("GR clean baseline (blank = well P05)", "")
    gr_max_text = st.text_input("GR shale baseline (blank = well P95)", "")
    rho_matrix = st.number_input("Matrix density (g/cc)", min_value=1.0, max_value=4.0, value=2.65, step=0.01)
    rho_fluid = st.number_input("Fluid density (g/cc)", min_value=0.5, max_value=2.0, value=1.0, step=0.01)
    rw = st.number_input("Rw (ohm.m)", min_value=0.0001, max_value=10.0, value=0.03, format="%.4f")
    archie_a = st.number_input("Archie a", min_value=0.01, max_value=5.0, value=1.0, step=0.05)
    archie_m = st.number_input("Archie m", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    archie_n = st.number_input("Archie n", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    rt_curve = st.selectbox("Resistivity curve for Archie", ["(none)"] + [c for c in numeric_logs if c in ["ILD", "LLD", "LLS", "RT", "RILD", "RESD", "AT90", "RLA5"]])
    if st.button("Calculate petrophysical curves"):
        try:
            gmin = float(gr_min_text) if gr_min_text.strip() else None
            gmax = float(gr_max_text) if gr_max_text.strip() else None
            if (gmin is None) != (gmax is None):
                raise ValueError("Enter both GR baselines, or leave both blank.")
            data = calculate_petrophysics(data, gmin, gmax, rho_matrix, rho_fluid, rw, archie_a, archie_m, archie_n, None if rt_curve == "(none)" else rt_curve)
            st.session_state.data = data
            st.success("Calculated curves. Review assumptions before interpretation.")
        except Exception as exc:
            st.error(str(exc))
    st.caption("VSH: Larionov Tertiary approximation using per-well P05/P95 unless baselines are supplied. PHID: density porosity. SW: Archie clean-formation model.")

# refresh after calculations
data = st.session_state.data.copy()
numeric_logs = [c for c in data.columns if c not in ["MD", "WELL_NAME", "ELECTROFACIES", "FACIES_NAME"] and pd.api.types.is_numeric_dtype(data[c])]

# -------------------------- Tabs --------------------------
tab_logs, tab_cluster, tab_cross, tab_export = st.tabs(["📊 Well logs", "🧩 Electrofacies", "📉 Crossplot", "📤 Export"])

with tab_logs:
    st.subheader(f"Well log panel — {selected_well}")
    view = data[(data["WELL_NAME"] == selected_well) & data["MD"].between(min_depth, max_depth)].sort_values("MD")
    default_tracks = [c for c in ["GR", "VSH", "RHOB", "PHID", "NPHI", "SW"] if c in numeric_logs]
    tracks = st.multiselect("Curves to display", numeric_logs, default=default_tracks)
    if view.empty:
        st.warning("No samples in the selected depth interval.")
    elif not tracks:
        st.info("Select at least one curve.")
    else:
        fig, axes = plt.subplots(1, len(tracks) + 1, figsize=(2.5 * (len(tracks) + 1), 10), sharey=True, squeeze=False)
        axes = axes[0]
        for i, curve in enumerate(tracks):
            axes[i].plot(view[curve], view["MD"], lw=1.0)
            axes[i].set_title(curve)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(view["MD"].max(), view["MD"].min())
        facies_ax = axes[-1]
        if "ELECTROFACIES" in view:
            vals = view["ELECTROFACIES"].to_numpy()
            depths = view["MD"].to_numpy()
            for j in range(max(0, len(view) - 1)):
                if pd.notna(vals[j]):
                    color = st.session_state.facies_colors.get(str(int(vals[j])), "#999999")
                    facies_ax.fill_betweenx([depths[j], depths[j + 1]], 0, 1, color=color, linewidth=0)
        facies_ax.set_xlim(0, 1)
        facies_ax.set_xticks([])
        facies_ax.set_title("Facies")
        facies_ax.set_ylim(view["MD"].max(), view["MD"].min())
        axes[0].set_ylabel("Measured depth")
        fig.tight_layout()
        st.pyplot(fig)
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", dpi=300, bbox_inches="tight")
        st.download_button("Download log panel (PNG, 300 DPI)", image_buffer.getvalue(), f"{selected_well}_logs.png", "image/png")
        plt.close(fig)

with tab_cluster:
    st.subheader("Unsupervised electrofacies classification")
    scope = st.radio("Classification scope", ["Each well independently", "All selected wells together"], horizontal=True)
    features = st.multiselect("Input curves", numeric_logs, default=[c for c in ["GR", "RHOB", "NPHI", "DT", "VSH", "PHID"] if c in numeric_logs])
    algorithm = st.selectbox("Algorithm", ["K-Means", "Gaussian Mixture", "Agglomerative"])
    k = st.slider("Number of clusters", 2, 10, 4)
    if st.button("Run electrofacies classification"):
        if len(features) < 2:
            st.error("Select at least two input curves.")
        else:
            updated = data.copy()
            updated["ELECTROFACIES"] = np.nan
            scores = []
            groups = [(name, updated["WELL_NAME"] == name) for name in wells] if scope == "Each well independently" else [("Combined", pd.Series(True, index=updated.index))]
            errors = []
            for group_name, mask in groups:
                valid_idx = updated.index[mask & updated[features].notna().all(axis=1)]
                if len(valid_idx) <= k:
                    errors.append(f"{group_name}: needs more than {k} complete samples (found {len(valid_idx)}).")
                    continue
                scaler = StandardScaler()
                X = scaler.fit_transform(updated.loc[valid_idx, features])
                try:
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=k, random_state=42, n_init=20)
                        labels = model.fit_predict(X)
                    elif algorithm == "Gaussian Mixture":
                        labels = GaussianMixture(n_components=k, random_state=42, n_init=3).fit_predict(X)
                    else:
                        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
                    updated.loc[valid_idx, "ELECTROFACIES"] = labels + 1
                    unique_count = len(np.unique(labels))
                    sil = silhouette_score(X, labels) if 1 < unique_count < len(X) else np.nan
                    db = davies_bouldin_score(X, labels) if 1 < unique_count else np.nan
                    scores.append({"Group": group_name, "Samples": len(valid_idx), "Clusters found": unique_count, "Silhouette": sil, "Davies-Bouldin": db})
                except Exception as exc:
                    errors.append(f"{group_name}: {exc}")
            st.session_state.data = updated
            st.session_state.analysis_info = {"features": features, "algorithm": algorithm, "k": k, "scope": scope}
            for error in errors:
                st.warning(error)
            if scores:
                st.success("Classification finished. Cluster labels are numerical groups, not geological facies names.")
                st.dataframe(pd.DataFrame(scores), use_container_width=True)
    st.caption("Silhouette: higher is generally better. Davies–Bouldin: lower is generally better. Neither metric replaces geological validation.")
    if "ELECTROFACIES" in data and data["ELECTROFACIES"].notna().any():
        counts = data.groupby(["WELL_NAME", "ELECTROFACIES"], dropna=True).size().rename("Samples").reset_index()
        st.dataframe(counts, use_container_width=True)

with tab_cross:
    st.subheader("Crossplot")
    cx1, cx2, cx3 = st.columns(3)
    x_curve = cx1.selectbox("X axis", numeric_logs, index=0, key="cross_x")
    y_curve = cx2.selectbox("Y axis", numeric_logs, index=min(1, len(numeric_logs)-1), key="cross_y")
    cross_well = cx3.selectbox("Well", ["All wells"] + wells)
    subset = data if cross_well == "All wells" else data[data["WELL_NAME"] == cross_well]
    fig, ax = plt.subplots(figsize=(8, 6))
    if "ELECTROFACIES" in subset and subset["ELECTROFACIES"].notna().any():
        scatter = ax.scatter(subset[x_curve], subset[y_curve], c=subset["ELECTROFACIES"], cmap="tab10", s=12, alpha=0.65)
        fig.colorbar(scatter, ax=ax, label="Electrofacies")
    else:
        ax.scatter(subset[x_curve], subset[y_curve], s=12, alpha=0.55)
    ax.set_xlabel(x_curve)
    ax.set_ylabel(y_curve)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab_export:
    st.subheader("Export data and project")
    csv_bytes = data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download processed data (CSV)", csv_bytes, "efa_results.csv", "text/csv")
    st.download_button("Save EFA project (.efa)", make_project_bytes(data), "efa_project.efa", "application/zip")
    st.dataframe(data.head(200), use_container_width=True)
