import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tempfile
import io

st.set_page_config(page_title="Zubair Electrofacies Suite", layout="wide")

st.title("🛢️ Zubair Electrofacies & Petrophysical Suite")

st.sidebar.header("📁 رفع بيانات الآبار")
uploaded_files = st.sidebar.file_uploader("ارفع ملفات الـ LAS الخاصة بالآبار:", type=["las"], accept_multiple_files=True)

formation_name = st.sidebar.text_input("اسم التكوين الجيولوجي:", value="Zubair Formation")

# تهيئة Session State لحفظ البيانات لكي لا تختفي عند إعادة التحميل
if 'master_df' not in st.session_state:
    st.session_state.master_df = None

if uploaded_files:
    # قراءة الملفات فقط إذا لم يتم تحميلها مسبقاً أو إذا تغيرت الملفات
    if st.session_state.master_df is None:
        all_wells_data = []
        for uploaded_file in uploaded_files:
            well_name = uploaded_file.name.split('.')[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                las = lasio.read(tmp_path)
                df = las.df().reset_index()
                depth_col = 'MD' if 'MD' in df.columns else ('DEPT' if 'DEPT' in df.columns else df.columns[0])
                df = df.rename(columns={depth_col: 'MD'})
                df['MD'] = pd.to_numeric(df['MD'], errors='coerce')
                df.columns = [str(c).upper().strip() for c in df.columns]
                df['WELL_NAME'] = well_name
                all_wells_data.append(df)
            except Exception as e:
                st.error(f"خطأ في قراءة البئر {well_name}: {e}")
                
        if all_wells_data:
            st.session_state.master_df = pd.concat(all_wells_data, ignore_index=True)
            if 'VSH' not in st.session_state.master_df.columns:
                st.session_state.master_df['VSH'] = np.nan
            if 'PHID' not in st.session_state.master_df.columns:
                st.session_state.master_df['PHID'] = np.nan
            if 'SW' not in st.session_state.master_df.columns:
                st.session_state.master_df['SW'] = np.nan
            if 'ELECTROFACIES' not in st.session_state.master_df.columns:
                st.session_state.master_df['ELECTROFACIES'] = np.nan
                st.session_state.master_df['FACIES_NAME'] = "Unclassified"

    if st.session_state.master_df is not None:
        master_df = st.session_state.master_df
        exclude_cols = ['MD', 'WELL_NAME', 'ELECTROFACIES', 'FACIES_NAME']
        available_logs = [col for col in master_df.columns if col not in exclude_cols]
        
        st.sidebar.markdown("---")
        st.sidebar.header("🛠️ لوحة التحكم والتحليل اليدوي")
        
        wells_list = master_df['WELL_NAME'].unique().tolist()
        selected_well = st.sidebar.selectbox("اختر البئر للاستعراض:", options=wells_list)
        
        well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
        well_min_d = float(well_raw_df['MD'].min()) if not well_raw_df['MD'].isna().all() else 0.0
        well_max_d = float(well_raw_df['MD'].max()) if not well_raw_df['MD'].isna().all() else 1000.0
        
        min_depth = st.sidebar.number_input("العمق الأدنى (متر):", value=well_min_d)
        max_depth = st.sidebar.number_input("العمق الأقصى (متر):", value=well_max_d)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("1️⃣ الحسابات البتروفيزيائية")
        rw_val = st.sidebar.slider("مقاومة ماء التكوين (Rw):", 0.01, 0.1, 0.03, 0.01)
        compute_petro = st.sidebar.button("تنفيذ الحسابات (VSH, PHID, Sw)")
        
        if compute_petro:
            if 'GR' in master_df.columns:
                gr_clean = master_df['GR'].quantile(0.05)
                gr_shale = master_df['GR'].quantile(0.95)
                if gr_shale != gr_clean:
                    igr = (master_df['GR'] - gr_clean) / (gr_shale - gr_clean)
                    master_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            if 'RHOB' in master_df.columns:
                master_df['PHID'] = ((2.65 - master_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
                
            res_col = None
            for r in ['ILD', 'LLD', 'RT', 'RILD', 'RESD']:
                if r in master_df.columns:
                    res_col = r
                    break
            if res_col and 'PHID' in master_df.columns:
                phid_safe = master_df['PHID'].replace(0, 0.001)
                rt_safe = master_df[res_col].replace(0, 1)
                sw = np.power((1.0 * rw_val) / (np.power(phid_safe, 2.0) * rt_safe), 0.5)
                master_df['SW'] = sw.clip(0, 1)
            st.session_state.master_df = master_df
            st.sidebar.success("✅ تمت الحسابات بنجاح!")

        current_logs = [col for col in master_df.columns if col not in exclude_cols]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("2️⃣ التصنيف غير الموجه للسحنات")
        clustering_algo = st.sidebar.selectbox("اختر الخوارزمية:", options=[
            "K-Means Clustering", 
            "Gaussian Mixture Model (GMM)", 
            "Agglomerative Hierarchical Clustering"
        ])
        
        n_clusters = st.sidebar.slider("عدد السحنات (K):", min_value=2, max_value=8, value=5)
        default_feats = [f for f in ['GR', 'VSH', 'PHID'] if f in current_logs]
        selected_features = st.sidebar.multiselect("اختر المجسات المدخلة للتصنيف:", options=current_logs, default=default_feats)
        
        run_clustering = st.sidebar.button("⚙️ تشغيل خوارزمية التصنيف")
        
        if run_clustering:
            if len(selected_features) < 2:
                st.error("⚠️ يرجى اختيار مجسين على الأقل للتصنيف.")
            else:
                valid_idx = master_df.dropna(subset=selected_features).index
                if len(valid_idx) > 0:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(master_df.loc[valid_idx, selected_features])
                    
                    if clustering_algo == "K-Means Clustering":
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        preds = model.fit_predict(X_scaled) + 1
                    elif clustering_algo == "Gaussian Mixture Model (GMM)":
                        model = GaussianMixture(n_components=n_clusters, random_state=42)
                        preds = model.fit_predict(X_scaled) + 1
                    else:
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                        preds = model.fit_predict(X_scaled) + 1
                        
                    master_df['ELECTROFACIES'] = np.nan
                    master_df.loc[valid_idx, 'ELECTROFACIES'] = preds
                    st.session_state.master_df = master_df
                    st.sidebar.success(f"✅ تم التصنيف وحفظ النتائج بنجاح!")

        # تسمية السحنات والألوان
        st.sidebar.markdown("---")
        st.sidebar.subheader("3️⃣ تسمية السحنات والألوان")
        facies_names = {}
        facies_colors = {}
        default_color_list = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i in range(1, n_clusters + 1):
            default_name = "Tight Sandstone" if (i == 5 and n_clusters >= 5) else f"Facies {i}"
            facies_names[i] = st.sidebar.text_input(f"اسم السحنة {i}:", value=default_name)
            default_col = default_color_list[(i - 1) % len(default_color_list)]
            facies_colors[i] = st.sidebar.color_picker(f"لون سحنة {i}:", value=default_col)
            
        if 'ELECTROFACIES' in master_df.columns:
            master_df['FACIES_NAME'] = master_df['ELECTROFACIES'].map(facies_names)
            st.session_state.master_df = master_df

        # التبويبات
        tab1, tab2, tab3, tab4 = st.tabs(["📊 لوحة المسارات", "📉 الرسوم التقاطعية", "📐 التحقق الرياضي (Elbow)", "📥 تصدير النتائج"])
        
        single_well_df = well_raw_df[(well_raw_df['MD'] >= min_depth) & (well_raw_df['MD'] <= max_depth)].sort_values(by='MD')
        
        with tab1:
            st.subheader(f"لوحة المسارات للبئر: {selected_well}")
            default_tr = [t for t in ['GR', 'VSH', 'PHID'] if t in current_logs]
            chosen_tracks = st.multiselect("اختر المجسات للعرض:", options=current_logs, default=default_tr, key="chosen_tracks_tab1")
            
            if len(chosen_tracks) == 0:
                st.warning("⚠️ اختر مساراً واحداً على الأقل للعرض.")
            elif len(single_well_df) == 0:
                st.warning("⚠️ لا توجد بيانات ضمن نطاق العمق المحدد.")
            else:
                col1, col2 = st.columns([1.3, 1])
                with col1:
                    n_plots = len(chosen_tracks) + 1  
                    fig, axes = plt.subplots(1, n_plots, figsize=(2.3 * n_plots, 8), sharey=True)
                    if n_plots == 1:
                        axes = [axes]
                        
                    d = single_well_df['MD']
                    for idx, track in enumerate(chosen_tracks):
                        axes[idx].plot(single_well_df[track], d, lw=1.2)
                        axes[idx].set_title(track, fontsize=9, fontweight='bold')
                        axes[idx].grid(True, linestyle='--', alpha=0.5)
                        
                    ax_f = axes[len(chosen_tracks)]
                    depths = d.values
                    facies_vals = single_well_df['ELECTROFACIES'].values if 'ELECTROFACIES' in single_well_df.columns else np.zeros_like(depths)
                    
                    for j in range(len(depths) - 1):
                        f_val = facies_vals[j]
                        if not np.isnan(f_val):
                            col_hex = facies_colors.get(int(f_val), '#1f77b4')
                            ax_f.fill_betweenx([depths[j], depths[j+1]], 0, 1, color=col_hex, linewidth=0)
                            
                    ax_f.set_xlim(0, 1)
                    ax_f.set_xticks([0.5])
                    ax_f.set_xticklabels(['Facies'], fontsize=8, fontweight='bold')
                    ax_f.set_title('Electrofacies', fontsize=9, fontweight='bold')
                    ax_f.grid(False)
                    
                    for ax in axes:
                        ax.set_ylim(d.max(), d.min())
                    axes[0].set_ylabel('Depth (m)', fontsize=9, fontweight='bold')
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.pyplot(fig)
                    st.download_button("📥 تحميل لوحة المسارات (300 DPI)", data=buf, file_name=f"{selected_well}_Track.png", mime="image/png", key="dl_track")
                    
                with col2:
                    st.subheader("📋 إحصائيات السحنات")
                    if 'ELECTROFACIES' in single_well_df.columns and single_well_df['ELECTROFACIES'].notna().sum() > 0:
                        stat_tr = chosen_tracks[0] if len(chosen_tracks) > 0 else current_logs[0]
                        stats = single_well_df.groupby(['ELECTROFACIES', 'FACIES_NAME']).agg(
                            Count=(stat_tr, 'count'),
                            Mean_Value=(stat_tr, 'mean')
                        ).reset_index()
                        total_pts = len(single_well_df.dropna(subset=['ELECTROFACIES']))
                        stats['Thickness_Percentage(%)'] = ((stats['Count'] / total_pts) * 100).round(2) if total_pts > 0 else 0.0
                        stats['Mean_Value'] = stats['Mean_Value'].round(3)
                        st.dataframe(stats, use_container_width=True)
                    else:
                        st.info("قم بتشغيل التصنيف غير الموجه من القائمة الجانبية لعرض الإحصائيات.")

        with tab2:
            st.subheader("📉 الرسوم التقاطعية التفاعلية (Crossplot)")
            if len(current_logs) >= 2:
                cp_x = st.selectbox("محور (X):", options=current_logs, index=0, key="cp_x")
                cp_y = st.selectbox("محور (Y):", options=current_logs, index=1, key="cp_y")
                
                fig_cp, ax_cp = plt.subplots(figsize=(8, 6))
                if 'ELECTROFACIES' in master_df.columns and master_df['ELECTROFACIES'].notna().sum() > 0:
                    for f_val in range(1, n_clusters + 1):
                        subset_f = master_df[master_df['ELECTROFACIES'] == f_val]
                        if not subset_f.empty:
                            col_hex = facies_colors.get(f_val, '#1f77b4')
                            f_label = facies_names.get(f_val, f"Facies {f_val}")
                            ax_cp.scatter(subset_f[cp_x], subset_f[cp_y], color=col_hex, label=f_label, alpha=0.6, s=20)
                else:
                    ax_cp.scatter(master_df[cp_x], master_df[cp_y], color='gray', alpha=0.5, s=20)
                    
                ax_cp.set_xlabel(cp_x, fontweight='bold')
                ax_cp.set_ylabel(cp_y, fontweight='bold')
                ax_cp.set_title(f'Crossplot: {cp_x} vs {cp_y}', fontweight='bold')
                ax_cp.grid(True, linestyle='--', alpha=0.5)
                ax_cp.legend(loc='best')
                
                buf_cp = io.BytesIO()
                fig_cp.savefig(buf_cp, format="png", dpi=300, bbox_inches='tight')
                buf_cp.seek(0)
                st.pyplot(fig_cp)
                st.download_button("📥 تحميل الرسم التقاطعي (300 DPI)", data=buf_cp, file_name="Crossplot.png", mime="image/png", key="dl_cp")

        with tab3:
            st.subheader("📐 البرهان الرياضي لاختيار عدد السحنات (Elbow Method)")
            if st.button("حساب منحنى Elbow", key="btn_elbow"):
                valid_idx = master_df.dropna(subset=default_feats).index
                if len(valid_idx) > 0:
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(master_df.loc[valid_idx, default_feats])
                    inertias = []
                    k_rng = range(1, 9)
                    for k in k_rng:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(X_sc)
                        inertias.append(km.inertia_)
                        
                    fig_elb, ax_elb = plt.subplots(figsize=(8, 5))
                    ax_elb.plot(k_rng, inertias, marker='o', color='b', lw=2)
                    ax_elb.set_xlabel('Number of Clusters (K)', fontweight='bold')
                    ax_elb.set_ylabel('Inertia', fontweight='bold')
                    ax_elb.set_title('Elbow Method', fontweight='bold')
                    ax_elb.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig_elb)

        with tab4:
            st.subheader("📥 تصدير النتائج")
            csv_data = master_df.to_csv(index=False).encode('utf-8')
            st.download_button("تحميل جدول النتائج الشامل (CSV)", data=csv_data, file_name=f"{formation_name}_Results.csv", mime="text/csv", key="dl_csv")
            st.dataframe(master_df.head(100), use_container_width=True)
