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

# واجهة نظيفة ومباشرة بدون نصوص طويلة
st.title("🛢️ Zubair Electrofacies & Petrophysical Suite")

st.sidebar.header("📁 رفع بيانات الآبار")
uploaded_files = st.sidebar.file_uploader("ارفع ملفات الـ LAS الخاصة بالآبار هنا:", type=["las"], accept_multiple_files=True)

formation_name = st.sidebar.text_input("اسم التكوين الجيولوجي:", value="Zubair Formation")

if uploaded_files:
    all_wells_data = []
    
    for uploaded_file in uploaded_files:
        well_name = uploaded_file.name.split('.')[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        try:
            las = lasio.read(tmp_path)
            df = las.df().reset_index()
            
            depth_col = 'MD' if 'MD' in df.columns else 'DEPT'
            df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
            
            available_logs = [col for col in df.columns if col != depth_col]
            for col in available_logs:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['Well_Name'] = well_name
            all_wells_data.append(df)
        except Exception as e:
            st.error(f"خطأ في قراءة البئر {well_name}: {e}")
            
    if all_wells_data:
        master_df = pd.concat(all_wells_data, ignore_index=True)
        all_processed_logs = [col for col in master_df.columns if col not in ['MD', 'DEPT', 'Well_Name', 'Electrofacies', 'Facies_Name', 'Sw', 'VSH', 'PHID']]
        
        st.sidebar.markdown("---")
        st.sidebar.header("🛠️ لوحة التحكم والتحليل")
        
        # اختيار البئر ونطاق العمق
        wells_list = master_df['Well_Name'].unique().tolist()
        selected_well = st.sidebar.selectbox("اختر البئر للاستعراض:", options=wells_list)
        
        depth_col_name = 'MD' if 'MD' in master_df.columns else 'DEPT'
        well_raw_df = master_df[master_df['Well_Name'] == selected_well]
        
        well_min_d = float(well_raw_df[depth_col_name].min())
        well_max_d = float(well_raw_df[depth_col_name].max())
        
        min_depth = st.sidebar.number_input("العمق الأدنى (متر):", value=well_min_d)
        max_depth = st.sidebar.number_input("العمق الأقصى (متر):", value=well_max_d)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("1️⃣ الحسابات البتروفيزيائية الأساسية")
        compute_petro = st.sidebar.button("تنفيذ الحسابات (VSH, PHID, Sw)")
        
        if 'VSH' not in master_df.columns:
            master_df['VSH'] = np.nan
        if 'PHID' not in master_df.columns:
            master_df['PHID'] = np.nan
        if 'Sw' not in master_df.columns:
            master_df['Sw'] = np.nan
            
        rw_val = st.sidebar.slider("مقاومة ماء التكوين (Rw):", 0.01, 0.1, 0.03, 0.01)
        
        if compute_petro:
            if 'GR' in master_df.columns:
                gr_clean = master_df['GR'].quantile(0.05)
                gr_shale = master_df['GR'].quantile(0.95)
                igr = (master_df['GR'] - gr_clean) / (gr_shale - gr_clean)
                master_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            if 'RHOB' in master_df.columns:
                master_df['PHID'] = ((2.65 - master_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
                
            res_col = None
            for r in ['ILD', 'LLD', 'RT', 'RILD']:
                if r in master_df.columns:
                    res_col = r
                    break
            if res_col and 'PHID' in master_df.columns:
                phid_safe = master_df['PHID'].replace(0, 0.001)
                rt_safe = master_df[res_col].replace(0, 1)
                sw = np.power((1.0 * rw_val) / (np.power(phid_safe, 2.0) * rt_safe), 0.5)
                master_df['Sw'] = sw.clip(0, 1)
            st.sidebar.success("✅ تمت الحسابات البتروفيزيائية بنجاح!")
            
        # تحديث قائمة المجسات المتاحة بعد الحسابات
        current_logs = [col for col in master_df.columns if col not in ['MD', 'DEPT', 'Well_Name', 'Electrofacies', 'Facies_Name']]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("2️⃣ التصنيف غير الموجه للسحنات (Unsupervised Facies)")
        
        # اختيار خوارزمية التصنيف غير الموجه
        clustering_algo = st.sidebar.selectbox("اختر خوارزمية التصنيف:", options=[
            "K-Means Clustering", 
            "Gaussian Mixture Model (GMM)", 
            "Agglomerative Hierarchical Clustering"
        ])
        
        n_clusters = st.sidebar.slider("عدد السحنات (K / Clusters):", min_value=2, max_value=8, value=5)
        default_features = [f for f in ['GR', 'VSH', 'PHID'] if f in current_logs]
        selected_features = st.sidebar.multiselect("اختر المجسات المدخلة للتصنيف:", options=current_logs, default=default_features)
        
        run_clustering = st.sidebar.button("⚙️ تشغيل خوارزمية التصنيف")
        
        if 'Electrofacies' not in master_df.columns:
            master_df['Electrofacies'] = np.nan
            master_df['Facies_Name'] = "Unclassified"
            
        if run_clustering:
            if len(selected_features) < 2:
                st.error("⚠️ يرجى اختيار مجسين على الأقل للتصنيف.")
            else:
                valid_idx = master_df.dropna(subset=selected_features).index
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
                    
                master_df['Electrofacies'] = np.nan
                master_df.loc[valid_idx, 'Electrofacies'] = preds
                st.success(f"✅ تم تنفيذ التصنيف باستخدام ({clustering_algo}) لجميع الآبار بنجاح!")

        # تخصيص الألوان والأسماء
        st.sidebar.markdown("---")
        st.sidebar.subheader("3️⃣ تسمية السحنات وتخصيص الألوان")
        facies_names = {}
        facies_colors = {}
        default_color_list = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i in range(1, n_clusters + 1):
            default_name = "Tight Sandstone" if (i == 5 and n_clusters >= 5) else f"Facies {i}"
            facies_names[i] = st.sidebar.text_input(f"اسم السحنة {i}:", value=default_name)
            default_col = default_color_list[(i - 1) % len(default_color_list)]
            facies_colors[i] = st.sidebar.color_picker(f"لون سحنة {i}:", value=default_col)
            
        if 'Electrofacies' in master_df.columns:
            master_df['Facies_Name'] = master_df['Electrofacies'].map(facies_names)

        # التبويبات الرئيسية للعرض
        tab1, tab2, tab3, tab4 = st.tabs(["📊 لوحة مسارات البئر", "📉 الرسوم التقاطعية", "📐 التحقق الرياضي (Elbow)", "📥 تصدير النتائج"])
        
        single_well_df = well_raw_df[(well_raw_df[depth_col_name] >= min_depth) & (well_raw_df[depth_col_name] <= max_depth)].sort_values(by=depth_col_name)
        
        with tab1:
            st.subheader(f"لوحة المسارات للبئر: {selected_well}")
            chosen_tracks = st.multiselect("اختر المجسات للعرض في اللوحة:", options=current_logs, default=[t for t in ['GR', 'VSH', 'PHID'] if t in current_logs])
            
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
                        
                    d = single_well_df[depth_col_name]
                    for idx, track in enumerate(chosen_tracks):
                        axes[idx].plot(single_well_df[track], d, lw=1.2)
                        axes[idx].set_title(track, fontsize=9, fontweight='bold')
                        axes[idx].grid(True, linestyle='--', alpha=0.5)
                        
                    ax_f = axes[len(chosen_tracks)]
                    depths = d.values
                    facies_vals = single_well_df['Electrofacies'].values if 'Electrofacies' in single_well_df.columns else np.zeros_like(depths)
                    
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
                    
                    st.download_button("📥 تحميل لوحة المسارات (300 DPI)", data=buf, file_name=f"{selected_well}_Track.png", mime="image/png")
                    
                with col2:
                    st.subheader("📋 إحصائيات السحنات")
                    if 'Electrofacies' in single_well_df.columns and single_well_df['Electrofacies'].notna().sum() > 0:
                        stats = single_well_df.groupby(['Electrofacies', 'Facies_Name']).agg(
                            Count=(chosen_tracks[0], 'count'),
                            Mean_Value=(chosen_tracks[0], 'mean')
                        ).reset_index()
                        total_pts = len(single_well_df.dropna(subset=['Electrofacies']))
                        stats['Thickness_Percentage(%)'] = ((stats['Count'] / total_pts) * 100).round(2)
                        stats['Mean_Value'] = stats['Mean_Value'].round(3)
                        st.dataframe(stats, use_container_width=True)
                    else:
                        st.info("قم بتشغيل التصنيف غير الموجه أولاً لعرض الإحصائيات.")

        with tab2:
            st.subheader("📉 الرسوم التقاطعية التفاعلية (Crossplot)")
            if len(current_logs) >= 2:
                cp_x = st.selectbox("محور (X):", options=current_logs, index=0)
                cp_y = st.selectbox("محور (Y):", options=current_logs, index=1)
                
                fig_cp, ax_cp = plt.subplots(figsize=(8, 6))
                if 'Electrofacies' in master_df.columns and master_df['Electrofacies'].notna().sum() > 0:
                    for f_val in range(1, n_clusters + 1):
                        subset_f = master_df[master_df['Electrofacies'] == f_val]
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
                st.download_button("📥 تحميل الرسم التقاطعي (300 DPI)", data=buf_cp, file_name="Crossplot.png", mime="image/png")

        with tab3:
            st.subheader("📐 البرهان الرياضي لاختيار عدد السحنات (Elbow Method)")
            if st.button("حساب منحنى Elbow"):
                valid_idx = master_df.dropna(subset=default_features).index
                if len(valid_idx) > 0:
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(master_df.loc[valid_idx, default_features])
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
            st.download_button("تحميل جدول النتائج الشامل (CSV)", data=csv_data, file_name=f"{formation_name}_Results.csv", mime="text/csv")
            st.dataframe(master_df.head(100), use_container_width=True)
