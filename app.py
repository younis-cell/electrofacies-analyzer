import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import io

st.set_page_config(page_title="Zubair Multi-Well Electrofacies Suite", layout="wide")

st.title("🛢️ Zubair Multi-Well Electrofacies & Petrophysical Suite")
st.write("منصة أكاديمية متقدمة لدمج وتحليل آبار الحقل معاً، توحيد تصنيف السحنات، وتحديد نطاق الأعماق لرسالة الماجستير.")

st.sidebar.header("📁 إدارة ورفع الآبار")
uploaded_files = st.file_uploader("اختر ملفات الـ LAS الخاصة بالآبار (ارفع عدة آبار معاً)", type=["las"], accept_multiple_files=True)

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
            st.error(f"خطأ في قراءة ملف البئر {well_name}: {e}")
            
    if all_wells_data:
        master_df = pd.concat(all_wells_data, ignore_index=True)
        
        # حساب الخصائص البتروفيزيائية لكل الآبار معاً
        if 'GR' in master_df.columns:
            gr_clean = master_df['GR'].quantile(0.05)
            gr_shale = master_df['GR'].quantile(0.95)
            igr = (master_df['GR'] - gr_clean) / (gr_shale - gr_clean)
            master_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
        if 'RHOB' in master_df.columns and 'PHID' not in master_df.columns:
            master_df['PHID'] = ((2.65 - master_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
            
        res_col = None
        for r in ['ILD', 'LLD', 'RT', 'RILD']:
            if r in master_df.columns:
                res_col = r
                break
        
        if res_col and 'PHID' in master_df.columns:
            rw = st.sidebar.slider("مقاومة ماء التكوين (Rw):", 0.01, 0.1, 0.03, 0.01)
            phid_safe = master_df['PHID'].replace(0, 0.001)
            rt_safe = master_df[res_col].replace(0, 1)
            sw = np.power((1.0 * rw) / (np.power(phid_safe, 2.0) * rt_safe), 0.5)
            master_df['Sw'] = sw.clip(0, 1)
            
        all_processed_logs = [col for col in master_df.columns if col not in ['MD', 'DEPT', 'Well_Name']]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ التصنيف الموحد للسحنات (K-Means)")
        n_clusters = st.sidebar.slider("عدد السحنات الكهربائية الموحدة (K):", min_value=2, max_value=8, value=5)
        
        default_features = [f for f in ['GR', 'VSH', 'PHID'] if f in master_df.columns]
        selected_features = st.sidebar.multiselect("اختر المجسات المستخدمة لتصنيف السحنات الموحد:", options=all_processed_logs, default=default_features)
        
        if len(selected_features) < 2:
            st.warning("⚠️ يرجى اختيار مجسين على الأقل لتنفيذ التصنيف.")
        else:
            valid_idx = master_df.dropna(subset=selected_features).index
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(master_df.loc[valid_idx, selected_features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_results = kmeans.fit_predict(X_scaled)
            
            master_df['Electrofacies'] = np.nan
            master_df.loc[valid_idx, 'Electrofacies'] = cluster_results + 1
            
            st.success(f"✅ تم تدريب وتطبيق النموذج الموحد بنجاح على جميع الآبار ({len(uploaded_files)} آبار) معاً!")
            
            # --- تسمية السحنات والألوان الموحدة ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🏷️ تسمية السحنات وتخصيص ألوانها")
            facies_names = {}
            facies_colors = {}
            default_color_list = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            for i in range(1, n_clusters + 1):
                default_name = "Tight Sandstone" if (i == 5 and n_clusters >= 5) else f"Facies {i}"
                facies_names[i] = st.sidebar.text_input(f"اسم السحنة {i}:", value=default_name)
                default_col = default_color_list[(i - 1) % len(default_color_list)]
                facies_colors[i] = st.sidebar.color_picker(f"لون سحنة {i}:", value=default_col)
            
            master_df['Facies_Name'] = master_df['Electrofacies'].map(facies_names)
            
            # --- اختيار البئر ونطاق العمق المخصص له ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 استعراض البئر وتحديد نطاق العمق")
            wells_list = master_df['Well_Name'].unique().tolist()
            selected_well = st.sidebar.selectbox("اختر البئر لعرضه:", options=wells_list)
            
            depth_col_name = 'MD' if 'MD' in master_df.columns else 'DEPT'
            well_raw_df = master_df[master_df['Well_Name'] == selected_well]
            
            well_min_d = float(well_raw_df[depth_col_name].min())
            well_max_d = float(well_raw_df[depth_col_name].max())
            
            # إعادة تفعيل مربعات إدخال العمق لكل بئر بدقة
            min_depth = st.sidebar.number_input("العمق الأدنى للبئر (متر):", value=well_min_d)
            max_depth = st.sidebar.number_input("العمق الأقصى للبئر (متر):", value=well_max_d)
            
            single_well_df = well_raw_df[(well_raw_df[depth_col_name] >= min_depth) & (well_raw_df[depth_col_name] <= max_depth)].sort_values(by=depth_col_name)
            
            default_tracks = [t for t in ['GR', 'VSH', 'PHID'] if t in single_well_df.columns]
            chosen_tracks = st.sidebar.multiselect("اختر المجسات المراد عرضها في لوحة المسارات:", options=all_processed_logs, default=default_tracks)
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📉 إعدادات الرسم التقاطعي الشامل")
            x_var = st.sidebar.selectbox("محور (X):", options=all_processed_logs, index=all_processed_logs.index('PHID') if 'PHID' in all_processed_logs else 0)
            y_var = st.sidebar.selectbox("محور (Y):", options=all_processed_logs, index=all_processed_logs.index('GR') if 'GR' in all_processed_logs else 0)

            # --- التبويبات الأكاديمية ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 مسارات البئر والجدول", "📉 الرسوم التقاطعية الشاملة", "📐 التحقق الرياضي الموحد (Elbow)", "📥 تصدير البيانات الشاملة"])
            
            with tab1:
                if len(chosen_tracks) == 0:
                    st.warning("⚠️ يرجى اختيار مسار واحد على الأقل.")
                elif len(single_well_df) == 0:
                    st.warning("⚠️ لا توجد بيانات ضمن نطاق العمق المحدد لهذا البئر!")
                else:
                    st.info(f"📍 لوحة المسارات للبئر: **{selected_well}** ضمن العمق ({min_depth} - {max_depth} متر)")
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
                            
                        facies_ax_idx = len(chosen_tracks)
                        ax_f = axes[facies_ax_idx]
                        depths = d.values
                        facies_vals = single_well_df['Electrofacies'].values
                        
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
                        
                        st.download_button(
                            label=f"📥 تحميل لوحة مسارات البئر {selected_well} (300 DPI)",
                            data=buf,
                            file_name=f"{selected_well}_{formation_name}_Track.png",
                            mime="image/png"
                        )
                        
                    with col2:
                        st.subheader(f"📋 إحصائيات السحنات للبئر: {selected_well}")
                        stat_col = chosen_tracks[0] if len(chosen_tracks) > 0 else available_logs[0]
                        stats = single_well_df.groupby(['Electrofacies', 'Facies_Name']).agg(
                            Count=(stat_col, 'count'),
                            Mean_Value=(stat_col, 'mean')
                        ).reset_index()
                        
                        total_points = len(single_well_df.dropna(subset=['Electrofacies']))
                        if total_points > 0:
                            stats['Thickness_Percentage(%)'] = ((stats['Count'] / total_points) * 100).round(2)
                        else:
                            stats['Thickness_Percentage(%)'] = 0.0
                        stats['Mean_Value'] = stats['Mean_Value'].round(3)
                        
                        st.dataframe(stats, use_container_width=True)
            
            with tab2:
                st.subheader(f"📉 رسم تقاطعي شامل لكل آبار الحقل: ({x_var}) مقابل ({y_var})")
                fig_cp, ax_cp = plt.subplots(figsize=(8, 6))
                
                for f_val in range(1, n_clusters + 1):
                    subset_f = master_df[master_df['Electrofacies'] == f_val]
                    if not subset_f.empty:
                        col_hex = facies_colors.get(f_val, '#1f77b4')
                        f_label = facies_names.get(f_val, f"Facies {f_val}")
                        ax_cp.scatter(subset_f[x_var], subset_f[y_var], color=col_hex, label=f_label, alpha=0.6, s=20)
                        
                ax_cp.set_xlabel(x_var, fontweight='bold')
                ax_cp.set_ylabel(y_var, fontweight='bold')
                ax_cp.set_title(f'Multi-Well Unified Crossplot: {x_var} vs {y_var} - {formation_name}', fontweight='bold')
                ax_cp.grid(True, linestyle='--', alpha=0.5)
                ax_cp.legend(loc='best', frameon=True)
                
                buf_cp = io.BytesIO()
                fig_cp.savefig(buf_cp, format="png", dpi=300, bbox_inches='tight')
                buf_cp.seek(0)
                st.pyplot(fig_cp)
                
                st.download_button(
                    label="📥 تحميل الرسم التقاطعي الشامل لكل الآبار (300 DPI)",
                    data=buf_cp,
                    file_name=f"All_Wells_{formation_name}_Crossplot.png",
                    mime="image/png"
                )

            with tab3:
                st.subheader("📐 المبرر الرياضي الموحد لعدد السحنات (Elbow Method لكل الآبار مجتمعة)")
                st.write("هذا المنحنى مبني ومحسوب على مصفوفة بيانات **جميع الآبار المرفوعة معاً** ليكون البرهان الرياضي متكاملاً للحقل:")
                
                inertias = []
                k_range = range(1, 9)
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    
                fig_elb, ax_elb = plt.subplots(figsize=(8, 5))
                ax_elb.plot(k_range, inertias, marker='o', color='b', lw=2)
                ax_elb.set_xlabel('Number of Clusters (K)', fontweight='bold')
                ax_elb.set_ylabel('Inertia (Sum of squared distances)', fontweight='bold')
                ax_elb.set_title('Unified Multi-Well Elbow Method for Optimal K', fontweight='bold')
                ax_elb.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_elb)
                st.info("💡 استخدام هذا المنحنى الموحد لكل الآبار يمثل دليلاً قاطعاً وموثوقاً أمام لجنة المناقشة لاختيار عدد السحنات.")

            with tab4:
                st.subheader("📥 تصدير النتائج الموحدة لكل الآبار")
                csv_data = master_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="تحميل جدول النتائج والمسارات الشامل لكل الآبار (CSV)",
                    data=csv_data,
                    file_name=f"All_Wells_{formation_name}_Unified_Electrofacies_Results.csv",
                    mime="text/csv",
                )
                st.markdown("---")
                st.dataframe(master_df.head(100), use_container_width=True)
