import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import matplotlib.colors as mcolors

st.set_page_config(page_title="Electrofacies & Petrophysical Analyzer", layout="wide")

st.title("🛢️ Advanced Electrofacies & Petrophysical Analyzer")
st.write("أداة بحثية متقدمة لتصنيف السحنات الكهربائية، تخصيص الألوان، والرسوم التقاطعية التفاعلية لرسائل الماجستير.")

st.sidebar.header("📁 بيانات البئر ونطاق العمق")
uploaded_file = st.file_uploader("اختر ملف الـ LAS الخاص ببيانات البئر", type=["las"])

formation_name = st.sidebar.text_input("اسم التكوين الجيولوجي:", value="Zubair Formation")

if uploaded_file is not None:
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
            
        min_d_val = float(df[depth_col].min())
        max_d_val = float(df[depth_col].max())
        
        st.sidebar.markdown("---")
        min_depth = st.sidebar.number_input("العمق الأدنى (متر):", value=min_d_val)
        max_depth = st.sidebar.number_input("العمق الأقصى (متر):", value=max_d_val)
        n_clusters = st.sidebar.slider("عدد السحنات الكهربائية (K):", min_value=2, max_value=8, value=5)
        
        sub_df = df[(df[depth_col] >= min_depth) & (df[depth_col] <= max_depth)].copy()
        
        if len(sub_df) == 0:
            st.warning("⚠️ لا توجد بيانات ضمن نطاق العمق المحدد لهذا البئر!")
        else:
            if 'GR' in sub_df.columns:
                gr_clean = sub_df['GR'].quantile(0.05)
                gr_shale = sub_df['GR'].quantile(0.95)
                igr = (sub_df['GR'] - gr_clean) / (gr_shale - gr_clean)
                sub_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            if 'RHOB' in sub_df.columns and 'PHID' not in sub_df.columns:
                sub_df['PHID'] = ((2.65 - sub_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
                
            all_processed_logs = [col for col in sub_df.columns if col != depth_col]
            
            default_features = [f for f in ['GR', 'VSH', 'PHID'] if f in sub_df.columns]
            st.sidebar.markdown("---")
            st.sidebar.subheader("⚙️ متغيرات التصنيف الآلي (K-Means)")
            selected_features = st.sidebar.multiselect("اختر المجسات المستخدمة لتصنيف السحنات:", options=all_processed_logs, default=default_features)
            
            if len(selected_features) < 2:
                st.warning("⚠️ يرجى اختيار مجسين على الأقل لتنفيذ تصنيف السحنات الكهربائية.")
            else:
                model_df = sub_df.dropna(subset=selected_features).copy()
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(model_df[selected_features])
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                model_df['Electrofacies'] = kmeans.fit_predict(X_scaled) + 1
                
                st.success(f"✅ تمت معالجة وتصنيف السحنات لـ ({formation_name}) بنجاح!")
                
                # --- تسمية السحنات يدوياً وتخصيص الألوان ---
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
                
                model_df['Facies_Name'] = model_df['Electrofacies'].map(facies_names)
                
                # --- لوحة التحكم بالمسارات للرسم ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("📊 تخصيص المسارات")
                default_tracks = [t for t in ['GR', 'VSH', 'PHID'] if t in model_df.columns]
                chosen_tracks = st.sidebar.multiselect("اختر المجسات المراد عرضها:", options=all_processed_logs, default=default_tracks)
                
                # --- إعدادات Crossplot ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("📉 إعدادات الرسم التقاطعي (Crossplot)")
                x_var = st.sidebar.selectbox("محور (X):", options=all_processed_logs, index=all_processed_logs.index('PHID') if 'PHID' in all_processed_logs else 0)
                y_var = st.sidebar.selectbox("محور (Y):", options=all_processed_logs, index=all_processed_logs.index('GR') if 'GR' in all_processed_logs else 0)

                # --- التبويبات (Tabs) ---
                tab1, tab2, tab3 = st.tabs(["📊 المسارات والجدول", "📉 الرسوم التقاطعية التفاعلية", "📥 بيانات التصدير"])
                
                with tab1:
                    if len(chosen_tracks) == 0:
                        st.warning("⚠️ يرجى اختيار مسار واحد على الأقل من القائمة الجانبية.")
                    else:
                        col1, col2 = st.columns([1.3, 1])
                        
                        with col1:
                            n_plots = len(chosen_tracks) + 1  
                            fig, axes = plt.subplots(1, n_plots, figsize=(2.3 * n_plots, 8), sharey=True)
                            if n_plots == 1:
                                axes = [axes]
                                
                            d = model_df[depth_col]
                            
                            for idx, track in enumerate(chosen_tracks):
                                axes[idx].plot(model_df[track], d, lw=1.2)
                                axes[idx].set_title(track, fontsize=9, fontweight='bold')
                                axes[idx].grid(True, linestyle='--', alpha=0.5)
                                
                            # مسار السحنات الملون على شكل عمود شريطي (Lithology/Facies Log Track)
                            facies_ax_idx = len(chosen_tracks)
                            ax_f = axes[facies_ax_idx]
                            
                            # تحويل البيانات إلى مصفوفة لرسمها كشريط عمودي متصل عبر الأعماق
                            depths = d.values
                            facies_vals = model_df['Electrofacies'].values
                            
                            # رسم شرائط ملونة دقيقة لكل نقطة عمق
                            for j in range(len(depths) - 1):
                                f_val = facies_vals[j]
                                col_hex = facies_colors.get(f_val, '#1f77b4')
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
                            st.pyplot(fig)
                            
                        with col2:
                            st.subheader("📋 جدول إحصائيات السحنات")
                            stat_col = chosen_tracks[0] if len(chosen_tracks) > 0 else available_logs[0]
                            stats = model_df.groupby(['Electrofacies', 'Facies_Name']).agg(
                                Count=(stat_col, 'count'),
                                Mean_Value=(stat_col, 'mean')
                            ).reset_index()
                            
                            total_points = len(model_df)
                            stats['Thickness_Percentage(%)'] = ((stats['Count'] / total_points) * 100).round(2)
                            stats['Mean_Value'] = stats['Mean_Value'].round(3)
                            
                            st.dataframe(stats, use_container_width=True)
                
                with tab2:
                    st.subheader(f"📉 رسم تقاطعي تفاعلي: ({x_var}) مقابل ({y_var})")
                    fig_cp, ax_cp = plt.subplots(figsize=(8, 6))
                    
                    # تلوين نقاط الـ Crossplot حسب ألوان السحنات المختارة
                    for f_val in range(1, n_clusters + 1):
                        subset_f = model_df[model_df['Electrofacies'] == f_val]
                        if not subset_f.empty:
                            col_hex = facies_colors.get(f_val, '#1f77b4')
                            f_label = facies_names.get(f_val, f"Facies {f_val}")
                            ax_cp.scatter(subset_f[x_var], subset_f[y_var], color=col_hex, label=f_label, alpha=0.7, s=25)
                            
                    ax_cp.set_xlabel(x_var, fontweight='bold')
                    ax_cp.set_ylabel(y_var, fontweight='bold')
                    ax_cp.set_title(f'Crossplot: {x_var} vs {y_var} - {formation_name}', fontweight='bold')
                    ax_cp.grid(True, linestyle='--', alpha=0.5)
                    ax_cp.legend(loc='best', frameon=True)
                    st.pyplot(fig_cp)

                with tab3:
                    st.subheader("📥 تصدير النتائج الكاملة")
                    csv_data = model_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="تحميل ملف النتائج والمسارات الكامل (CSV)",
                        data=csv_data,
                        file_name=f"{formation_name}_Electrofacies_Results.csv",
                        mime="text/csv",
                    )
                    st.markdown("---")
                    st.dataframe(model_df.head(50), use_container_width=True)
                
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف أو المعالجة: {e}")
