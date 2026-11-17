import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile

st.set_page_config(page_title="Electrofacies & Petrophysical Analyzer", layout="wide")

st.title("🛢️ Electrofacies & Petrophysical Analyzer")
st.write("أداة بحثية متقدمة لتصنيف السحنات الكهربائية، تسمية السحنات، الرسوم التقاطعية، والإحصائيات لتكوين الزبير والآبار.")

st.sidebar.header("إعدادات التكوين والعمق")
formation_name = st.sidebar.text_input("اسم التكوين الجيولوجي:", value="Zubair Formation")
min_depth = st.sidebar.number_input("العمق الأدنى (متر):", value=3126.32)
max_depth = st.sidebar.number_input("العمق الأقصى (متر):", value=3269.52)
n_clusters = st.sidebar.slider("عدد السحنات الكهربائية المطلوبة (K):", min_value=2, max_value=8, value=5)

uploaded_file = st.file_uploader("اختر ملف الـ LAS الخاص ببيانات البئر", type=["las"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        
    try:
        las = lasio.read(tmp_path)
        df = las.df().reset_index()
        
        depth_col = 'MD' if 'MD' in df.columns else 'DEPT'
        df[depth_col] = pd.to_numeric(df[depth_col], errors='coerce')
        df['GR'] = pd.to_numeric(df['GR'], errors='coerce')
        df['RHOB'] = pd.to_numeric(df['RHOB'], errors='coerce')
        
        sub_df = df[(df[depth_col] >= min_depth) & (df[depth_col] <= max_depth)].copy()
        
        if len(sub_df) == 0:
            st.warning("⚠️ لا توجد بيانات ضمن نطاق العمق المحدد لهذا البئر!")
        else:
            # حساب الخصائص البتروفيزيائية القياسية
            gr_clean = sub_df['GR'].quantile(0.05)
            gr_shale = sub_df['GR'].quantile(0.95)
            igr = (sub_df['GR'] - gr_clean) / (gr_shale - gr_clean)
            sub_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            sub_df['PHID'] = ((2.65 - sub_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
            
            features = ['GR', 'VSH', 'PHID']
            model_df = sub_df.dropna(subset=features).copy()
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(model_df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model_df['Electrofacies'] = kmeans.fit_predict(X_scaled) + 1
            
            st.success(f"✅ تمت معالجة البيانات وتصنيف السحنات الكهربائية لـ ({formation_name}) بنجاح!")
            
            # --- ميزة تسمية السحنات يدوياً من القائمة الجانبية ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🏷️ تخصيص وتسمية السحنات")
            facies_names = {}
            for i in range(1, n_clusters + 1):
                # تعيين اسم افتراضي (تسمية السحنة الخامسة بـ Tight Sandstone تلقائياً بناءً على طبيعة تكوين الزبير)
                default_name = "Tight Sandstone" if (i == 5 and n_clusters >= 5) else f"Facies {i}"
                facies_names[i] = st.sidebar.text_input(f"اسم السحنة رقم {i}:", value=default_name)
            
            model_df['Facies_Name'] = model_df['Electrofacies'].map(facies_names)
            
            # تقسيم العرض إلى تبويبات (Tabs) منظمة واحترافية
            tab1, tab2, tab3 = st.tabs(["📊 لوحة المسارات والجدول", "📉 الرسوم التقاطعية (Crossplots)", "📥 بيانات التصدير"])
            
            with tab1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"لوحة المسارات - {formation_name}")
                    fig, axes = plt.subplots(1, 4, figsize=(7, 8), sharey=True)
                    d = model_df[depth_col]
                    
                    axes[0].plot(model_df['GR'], d, color='green', lw=1.2)
                    axes[0].set_title('GR (API)', fontsize=9, fontweight='bold')
                    axes[0].set_xlim(0, 150)
                    axes[0].grid(True, linestyle='--', alpha=0.5)
                    
                    axes[1].plot(model_df['VSH'], d, color='brown', lw=1.2)
                    axes[1].set_title('VSH', fontsize=9, fontweight='bold')
                    axes[1].set_xlim(0, 1)
                    axes[1].grid(True, linestyle='--', alpha=0.5)
                    
                    axes[2].plot(model_df['PHID'], d, color='blue', lw=1.2)
                    axes[2].set_title('PHID', fontsize=9, fontweight='bold')
                    axes[2].set_xlim(0.45, 0)
                    axes[2].grid(True, linestyle='--', alpha=0.5)
                    
                    axes[3].scatter(model_df['Electrofacies'], d, c=model_df['Electrofacies'], cmap='tab10', s=15)
                    axes[3].set_title('Electrofacies', fontsize=9, fontweight='bold')
                    axes[3].set_xlim(0.5, n_clusters + 0.5)
                    axes[3].set_xticks(list(range(1, n_clusters + 1)))
                    axes[3].grid(True, linestyle='--', alpha=0.5)
                    
                    for ax in axes:
                        ax.set_ylim(d.max(), d.min())
                        
                    axes[0].set_ylabel('Depth (m)', fontsize=9, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                with col2:
                    st.subheader("📋 جدول إحصائيات السحنات الكهربائية")
                    stats = model_df.groupby(['Electrofacies', 'Facies_Name']).agg(
                        Count=('GR', 'count'),
                        Avg_GR=('GR', 'mean'),
                        Avg_VSH=('VSH', 'mean'),
                        Avg_PHID=('PHID', 'mean')
                    ).reset_index()
                    
                    total_points = len(model_df)
                    stats['Thickness_Percentage(%)'] = (stats['Count'] / total_points) * 100
                    stats['Avg_GR'] = stats['Avg_GR'].round(2)
                    stats['Avg_VSH'] = stats['Avg_VSH'].round(3)
                    stats['Avg_PHID'] = stats['Avg_PHID'].round(3)
                    stats['Thickness_Percentage(%)'] = stats['Thickness_Percentage(%)'].round(2)
                    
                    st.dataframe(stats[['Electrofacies', 'Facies_Name', 'Thickness_Percentage(%)', 'Avg_GR', 'Avg_VSH', 'Avg_PHID']], use_container_width=True)
            
            with tab2:
                st.subheader("📉 الرسوم التقاطعية البتروفيزيائية (Crossplot: GR vs PHID)")
                fig_cp, ax_cp = plt.subplots(figsize=(8, 6))
                scatter = ax_cp.scatter(model_df['PHID'], model_df['GR'], c=model_df['Electrofacies'], cmap='tab10', alpha=0.7, s=25)
                ax_cp.set_xlabel('PHID (Porosity)', fontweight='bold')
                ax_cp.set_ylabel('GR (API)', fontweight='bold')
                ax_cp.set_title(f'Crossplot - {formation_name}', fontweight='bold')
                ax_cp.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_cp)
                st.info("💡 هذا الرسم التقاطعي يساعدك في مطابقة وتحليل المجموعات الخزانية بصرياً تماماً كما تفعل في برنامج Techlog.")

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
