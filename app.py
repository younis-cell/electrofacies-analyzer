import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile

# إعدادات الصفحة بالاسم الجديد
st.set_page_config(page_title="Electrofacies Analyzer", layout="centered")

st.title("🛢️ Electrofacies Analyzer")
st.write("أداة بحثية متقدمة لتحليل بيانات الآبار وتصنيف السحنات الكهربائية لأي تكوين جيولوجي.")

# القائمة الجانبية لإدخال اسم التكوين ونطاق الأعماق بحرية
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
        
        # تصفية البيانات حسب النطاق العمقي
        sub_df = df[(df[depth_col] >= min_depth) & (df[depth_col] <= max_depth)].copy()
        
        if len(sub_df) == 0:
            st.warning("⚠️ لا توجد بيانات ضمن نطاق العمق المحدد لهذا البئر!")
        else:
            # حسابات الخصائص البتروفيزيائية الأساسية
            gr_clean = sub_df['GR'].quantile(0.05)
            gr_shale = sub_df['GR'].quantile(0.95)
            igr = (sub_df['GR'] - gr_clean) / (gr_shale - gr_clean)
            sub_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
            sub_df['PHID'] = ((2.65 - sub_df['RHOB']) / (2.65 - 1.0)).clip(0, 0.45)
            
            # تجهيز البيانات لتصنيف السحنات بالذكاء الاصطناعي
            features = ['GR', 'VSH', 'PHID']
            model_df = sub_df.dropna(subset=features).copy()
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(model_df[features])
            
            # تطبيق خوارزمية KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model_df['Electrofacies'] = kmeans.fit_predict(X_scaled)
            
            st.success(f"✅ تمت معالجة البيانات وتصنيف السحنات الكهربائية لـ ({formation_name}) بنجاح!")
            
            # رسم اللوحة البتروفيزيائية
            st.subheader(f"📊 لوحة المسارات والسحنات الكهربائية - {formation_name}")
            fig, axes = plt.subplots(1, 4, figsize=(9, 8), sharey=True)
            d = model_df[depth_col]
            
            axes[0].plot(model_df['GR'], d, color='green', lw=1.2)
            axes[0].set_title('GR (API)', fontsize=10, fontweight='bold')
            axes[0].set_xlim(0, 150)
            axes[0].grid(True, linestyle='--', alpha=0.5)
            
            axes[1].plot(model_df['VSH'], d, color='brown', lw=1.2)
            axes[1].set_title('VSH', fontsize=10, fontweight='bold')
            axes[1].set_xlim(0, 1)
            axes[1].grid(True, linestyle='--', alpha=0.5)
            
            axes[2].plot(model_df['PHID'], d, color='blue', lw=1.2)
            axes[2].set_title('PHID', fontsize=10, fontweight='bold')
            axes[2].set_xlim(0.45, 0)
            axes[2].grid(True, linestyle='--', alpha=0.5)
            
            axes[3].scatter(model_df['Electrofacies'], d, c=model_df['Electrofacies'], cmap='tab10', s=15)
            axes[3].set_title('Electrofacies', fontsize=10, fontweight='bold')
            axes[3].set_xlim(-0.5, n_clusters - 0.5)
            axes[3].set_xticks(list(range(n_clusters)))
            axes[3].grid(True, linestyle='--', alpha=0.5)
            
            for ax in axes:
                ax.set_ylim(d.max(), d.min())
                
            axes[0].set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
            plt.tight_layout()
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف أو المعالجة: {e}")