import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import tempfile
import pickle
import io

st.set_page_config(page_title="AI Petrophysical Suite", layout="wide")
st.title("🛢️ AI Petrophysical & Electrofacies Suite")

# تهيئة ذاكرة الجلسة
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'track_props' not in st.session_state:
    st.session_state.track_props = {}
if 'ai_summary' not in st.session_state:
    st.session_state.ai_summary = None

# ==========================================
# 1. إدارة وحفظ المشروع (.zubair)
# ==========================================
st.sidebar.header("💾 إدارة المشروع")

saved_project = st.sidebar.file_uploader("استئناف مشروع سابق (.zubair):", type=["zubair"])
if saved_project is not None:
    if st.sidebar.button("🔄 تحميل المشروع"):
        project_data = pickle.load(saved_project)
        st.session_state.master_df = project_data.get('master_df')
        st.session_state.track_props = project_data.get('track_props', {})
        st.session_state.ai_summary = project_data.get('ai_summary')
        st.sidebar.success("✅ تم استرجاع المشروع.")

if st.session_state.master_df is not None:
    project_snapshot = {
        'master_df': st.session_state.master_df,
        'track_props': st.session_state.track_props,
        'ai_summary': st.session_state.ai_summary
    }
    st.sidebar.download_button(
        label="💾 حفظ المشروع (.zubair)",
        data=pickle.dumps(project_snapshot),
        file_name="Petrophysics_AI_Project.zubair",
        mime="application/octet-stream"
    )

st.sidebar.markdown("---")
st.sidebar.header("📁 رفع ملفات LAS")
uploaded_files = st.sidebar.file_uploader("ارفع ملفات الـ LAS:", type=["las"], accept_multiple_files=True)
formation_name = st.sidebar.text_input("اسم التكوين:", value="Zubair Formation")

if uploaded_files:
    if st.sidebar.button("📥 تحميل وقراءة الآبار"):
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
            new_df = pd.concat(all_wells_data, ignore_index=True)
            for col in ['VSH', 'PHID', 'SW', 'WASHOUT_FLAG']:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            st.session_state.master_df = new_df
            st.sidebar.success("✅ تم تحميل الآبار.")

# ==========================================
# 2. لوحة التحكم والذكاء الاصطناعي (المرحلة الأولى)
# ==========================================
if st.session_state.master_df is not None:
    master_df = st.session_state.master_df
    exclude_cols = ['MD', 'WELL_NAME', 'WASHOUT_FLAG']
    current_logs = [c for c in master_df.columns if c not in exclude_cols and not c.endswith('_AI') and not c.endswith('_UNC')]

    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ التحكم ونطاق العمق")

    wells_list = master_df['WELL_NAME'].unique().tolist()
    selected_well = st.sidebar.selectbox("اختر البئر:", options=wells_list)

    well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
    well_min_d = float(well_raw_df['MD'].min()) if not well_raw_df['MD'].isna().all() else 0.0
    well_max_d = float(well_raw_df['MD'].max()) if not well_raw_df['MD'].isna().all() else 1000.0

    min_depth = st.sidebar.number_input("العمق الأدنى (م):", value=well_min_d)
    max_depth = st.sidebar.number_input("العمق الأقصى (م):", value=well_max_d)

    # ---- الحسابات البتروفيزيائية الأساسية ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("1️⃣ الحسابات الأساسية")
    matrix_rho = st.sidebar.number_input("كثافة المصفوفة (g/cc):", value=2.65, step=0.01)
    fluid_rho = st.sidebar.number_input("كثافة المائع (g/cc):", value=1.00, step=0.05)
    rw_val = st.sidebar.number_input("مقاومة ماء التكوين (Rw):", value=0.03, step=0.01)

    if st.sidebar.button("حساب VSH, PHID, SW"):
        if 'GR' in master_df.columns:
            gr_clean = master_df['GR'].quantile(0.05)
            gr_shale = master_df['GR'].quantile(0.95)
            if gr_shale != gr_clean:
                igr = (master_df['GR'] - gr_clean) / (gr_shale - gr_clean)
                master_df['VSH'] = (0.083 * (np.power(2, 2 * igr) - 1)).clip(0, 1)
        if 'RHOB' in master_df.columns and (matrix_rho - fluid_rho) != 0:
            master_df['PHID'] = ((matrix_rho - master_df['RHOB']) / (matrix_rho - fluid_rho)).clip(0, 0.45)

        res_col = next((r for r in ['ILD', 'LLD', 'RT', 'RILD', 'RESD'] if r in master_df.columns), None)
        if res_col and 'PHID' in master_df.columns:
            phid_safe = master_df['PHID'].replace(0, 0.001)
            rt_safe = master_df[res_col].replace(0, 1)
            sw = np.power((1.0 * rw_val) / (np.power(phid_safe, 2.0) * rt_safe), 0.5)
            master_df['SW'] = sw.clip(0, 1)

        st.session_state.master_df = master_df
        st.sidebar.success("✅ تم الحساب.")

    # ---- وحدة الذكاء الاصطناعي: التنبؤ بالمجسات المفقودة وتصحيح التكهف ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("2️⃣ AI: تعويض المجسات وتصحيح التكهف")

    all_available_now = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG']]
    target_log = st.sidebar.selectbox("المجس المستهدف (المفقود أو المشوّه):", options=all_available_now)

    default_preds = [c for c in ['GR', 'NPHI', 'DT', 'ILD', 'LLD', 'VSH'] if c in all_available_now and c != target_log]
    predictor_logs = st.sidebar.multiselect("المجسات المرجعية للإدخال:", options=[c for c in all_available_now if c != target_log], default=default_preds)

    ai_algo = st.sidebar.selectbox("نموذج الذكاء الاصطناعي:", options=["Random Forest Regressor", "Gradient Boosting Regressor"])

    cali_col = st.sidebar.selectbox("مجس الفرجار لكشف التكهف (اختياري):", options=["None"] + all_available_now)
    bit_size = st.sidebar.number_input("قطر المثقاب الاسمي (Bit Size):", value=8.5, step=0.5)
    washout_delta = st.sidebar.number_input("عتبة زيادة التكهف (ΔCali):", value=1.5, step=0.25)

    replace_mode = st.sidebar.selectbox("آلية التعويض في العمود المدمج:", options=[
        "تعويض القيم المفقودة والمتكهفة فقط (Smart Fill)",
        "استبدال كامل المجس بالقيمة المتنبأة (Full Synthetic)"
    ])

    if st.sidebar.button("🤖 تشغيل نموذج التنبؤ وعدم اليقين"):
        if len(predictor_logs) < 1:
            st.error("⚠️ اختر مجساً مرجعياً واحداً على الأقل.")
        else:
            # تحديد نقاط التكهف إن توفر مجس الفرجار
            if cali_col != "None":
                washout_mask = (master_df[cali_col] - bit_size) > washout_delta
            else:
                washout_mask = pd.Series(False, index=master_df.index)
            master_df['WASHOUT_FLAG'] = washout_mask.astype(int)

            # بيانات التدريب: نقاط لا تحتوي على تكهف ولها قيمة حقيقية للمجس المستهدف والمدخلات
            train_mask = (~washout_mask) & master_df[target_log].notna() & master_df[predictor_logs].notna().all(axis=1)
            pred_mask = master_df[predictor_logs].notna().all(axis=1)

            X_train = master_df.loc[train_mask, predictor_logs]
            y_train = master_df.loc[train_mask, target_log]
            X_pred = master_df.loc[pred_mask, predictor_logs]

            if len(X_train) < 20 or len(X_pred) == 0:
                st.error("⚠️ بيانات التدريب أو المدخلات غير كافية لتشغيل النموذج.")
            else:
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_pred_sc = scaler.transform(X_pred)

                if ai_algo == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                    model.fit(X_train_sc, y_train)
                    y_hat = model.predict(X_pred_sc)
                    # حساب عدم اليقين من تشتت تنبؤات الأشجار
                    tree_preds = np.array([est.predict(X_pred_sc) for est in model.estimators_])
                    unc_vals = np.std(tree_preds, axis=0)
                else:
                    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
                    model.fit(X_train_sc, y_train)
                    y_hat = model.predict(X_pred_sc)
                    # تقدير عدم اليقين عبر الانحراف المعياري للبواقي على بيانات التدريب
                    train_res_std = float(np.std(y_train - model.predict(X_train_sc)))
                    unc_vals = np.full(len(y_hat), train_res_std)

                ai_col_name = f"{target_log}_AI"
                unc_col_name = f"{target_log}_UNC"
                merged_col_name = f"{target_log}_MERGED"

                master_df[ai_col_name] = np.nan
                master_df[unc_col_name] = np.nan
                master_df.loc[pred_mask, ai_col_name] = y_hat
                master_df.loc[pred_mask, unc_col_name] = unc_vals

                if replace_mode == "Full Synthetic":
                    master_df[merged_col_name] = master_df[ai_col_name]
                else:
                    needs_fix = master_df[target_log].isna() | washout_mask
                    master_df[merged_col_name] = np.where(needs_fix & master_df[ai_col_name].notna(), master_df[ai_col_name], master_df[target_log])

                r2_train = model.score(X_train_sc, y_train)
                st.session_state.master_df = master_df
                st.session_state.ai_summary = {
                    'formation': formation_name,
                    'target': target_log,
                    'algo': ai_algo,
                    'r2_train': round(float(r2_train), 4),
                    'washout_count': int(washout_mask.sum()),
                    'filled_count': int((master_df[target_log].isna() | washout_mask).sum()),
                    'mean_unc': round(float(np.nanmean(unc_vals)), 4)
                }
                st.sidebar.success("✅ اكتمل التنبؤ وحساب عدم اليقين.")

    # ==========================================
    # 3. التبويبات الرئيسية
    # ==========================================
    all_plot_logs = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG']]
    single_well_df = well_raw_df[(well_raw_df['MD'] >= min_depth) & (well_raw_df['MD'] <= max_depth)].sort_values(by='MD')

    tab_ai, tab_tracks, tab_props, tab_export = st.tabs([
        "🤖 التنبؤ وعدم اليقين (AI)",
        "📊 لوحة المسارات",
        "🎨 Properties",
        "📥 تصدير النتائج"
    ])

    with tab_props:
        st.subheader("🎨 Properties - خصائص المنحنيات")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_height = st.slider("ارتفاع اللوحة:", min_value=5, max_value=20, value=9, key="p_h")
            track_width = st.slider("عرض كل مسار:", min_value=1.5, max_value=5.0, value=2.5, step=0.1, key="p_w")
        with p_col2:
            title_fontsize = st.slider("حجم خط العناوين:", min_value=8, max_value=16, value=10, key="p_fs")
            show_grid = st.checkbox("إظهار الشبكة:", value=True, key="p_gr")

        st.markdown("---")
        sel_curve = st.selectbox("اختر المنحنى لتعديل خصائصه:", options=all_plot_logs, key="p_curve")
        curr_p = st.session_state.track_props.get(sel_curve, {'color': '#1f77b4', 'lw': 1.3, 'ls': '-'})

        c1, c2, c3 = st.columns(3)
        with c1:
            new_col = st.color_picker("اللون:", value=curr_p['color'], key=f"col_{sel_curve}")
        with c2:
            new_lw = st.slider("السمك:", min_value=0.5, max_value=5.0, value=float(curr_p['lw']), step=0.1, key=f"lw_{sel_curve}")
        with c3:
            ls_map = {'متصل (-)': '-', 'متقطع (--)': '--', 'منقط (:)': ':', 'خط ونقطة (-.)': '-.'}
            ls_lbl = st.selectbox("النمط:", options=list(ls_map.keys()), index=list(ls_map.values()).index(curr_p['ls']), key=f"ls_{sel_curve}")

        st.session_state.track_props[sel_curve] = {'color': new_col, 'lw': new_lw, 'ls': ls_map[ls_lbl]}

    with tab_ai:
        st.subheader(f"🤖 نتائج التنبؤ بالمجسات وتقدير عدم اليقين — {selected_well} ({formation_name})")
        if st.session_state.ai_summary is not None:
            s = st.session_state.ai_summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("معامل التحديد للتدريب (R²)", s['r2_train'])
            m2.metric("نقاط التكهف المرصودة", s['washout_count'])
            m3.metric("نقاط التعويض / التصحيح", s['filled_count'])
            m4.metric("متوسط عدم اليقين (±1σ)", s['mean_unc'])

            t_name = s['target']
            ai_c = f"{t_name}_AI"
            mg_c = f"{t_name}_MERGED"
            uc_c = f"{t_name}_UNC"

            if ai_c in single_well_df.columns and len(single_well_df) > 0:
                fig_ai, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, fig_height), sharey=True)
                d_vals = single_well_df['MD'].values

                # المسار الأول: مقارنة المجس الأصلي والمُعوّض مع مجال عدم اليقين
                ax1.plot(single_well_df[t_name], d_vals, label=f"{t_name} (Original)", color="#1f77b4", lw=1.2)
                if mg_c in single_well_df.columns:
                    ax1.plot(single_well_df[mg_c], d_vals, label=f"{mg_c} (AI Corrected)", color="#d62728", lw=1.2, ls="--")

                if uc_c in single_well_df.columns and mg_c in single_well_df.columns:
                    y_m = single_well_df[mg_c].values
                    u_m = single_well_df[uc_c].values
                    ax1.fill_betweenx(d_vals, y_m - u_m, y_m + u_m, color="#d62728", alpha=0.18, label="Uncertainty (±1σ)")

                ax1.set_title(f"{t_name} Reconstruction", fontsize=title_fontsize, fontweight="bold")
                ax1.set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
                ax1.set_ylim(d_vals.max(), d_vals.min())
                ax1.legend(loc="best", fontsize=8)
                if show_grid:
                    ax1.grid(True, linestyle="--", alpha=0.5)

                # المسار الثاني: منحنى عدم اليقين مع العمق
                if uc_c in single_well_df.columns:
                    ax2.plot(single_well_df[uc_c], d_vals, color="#ff7f0e", lw=1.3)
                    ax2.fill_betweenx(d_vals, 0, single_well_df[uc_c].values, color="#ff7f0e", alpha=0.25)
                ax2.set_title("Uncertainty (±1σ)", fontsize=title_fontsize, fontweight="bold")
                if show_grid:
                    ax2.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                buf_ai = io.BytesIO()
                fig_ai.savefig(buf_ai, format="png", dpi=300, bbox_inches="tight")
                buf_ai.seek(0)
                st.pyplot(fig_ai)
                st.download_button("📥 تحميل رسم التنبؤ وعدم اليقين (300 DPI)", data=buf_ai, file_name=f"{selected_well}_{t_name}_AI_Uncertainty.png", mime="image/png")
        else:
            st.info("اضبط إعدادات التنبؤ في القائمة الجانبية ثم اضغط (تشغيل نموذج التنبؤ وعدم اليقين).")

    with tab_tracks:
        st.subheader(f"📊 لوحة المسارات — {selected_well}")
        default_tr = [t for t in ['GR', 'VSH', 'PHID', 'SW'] if t in all_plot_logs]
        chosen_tracks = st.multiselect("اختر المسارات للعرض:", options=all_plot_logs, default=default_tr)

        if len(chosen_tracks) > 0 and len(single_well_df) > 0:
            fig_t, axes = plt.subplots(1, len(chosen_tracks), figsize=(track_width * len(chosen_tracks), fig_height), sharey=True)
            if len(chosen_tracks) == 1:
                axes = [axes]
            d_t = single_well_df['MD']
            for i, tr in enumerate(chosen_tracks):
                tp = st.session_state.track_props.get(tr, {'color': '#1f77b4', 'lw': 1.3, 'ls': '-'})
                axes[i].plot(single_well_df[tr], d_t, color=tp['color'], lw=tp['lw'], ls=tp['ls'])
                axes[i].set_title(tr, fontsize=title_fontsize, fontweight="bold")
                axes[i].set_ylim(d_t.max(), d_t.min())
                if show_grid:
                    axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[0].set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
            plt.tight_layout()

            buf_t = io.BytesIO()
            fig_t.savefig(buf_t, format="png", dpi=300, bbox_inches="tight")
            buf_t.seek(0)
            st.pyplot(fig_t)
            st.download_button("📥 تحميل لوحة المسارات (300 DPI)", data=buf_t, file_name=f"{selected_well}_Tracks.png", mime="image/png")

    with tab_export:
        st.subheader("📥 تصدير البيانات")
        csv_bytes = master_df.to_csv(index=False).encode("utf-8")
        st.download_button("تحميل جدول النتائج الكامل (CSV)", data=csv_bytes, file_name=f"{formation_name}_AI_Stage1.csv", mime="text/csv")
        st.dataframe(master_df.head(100), use_container_width=True)
