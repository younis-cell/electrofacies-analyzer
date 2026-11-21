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

# ==========================================
# تهيئة ذاكرة الجلسة (Session State)
# ==========================================
defaults = {
    'master_df': None,
    'track_props': {},
    'ai_summary': None,
    'perm_summary': None,
    'cutoff_params': {
        'vsh_max': 0.40,
        'phi_min': 0.10,
        'sw_max': 0.60,
        'k_min': 1.0,
        'tight_k_max': 1.0,
        'min_unit_thickness': 2.0
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def normalize_depth_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    for candidate in ['MD', 'DEPT', 'DEPTH', 'TVD']:
        if candidate in df.columns:
            df = df.rename(columns={candidate: 'MD'})
            break
    if 'MD' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'MD'})
    df['MD'] = pd.to_numeric(df['MD'], errors='coerce')
    return df.dropna(subset=['MD'])


def assign_reservoir_units(df_well: pd.DataFrame, min_thick: float = 2.0) -> pd.Series:
    """تقسيم الفواصل المكمنية (PAY_FLAG == 1) إلى وحدات مكمنية مرقمة."""
    units = pd.Series("Non-Pay", index=df_well.index, dtype=object)
    if len(df_well) < 2:
        return units
    d_step = float(np.median(np.diff(np.sort(df_well['MD'].values))))
    if d_step <= 0 or np.isnan(d_step):
        d_step = 0.1524

    pay = df_well['PAY_FLAG'].fillna(0).astype(int).values
    unit_id = 0
    in_unit = False
    start_idx = 0
    labels = ["Non-Pay"] * len(df_well)

    for i, p in enumerate(pay):
        if p == 1 and not in_unit:
            in_unit = True
            start_idx = i
        elif p == 0 and in_unit:
            thick = (i - start_idx) * d_step
            if thick >= min_thick:
                unit_id += 1
                for j in range(start_idx, i):
                    labels[j] = f"Unit-{unit_id}"
            in_unit = False

    if in_unit:
        thick = (len(pay) - start_idx) * d_step
        if thick >= min_thick:
            unit_id += 1
            for j in range(start_idx, len(pay)):
                labels[j] = f"Unit-{unit_id}"

    return pd.Series(labels, index=df_well.index)


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
        st.session_state.perm_summary = project_data.get('perm_summary')
        if project_data.get('cutoff_params'):
            st.session_state.cutoff_params = project_data.get('cutoff_params')
        st.sidebar.success("✅ تم استرجاع المشروع بالكامل.")

if st.session_state.master_df is not None:
    project_snapshot = {
        'master_df': st.session_state.master_df,
        'track_props': st.session_state.track_props,
        'ai_summary': st.session_state.ai_summary,
        'perm_summary': st.session_state.perm_summary,
        'cutoff_params': st.session_state.cutoff_params
    }
    st.sidebar.download_button(
        label="💾 حفظ المشروع (.zubair)",
        data=pickle.dumps(project_snapshot),
        file_name="Petrophysics_AI_Suite.zubair",
        mime="application/octet-stream"
    )

st.sidebar.markdown("---")
st.sidebar.header("📁 رفع ملفات LAS والنوى")
uploaded_files = st.sidebar.file_uploader("ارفع ملفات الـ LAS:", type=["las"], accept_multiple_files=True)
core_file = st.sidebar.file_uploader("ملف النوى (اختياري CSV / XLSX):", type=["csv", "xlsx"])
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
                df = normalize_depth_col(df)
                df['WELL_NAME'] = well_name
                all_wells_data.append(df)
            except Exception as e:
                st.error(f"خطأ في قراءة البئر {well_name}: {e}")

        if all_wells_data:
            new_df = pd.concat(all_wells_data, ignore_index=True)
            for col in ['VSH', 'PHID', 'SW', 'WASHOUT_FLAG', 'CORE_K', 'PERM_AI', 'PERM_UNC', 'TIGHT_FLAG', 'PAY_FLAG']:
                if col not in new_df.columns:
                    new_df[col] = np.nan

            # دمج ملف النوى الخارجي إن وجد
            if core_file is not None:
                try:
                    cdf = pd.read_csv(core_file) if core_file.name.endswith('.csv') else pd.read_excel(core_file)
                    cdf = normalize_depth_col(cdf)
                    k_cand = next((c for c in ['CORE_K', 'PERM', 'K', 'PERMEABILITY', 'KH', 'KL'] if c in cdf.columns), None)
                    if k_cand:
                        cdf = cdf.rename(columns={k_cand: 'CORE_K'})
                        cdf['CORE_K'] = pd.to_numeric(cdf['CORE_K'], errors='coerce')
                        cdf = cdf[cdf['CORE_K'] > 0]
                        if 'WELL_NAME' in cdf.columns:
                            cdf['WELL_NAME'] = cdf['WELL_NAME'].astype(str).str.strip()
                            for w_name, w_group in cdf.groupby('WELL_NAME'):
                                w_idx = new_df['WELL_NAME'] == w_name
                                if w_idx.any() and len(w_group) > 0:
                                    interp_k = np.interp(
                                        new_df.loc[w_idx, 'MD'].values,
                                        w_group.sort_values('MD')['MD'].values,
                                        w_group.sort_values('MD')['CORE_K'].values,
                                        left=np.nan, right=np.nan
                                    )
                                    new_df.loc[w_idx, 'CORE_K'] = interp_k
                        else:
                            # مطابقة حسب العمق عبر جميع الآبار إذا لم يوجد عمود WELL_NAME
                            for w_name in new_df['WELL_NAME'].unique():
                                w_idx = new_df['WELL_NAME'] == w_name
                                w_md = new_df.loc[w_idx, 'MD']
                                c_sub = cdf[(cdf['MD'] >= w_md.min()) & (cdf['MD'] <= w_md.max())].sort_values('MD')
                                if len(c_sub) > 0:
                                    new_df.loc[w_idx, 'CORE_K'] = np.interp(w_md.values, c_sub['MD'].values, c_sub['CORE_K'].values, left=np.nan, right=np.nan)
                except Exception as e:
                    st.warning(f"تعذر دمج ملف النوى: {e}")

            st.session_state.master_df = new_df
            st.sidebar.success("✅ تم تحميل الآبار بنجاح.")

# ==========================================
# 2. لوحة التحكم والذكاء الاصطناعي (المرحلة 1 + المرحلة 2)
# ==========================================
if st.session_state.master_df is not None:
    master_df = st.session_state.master_df

    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ التحكم ونطاق العمق")

    wells_list = master_df['WELL_NAME'].unique().tolist()
    selected_well = st.sidebar.selectbox("اختر البئر:", options=wells_list)

    well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
    well_min_d = float(well_raw_df['MD'].min()) if not well_raw_df['MD'].isna().all() else 0.0
    well_max_d = float(well_raw_df['MD'].max()) if not well_raw_df['MD'].isna().all() else 1000.0

    min_depth = st.sidebar.number_input("العمق الأدنى (م):", value=well_min_d)
    max_depth = st.sidebar.number_input("العمق الأقصى (م):", value=well_max_d)

    # ---- 1. الحسابات البتروفيزيائية الأساسية ----
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

    # ---- 2. المرحلة الأولى: التنبؤ بالمجسات وتصحيح التكهف ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("2️⃣ المرحلة 1: تعويض المجسات والتكهف")

    all_available_now = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT']]
    target_log = st.sidebar.selectbox("المجس المستهدف (المفقود أو المشوّه):", options=all_available_now)

    default_preds = [c for c in ['GR', 'NPHI', 'DT', 'RHOB', 'VSH'] if c in all_available_now and c != target_log]
    predictor_logs = st.sidebar.multiselect(
        "المجسات المرجعية للإدخال:",
        options=[c for c in all_available_now if c != target_log],
        default=default_preds
    )

    ai_algo_s1 = st.sidebar.selectbox("خوارزمية المرحلة 1:", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="algo_s1")
    cali_col = st.sidebar.selectbox("مجس الفرجار لكشف التكهف (اختياري):", options=["None"] + all_available_now)
    bit_size = st.sidebar.number_input("قطر المثقاب الاسمي (Bit Size):", value=8.5, step=0.5)
    washout_delta = st.sidebar.number_input("عتبة زيادة التكهف (ΔCali):", value=1.5, step=0.25)

    replace_mode = st.sidebar.selectbox("آلية التعويض في العمود المدمج:", options=[
        "تعويض القيم المفقودة والمتكهفة فقط (Smart Fill)",
        "استبدال كامل المجس بالقيمة المتنبأة (Full Synthetic)"
    ])

    if st.sidebar.button("🤖 تشغيل المرحلة 1 (المجسات وعدم اليقين)"):
        if len(predictor_logs) < 1:
            st.error("⚠️ اختر مجساً مرجعياً واحداً على الأقل.")
        else:
            if cali_col != "None":
                washout_mask = (master_df[cali_col] - bit_size) > washout_delta
            else:
                washout_mask = pd.Series(False, index=master_df.index)
            master_df['WASHOUT_FLAG'] = washout_mask.astype(int)

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

                if ai_algo_s1 == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                    model.fit(X_train_sc, y_train)
                    y_hat = model.predict(X_pred_sc)
                    tree_preds = np.array([est.predict(X_pred_sc) for est in model.estimators_])
                    unc_vals = np.std(tree_preds, axis=0)
                else:
                    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
                    model.fit(X_train_sc, y_train)
                    y_hat = model.predict(X_pred_sc)
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
                    'algo': ai_algo_s1,
                    'r2_train': round(float(r2_train), 4),
                    'washout_count': int(washout_mask.sum()),
                    'filled_count': int((master_df[target_log].isna() | washout_mask).sum()),
                    'mean_unc': round(float(np.nanmean(unc_vals)), 4)
                }
                st.sidebar.success("✅ اكتملت المرحلة الأولى.")

    # ---- 3. المرحلة الثانية: تقدير النفاذية والوحدات المكمنية وحدود القطع ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("3️⃣ المرحلة 2: النفاذية وحدود القطع")

    available_for_perm = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT', 'PERM_AI', 'PERM_UNC']]
    k_source_col = st.sidebar.selectbox("عمود النفاذية المرجعية (نوى أو K):", options=available_for_perm, index=available_for_perm.index('CORE_K') if 'CORE_K' in available_for_perm else 0)

    default_perm_feats = [c for c in ['PHID', 'VSH', 'GR', 'NPHI', 'RHOB'] if c in available_for_perm and c != k_source_col]
    perm_features = st.sidebar.multiselect("مجسات التنبؤ بالنفاذية:", options=[c for c in available_for_perm if c != k_source_col], default=default_perm_feats)
    use_rolling_ctx = st.sidebar.checkbox("إضافة سياق العمق المتحرك (Rolling Context)", value=True)
    rolling_win = st.sidebar.slider("نافذة العمق المتحركة (نقاط):", min_value=3, max_value=21, value=7, step=2)
    ai_algo_s2 = st.sidebar.selectbox("خوارزمية تقدير النفاذية:", options=["Random Forest Regressor", "Gradient Boosting Regressor"], key="algo_s2")

    st.sidebar.markdown("**ضبط حدود القطع (Cut-off Limits) والفواصل الضيقة:**")
    cp = st.session_state.cutoff_params
    vsh_max_in = st.sidebar.number_input("الحد الأقصى لـ VSH:", value=float(cp['vsh_max']), min_value=0.0, max_value=1.0, step=0.02)
    phi_min_in = st.sidebar.number_input("الحد الأدنى للمسامية (PHID):", value=float(cp['phi_min']), min_value=0.0, max_value=0.5, step=0.01)
    sw_max_in = st.sidebar.number_input("الحد الأقصى للتشبع المائي (SW):", value=float(cp['sw_max']), min_value=0.0, max_value=1.0, step=0.02)
    k_min_in = st.sidebar.number_input("الحد الأدنى للنفاذية (mD):", value=float(cp['k_min']), min_value=0.001, step=0.5)
    tight_k_in = st.sidebar.number_input("عتبة الفواصل الضيقة Tight K_max (mD):", value=float(cp['tight_k_max']), min_value=0.001, step=0.2)
    min_thick_in = st.sidebar.number_input("أدنى سماكة للوحدة المكمنية (م):", value=float(cp['min_unit_thickness']), min_value=0.5, step=0.5)

    st.session_state.cutoff_params = {
        'vsh_max': vsh_max_in,
        'phi_min': phi_min_in,
        'sw_max': sw_max_in,
        'k_min': k_min_in,
        'tight_k_max': tight_k_in,
        'min_unit_thickness': min_thick_in
    }

    if st.sidebar.button("⚙️ تشغيل المرحلة 2 (النفاذية والوحدات)"):
        if len(perm_features) < 1:
            st.error("⚠️ اختر مجساً واحداً على الأقل للتنبؤ بالنفاذية.")
        else:
            work_df = master_df.copy()
            feat_cols = list(perm_features)

            if use_rolling_ctx:
                for fc in perm_features:
                    rm_col = f"{fc}_RM{rolling_win}"
                    rs_col = f"{fc}_RS{rolling_win}"
                    work_df[rm_col] = work_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).mean())
                    work_df[rs_col] = work_df.groupby('WELL_NAME')[fc].transform(lambda s: s.rolling(rolling_win, center=True, min_periods=1).std().fillna(0))
                    feat_cols.extend([rm_col, rs_col])

            k_vals = pd.to_numeric(work_df[k_source_col], errors='coerce')
            valid_k_pos = k_vals > 0
            log_k = np.where(valid_k_pos, np.log10(k_vals.clip(lower=1e-4)), np.nan)

            tr_mask = (~np.isnan(log_k)) & work_df[feat_cols].notna().all(axis=1)
            pr_mask = work_df[feat_cols].notna().all(axis=1)

            X_tr = work_df.loc[tr_mask, feat_cols]
            y_tr = log_k[tr_mask]
            X_pr = work_df.loc[pr_mask, feat_cols]

            if len(X_tr) < 10 or len(X_pr) == 0:
                st.error("⚠️ بيانات النفاذية المرجعية أو المجسات المدخلة غير كافية لتدريب نموذج النفاذية.")
            else:
                scaler_k = StandardScaler()
                X_tr_sc = scaler_k.fit_transform(X_tr)
                X_pr_sc = scaler_k.transform(X_pr)

                if ai_algo_s2 == "Random Forest Regressor":
                    k_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                    k_model.fit(X_tr_sc, y_tr)
                    log_k_hat = k_model.predict(X_pr_sc)
                    t_preds = np.array([est.predict(X_pr_sc) for est in k_model.estimators_])
                    log_k_std = np.std(t_preds, axis=0)
                else:
                    k_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
                    k_model.fit(X_tr_sc, y_tr)
                    log_k_hat = k_model.predict(X_pr_sc)
                    res_std = float(np.std(y_tr - k_model.predict(X_tr_sc)))
                    log_k_std = np.full(len(log_k_hat), res_std)

                k_pred_md = np.power(10.0, log_k_hat)
                # تقدير عدم اليقين بالـ mD عبر الفارق اللوغاريتمي (+/- 1 sigma)
                k_unc_md = np.power(10.0, log_k_hat + log_k_std) - np.power(10.0, log_k_hat - log_k_std)

                work_df['PERM_AI'] = np.nan
                work_df['PERM_UNC'] = np.nan
                work_df.loc[pr_mask, 'PERM_AI'] = k_pred_md
                work_df.loc[pr_mask, 'PERM_UNC'] = k_unc_md

                # تحديد الفواصل الضيقة (Tight Intervals)
                work_df['TIGHT_FLAG'] = np.where(work_df['PERM_AI'].notna() & (work_df['PERM_AI'] <= tight_k_in), 1, 0)

                # تطبيق حدود القطع (Cut-off Limits) لتحديد Pay
                vsh_ok = work_df['VSH'].fillna(0) <= vsh_max_in if 'VSH' in work_df.columns else pd.Series(True, index=work_df.index)
                phi_ok = work_df['PHID'].fillna(0) >= phi_min_in if 'PHID' in work_df.columns else pd.Series(True, index=work_df.index)
                sw_ok = work_df['SW'].fillna(0) <= sw_max_in if 'SW' in work_df.columns else pd.Series(True, index=work_df.index)
                k_ok = work_df['PERM_AI'].fillna(0) >= k_min_in

                work_df['PAY_FLAG'] = (vsh_ok & phi_ok & sw_ok & k_ok).astype(int)

                # تحديد الوحدات المكمنية لكل بئر
                work_df['RES_UNIT'] = "Non-Pay"
                for w_name in work_df['WELL_NAME'].unique():
                    w_sub = work_df[work_df['WELL_NAME'] == w_name].sort_values('MD')
                    work_df.loc[w_sub.index, 'RES_UNIT'] = assign_reservoir_units(w_sub, min_thick=min_thick_in)

                r2_k = k_model.score(X_tr_sc, y_tr)
                st.session_state.master_df = work_df
                st.session_state.perm_summary = {
                    'formation': formation_name,
                    'k_source': k_source_col,
                    'algo': ai_algo_s2,
                    'r2_logk': round(float(r2_k), 4),
                    'tight_count': int(work_df['TIGHT_FLAG'].sum()),
                    'pay_count': int(work_df['PAY_FLAG'].sum()),
                    'units_found': int(work_df[work_df['RES_UNIT'] != 'Non-Pay']['RES_UNIT'].nunique())
                }
                st.sidebar.success("✅ اكتملت المرحلة الثانية.")

    # ==========================================
    # 3. التبويبات الرئيسية
    # ==========================================
    master_df = st.session_state.master_df
    well_raw_df = master_df[master_df['WELL_NAME'] == selected_well]
    single_well_df = well_raw_df[(well_raw_df['MD'] >= min_depth) & (well_raw_df['MD'] <= max_depth)].sort_values(by='MD')
    all_plot_logs = [c for c in master_df.columns if c not in ['MD', 'WELL_NAME', 'WASHOUT_FLAG', 'TIGHT_FLAG', 'PAY_FLAG', 'RES_UNIT'] and pd.api.types.is_numeric_dtype(master_df[c])]

    tab_s2, tab_ai, tab_tracks, tab_props, tab_export = st.tabs([
        "🎯 النفاذية والوحدات (المرحلة 2)",
        "🤖 التنبؤ وعدم اليقين (المرحلة 1)",
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
        if len(all_plot_logs) > 0:
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

    with tab_s2:
        st.subheader(f"🎯 تقدير النفاذية والفواصل الضيقة والوحدات المكمنية — {selected_well} ({formation_name})")
        if st.session_state.perm_summary is not None and len(single_well_df) > 0:
            ps = st.session_state.perm_summary
            d_arr = single_well_df['MD'].values
            dz = float(np.median(np.diff(d_arr))) if len(d_arr) > 1 else 0.1524
            gross_thick = float(d_arr.max() - d_arr.min()) if len(d_arr) > 1 else 0.0
            pay_thick = float(single_well_df['PAY_FLAG'].sum() * dz)
            tight_thick = float(single_well_df['TIGHT_FLAG'].sum() * dz)
            ntg_pct = (pay_thick / gross_thick * 100.0) if gross_thick > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² لنموذج log10(K)", ps['r2_logk'])
            m2.metric("سمك Net Pay (م)", f"{pay_thick:.2f}")
            m3.metric("نسبة Net-to-Gross (%)", f"{ntg_pct:.1f}%")
            m4.metric("سماكة الفواصل الضيقة Tight (م)", f"{tight_thick:.2f}")

            fig_s2, (ax_k, ax_pay) = plt.subplots(1, 2, figsize=(9.5, fig_height), sharey=True)

            if 'PERM_AI' in single_well_df.columns and single_well_df['PERM_AI'].notna().any():
                k_plot = single_well_df['PERM_AI'].clip(lower=1e-3)
                ax_k.semilogx(k_plot, d_arr, color="#2ca02c", lw=1.4, label="PERM_AI (mD)")
                if 'CORE_K' in single_well_df.columns and single_well_df['CORE_K'].notna().any():
                    ck_sub = single_well_df[single_well_df['CORE_K'] > 0]
                    ax_k.scatter(ck_sub['CORE_K'], ck_sub['MD'], color="#000000", s=14, zorder=5, label="Core K (mD)")
                ax_k.axvline(st.session_state.cutoff_params['tight_k_max'], color="#d62728", ls="--", lw=1.1, label="Tight K Cutoff")
                ax_k.axvline(st.session_state.cutoff_params['k_min'], color="#1f77b4", ls=":", lw=1.2, label="Pay K Cutoff")
                ax_k.set_title("Permeability (mD)", fontsize=title_fontsize, fontweight="bold")
                ax_k.set_ylabel("Depth (m)", fontsize=title_fontsize, fontweight="bold")
                ax_k.set_ylim(d_arr.max(), d_arr.min())
                ax_k.legend(loc="best", fontsize=7.5)
                if show_grid:
                    ax_k.grid(True, which="both", linestyle="--", alpha=0.4)

            pay_arr = single_well_df['PAY_FLAG'].fillna(0).values
            tight_arr = single_well_df['TIGHT_FLAG'].fillna(0).values
            ax_pay.fill_betweenx(d_arr, 0, pay_arr, color="#2ca02c", alpha=0.65, step="mid", label="Pay Zone")
            ax_pay.fill_betweenx(d_arr, 0, np.where((pay_arr == 0) & (tight_arr == 1), 0.5, 0), color="#8c564b", alpha=0.55, step="mid", label="Tight / Low-K")
            ax_pay.set_xlim(0, 1.1)
            ax_pay.set_xticks([0, 0.5, 1])
            ax_pay.set_xticklabels(["Non-Pay", "Tight", "Pay"], fontsize=8)
            ax_pay.set_title("Reservoir Pay & Tight Zones", fontsize=title_fontsize, fontweight="bold")
            ax_pay.legend(loc="best", fontsize=7.5)
            if show_grid:
                ax_pay.grid(True, linestyle="--", alpha=0.4)

            plt.tight_layout()
            buf_s2 = io.BytesIO()
            fig_s2.savefig(buf_s2, format="png", dpi=300, bbox_inches="tight")
            buf_s2.seek(0)
            st.pyplot(fig_s2)
            st.download_button("📥 تحميل مخطط النفاذية والوحدات (300 DPI)", data=buf_s2, file_name=f"{selected_well}_Perm_Pay_Units.png", mime="image/png")

            # ملخص الوحدات المكمنية في النطاق المحدد
            if 'RES_UNIT' in single_well_df.columns:
                unit_rows = []
                for u_name, u_grp in single_well_df[single_well_df['RES_UNIT'] != 'Non-Pay'].groupby('RES_UNIT'):
                    u_thick = float(len(u_grp) * dz)
                    avg_phi = float(u_grp['PHID'].mean()) if 'PHID' in u_grp.columns and u_grp['PHID'].notna().any() else np.nan
                    avg_vsh = float(u_grp['VSH'].mean()) if 'VSH' in u_grp.columns and u_grp['VSH'].notna().any() else np.nan
                    avg_sw = float(u_grp['SW'].mean()) if 'SW' in u_grp.columns and u_grp['SW'].notna().any() else np.nan
                    avg_k = float(u_grp['PERM_AI'].mean()) if 'PERM_AI' in u_grp.columns and u_grp['PERM_AI'].notna().any() else np.nan
                    unit_rows.append({
                        "Unit": u_name,
                        "Top (m)": round(float(u_grp['MD'].min()), 2),
                        "Base (m)": round(float(u_grp['MD'].max()), 2),
                        "Thickness (m)": round(u_thick, 2),
                        "Avg PHID": round(avg_phi, 3) if not np.isnan(avg_phi) else None,
                        "Avg VSH": round(avg_vsh, 3) if not np.isnan(avg_vsh) else None,
                        "Avg SW": round(avg_sw, 3) if not np.isnan(avg_sw) else None,
                        "Avg K (mD)": round(avg_k, 2) if not np.isnan(avg_k) else None
                    })
                if unit_rows:
                    st.markdown("**ملخص الوحدات المكمنية (Reservoir Units) ضمن النطاق:**")
                    st.dataframe(pd.DataFrame(unit_rows), use_container_width=True)
        else:
            st.info("اضبط إعدادات النفاذية وحدود القطع في القائمة الجانبية ثم اضغط (تشغيل المرحلة 2).")

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
            st.info("اضبط إعدادات التنبؤ في القائمة الجانبية ثم اضغط (تشغيل المرحلة 1).")

    with tab_tracks:
        st.subheader(f"📊 لوحة المسارات — {selected_well}")
        default_tr = [t for t in ['GR', 'VSH', 'PHID', 'SW', 'PERM_AI'] if t in all_plot_logs]
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
        st.download_button("تحميل جدول النتائج الكامل (CSV)", data=csv_bytes, file_name=f"{formation_name}_AI_Stage1_Stage2.csv", mime="text/csv")
        st.dataframe(master_df.head(100), use_container_width=True)
