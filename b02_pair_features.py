from collections import defaultdict
import numpy as np
import pickle
from tqdm import tqdm
import joblib
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb



def get_pair_features(a, b, feat_manifest):
    vec = []
    va = a["vector"].astype(float)
    vb = b["vector"].astype(float)
    eps = 1e-6
    # =====================================================
    # 1️⃣ 向量级几何特征（identity geometry）
    # =====================================================
    diff = np.abs(va - vb)
    ratio = va / (vb + eps)
    # 全局距离 / 相似性
    mask = (~np.isnan(va)) & (~np.isnan(vb))
    if np.sum(mask) > 0:
        va_m = va[mask]
        vb_m = vb[mask]
        diff_m = np.abs(va_m - vb_m)
        euclidean = np.linalg.norm(diff_m)
        norm_a = np.linalg.norm(va_m)
        norm_b = np.linalg.norm(vb_m)
        denom = norm_a * norm_b + eps
        cosine = np.dot(va_m, vb_m) / denom
        cosine_dist = 1.0 - cosine
        if np.sum(mask) >= 2:
            correlation = np.corrcoef(va_m, vb_m)[0, 1]
        else:
            correlation = np.nan
    else:
        euclidean = np.nan
        cosine_dist = np.nan
        correlation = np.nan
    # ---- 核心向量特征 ----
    vec.extend(diff)
    vec.extend(ratio)
    vec.extend([euclidean, correlation, cosine_dist])
    # ---- 稳定性 / 一致性（新增，重点）----
    vec.append(np.nanmean(diff)) # 平均差异强度
    vec.append(np.nanstd(diff)) # 差异是否集中
    vec.append(np.nanmean(ratio))
    vec.append(np.nanstd(ratio))
    # ---- 非对称身份线索（轻量，仅 1 个）----
    vec.append(np.nanmean(va - vb)) # A→B vs B→A
    # =====================================================
    # 2️⃣ 分布级特征（beat / template consistency）
    # =====================================================
    skew_diffs = []
    kurt_diffs = []
    for feat in feat_manifest:
        da = a["distributions"][feat]
        db = b["distributions"][feat]
        # ---- 偏度 / 峰度 ----
        if not np.isnan(da["skew"]) and not np.isnan(db["skew"]):
            skew_diff = abs(da["skew"] - db["skew"])
            skew_prod = da["skew"] * db["skew"]
        else:
            skew_diff = np.nan
            skew_prod = np.nan
        vec.append(skew_diff)
        vec.append(skew_prod)
        skew_diffs.append(skew_diff)
        if not np.isnan(da["kurtosis"]) and not np.isnan(db["kurtosis"]):
            kurt_diff = abs(da["kurtosis"] - db["kurtosis"])
            kurt_prod = da["kurtosis"] * db["kurtosis"]
        else:
            kurt_diff = np.nan
            kurt_prod = np.nan
        vec.append(kurt_diff)
        vec.append(kurt_prod)
        kurt_diffs.append(kurt_diff)
        # ---- JS divergence（稳定版）----
        p = np.asarray(da["hist"], dtype=float) + eps
        q = np.asarray(db["hist"], dtype=float) + eps
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * (
            np.sum(p * np.log(p / m)) +
            np.sum(q * np.log(q / m))
        )
        vec.append(js)
        # ---- Histogram intersection ----
        intersection = np.sum(np.minimum(p, q))
        vec.append(intersection)
    # =====================================================
    # 3️⃣ 分布一致性汇总特征（VERY IMPORTANT）
    # =====================================================
    vec.append(np.nanmean(skew_diffs))
    vec.append(np.nanstd(skew_diffs))
    vec.append(np.nanmean(kurt_diffs))
    vec.append(np.nanstd(kurt_diffs))
    return np.asarray(vec, dtype=np.float32)

def create_pair_level_dataset(data_list, feat_manifest, n_reg=5, max_neg_users=None):
    user_groups = defaultdict(list)
    for d in data_list:
        user_groups[d["externalid"]].append(d)

    uids = sorted(user_groups.keys())

    X, y = [], []
    probe_ids = []
    reg_user_ids = []
    groups = []

    for uid in tqdm(uids, desc="Pair-level construction"):
        samples = user_groups[uid]
        if len(samples) <= n_reg:
            continue

        reg = samples[:n_reg]
        probes = samples[n_reg:]

        # ---------- 正样本 ----------
        for probe in probes:
            probe_id = id(probe)
            for r in reg:
                X.append(get_pair_features(probe, r, feat_manifest))
                y.append(1)
                probe_ids.append(probe_id)
                reg_user_ids.append(uid)   # ← 正确注册用户
                groups.append(probe_id)

        # ---------- 负样本 ----------
        neg_uids = [u for u in uids if u != uid]
        if max_neg_users is not None:
            neg_uids = neg_uids[:max_neg_users]

        for neg_uid in neg_uids:
            neg_samples = user_groups[neg_uid]
            if len(neg_samples) < n_reg:
                continue

            neg_reg = neg_samples[:n_reg]
            for probe in probes:
                probe_id = id(probe)
                for r in neg_reg:
                    X.append(get_pair_features(probe, r, feat_manifest))
                    y.append(0)
                    probe_ids.append(probe_id)
                    reg_user_ids.append(neg_uid)  # ← 关键
                    groups.append(probe_id)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=int),
        np.asarray(probe_ids),
        np.asarray(reg_user_ids),
        np.asarray(groups),
    )

def train_final_model(X, y, name):

    print(f"Final training samples: {X.shape[0]}")
    print(f"Feature dim: {X.shape[1]}")
    print(f"Pos / Neg: {np.sum(y==1)} / {np.sum(y==0)}")

    # 1. scaler（⚠️ 用全部数据 fit）
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. final XGB
    model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            tree_method='hist',
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

    model.fit(X_scaled, y)

    # 3. 保存
    joblib.dump(scaler, fr"{name}_scaler.pkl")
    joblib.dump(model, fr"{name}_xgb.pkl")

    print(f" Final {name} model & scaler saved.")






if __name__ == "__main__":
    with open(r"ecg_vectors.pkl", "rb") as f:
        ecg_data = pickle.load(f)

    ecg_feat_manifest = ['ECG_iqr', 'ECG_kurtosis', 'ECG_rms', 'ECG_skewness', 'PR_interval', 'PSD_HF_power',
                         'PSD_LF_HF_ratio', 'PSD_LF_power', 'PSD_MF_power', 'PSD_peak_freq', 'PSD_total_power',
                         'P_R_ratio', 'QRS_area', 'QRS_asymmetry', 'QRS_energy_ratio', 'QRS_time_asymmetry',
                         'QRS_width', 'QT_interval', 'Q_R_ratio', 'RMSSD', 'RR_entropy', 'R_S_ratio', 'R_amp',
                         'R_amp_mean', 'R_amp_std', 'R_height_cv', 'R_peak_jitter', 'R_upstroke_slope',
                         'ST_energy_ratio', 'T_R_ratio', 'T_energy_ratio', 'beat_corr_mean', 'beat_corr_std',
                         'beat_rms_mean', 'beat_rms_std', 'mean_RR', 'pNN50', 'spec_centroid', 'spec_flatness',
                         'spec_kurtosis', 'spec_skew', 'std_RR', 'template_corr_mean', 'template_corr_std',
                         'template_dtw_mean', 'template_dtw_std', 'template_kurtosis', 'template_skew',
                         'template_std', 'valid_beat_ratio']

    with open(r"ppg_vectors.pkl", "rb") as f:
        ppg_data = pickle.load(f)

    ppg_feat_manifest = ['green-AI_mean', 'green-AI_std', 'green-IPA_mean', 'green-IPA_std',
                         'green-Max_Upstroke_Slope_mean',
                         'green-Max_Upstroke_Slope_std', 'green-PSD_freq_0', 'green-PSD_freq_1', 'green-PSD_freq_2',
                         'green-PSD_pow_0', 'green-PSD_pow_1', 'green-PSD_pow_2', 'green-Pulse_Width_50_mean',
                         'green-Pulse_Width_50_std', 'green-RMSSD', 'green-Rise_time_mean', 'green-Rise_time_std',
                         'green-mean_IBI', 'infrared-AI_mean', 'infrared-AI_std', 'infrared-IPA_mean',
                         'infrared-IPA_std',
                         'infrared-Max_Upstroke_Slope_mean', 'infrared-Max_Upstroke_Slope_std', 'infrared-PSD_freq_0',
                         'infrared-PSD_freq_1', 'infrared-PSD_freq_2', 'infrared-PSD_pow_0', 'infrared-PSD_pow_1',
                         'infrared-PSD_pow_2', 'infrared-Pulse_Width_50_mean', 'infrared-Pulse_Width_50_std',
                         'infrared-RMSSD',
                         'infrared-Rise_time_mean', 'infrared-Rise_time_std', 'infrared-mean_IBI', 'red-AI_mean',
                         'red-AI_std',
                         'red-IPA_mean', 'red-IPA_std', 'red-Max_Upstroke_Slope_mean', 'red-Max_Upstroke_Slope_std',
                         'red-PSD_freq_0', 'red-PSD_freq_1', 'red-PSD_freq_2', 'red-PSD_pow_0', 'red-PSD_pow_1',
                         'red-PSD_pow_2',
                         'red-Pulse_Width_50_mean', 'red-Pulse_Width_50_std', 'red-RMSSD', 'red-Rise_time_mean',
                         'red-Rise_time_std', 'red-mean_IBI']

    X_ecg, y_ecg, _, _, _ = create_pair_level_dataset(ecg_data, ecg_feat_manifest, n_reg=5, max_neg_users=40)
    X_ppg, y_ppg, _, _, _ = create_pair_level_dataset(ppg_data, ppg_feat_manifest, n_reg=5, max_neg_users=40)

    train_final_model(X_ecg, y_ecg, name='ecg')
    train_final_model(X_ppg, y_ppg, name='ppg')
