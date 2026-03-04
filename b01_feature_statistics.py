import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import json
import pickle



def extract_vectors(root_dir, feat_manifest, feat_type="ECG_Features"):
    data_list = []
    user_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]

    for user_dir in tqdm(user_dirs, desc="生成特征向量"):
        external_id = user_dir.name

        for group_dir in user_dir.iterdir():
            if not group_dir.is_dir():
                continue

            feat_dir = group_dir / feat_type
            if not feat_dir.exists():
                continue

            group_id = group_dir.name  # 数据对 ID
            feat_data = {}
            vec_main = []
            valid = True

            for feat in feat_manifest:
                file_path = feat_dir / f"{feat}.npy"
                if not file_path.exists():
                    valid = False
                    break

                try:
                    val = np.load(file_path)
                    val = val[np.isfinite(val)]
                    if len(val) < 3:
                        print('有效数据量不足')
                        valid = False
                        break

                    # 基础统计量
                    m, s, md= np.nanmean(val), np.nanstd(val), np.nanmedian(val)
                    sk, ku = skew(val), kurtosis(val)

                    if not np.all(np.isfinite([m, s, sk, ku])):
                        valid = False
                        break

                    # 分布直方图
                    eps = 1e-6
                    hist, _ = np.histogram(val, bins=20, range=(np.min(val), np.max(val) + eps), density=True)

                    feat_data[feat] = {
                        "mean": m,
                        "std": s,
                        "median": md,
                        "skew": sk,
                        "kurtosis": ku,
                        "hist": hist.astype(np.float32)
                    }

                    vec_main.extend([m, s, md])

                except Exception as e:
                    valid = False
                    print(f"Error loading {file_path}: {e}")
                    break

            if valid:
                data_list.append({
                    "externalid": external_id,
                    "groupid": group_id,
                    "vector": np.array(vec_main, dtype=np.float32),
                    "distributions": feat_data
                })

    return data_list


root_path = r"data"
ecg_manifest = ['ECG_iqr', 'ECG_kurtosis', 'ECG_rms', 'ECG_skewness', 'PR_interval', 'PSD_HF_power',
                'PSD_LF_HF_ratio', 'PSD_LF_power', 'PSD_MF_power', 'PSD_peak_freq', 'PSD_total_power',
                'P_R_ratio', 'QRS_area', 'QRS_asymmetry', 'QRS_energy_ratio', 'QRS_time_asymmetry',
                'QRS_width', 'QT_interval', 'Q_R_ratio', 'RMSSD', 'RR_entropy', 'R_S_ratio', 'R_amp',
                'R_amp_mean', 'R_amp_std', 'R_height_cv', 'R_peak_jitter', 'R_upstroke_slope',
                'ST_energy_ratio', 'T_R_ratio', 'T_energy_ratio', 'beat_corr_mean', 'beat_corr_std',
                'beat_rms_mean', 'beat_rms_std', 'mean_RR', 'pNN50', 'spec_centroid', 'spec_flatness',
                'spec_kurtosis', 'spec_skew', 'std_RR', 'template_corr_mean', 'template_corr_std',
                'template_dtw_mean', 'template_dtw_std', 'template_kurtosis', 'template_skew',
                'template_std', 'valid_beat_ratio']



ecg_vector = extract_vectors(root_path, ecg_manifest, feat_type='ECG_Features')
ecg_save_path = r"ecg_vectors.pkl"
with open(ecg_save_path, 'wb') as f:
    pickle.dump(ecg_vector, f)



ppg_manifest = ['green-AI_mean', 'green-AI_std', 'green-IPA_mean', 'green-IPA_std', 'green-Max_Upstroke_Slope_mean',
                'green-Max_Upstroke_Slope_std', 'green-PSD_freq_0', 'green-PSD_freq_1', 'green-PSD_freq_2',
                'green-PSD_pow_0', 'green-PSD_pow_1', 'green-PSD_pow_2', 'green-Pulse_Width_50_mean',
                'green-Pulse_Width_50_std', 'green-RMSSD', 'green-Rise_time_mean', 'green-Rise_time_std',
                'green-mean_IBI', 'infrared-AI_mean', 'infrared-AI_std', 'infrared-IPA_mean',
                'infrared-IPA_std', 'infrared-Max_Upstroke_Slope_mean', 'infrared-Max_Upstroke_Slope_std', 'infrared-PSD_freq_0',
                'infrared-PSD_freq_1', 'infrared-PSD_freq_2', 'infrared-PSD_pow_0', 'infrared-PSD_pow_1',
                'infrared-PSD_pow_2', 'infrared-Pulse_Width_50_mean', 'infrared-Pulse_Width_50_std',
                'infrared-RMSSD', 'infrared-Rise_time_mean', 'infrared-Rise_time_std', 'infrared-mean_IBI', 'red-AI_mean',
                'red-AI_std', 'red-IPA_mean', 'red-IPA_std', 'red-Max_Upstroke_Slope_mean', 'red-Max_Upstroke_Slope_std',
                'red-PSD_freq_0', 'red-PSD_freq_1', 'red-PSD_freq_2', 'red-PSD_pow_0', 'red-PSD_pow_1',
                'red-PSD_pow_2', 'red-Pulse_Width_50_mean', 'red-Pulse_Width_50_std', 'red-RMSSD', 'red-Rise_time_mean',
                'red-Rise_time_std', 'red-mean_IBI']

ppg_vector = extract_vectors(root_path, ppg_manifest, feat_type="PPG_Features")
ppg_save_path = r"ppg_vectors.pkl"
with open(ppg_save_path, 'wb') as f:
    pickle.dump(ppg_vector, f)

