import os
import json
import joblib
import random
import numpy as np
import pandas as pd
import neurokit2 as nk
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter, welch
from pathlib import Path
from typing import Dict
from typing import List
from dataclasses import dataclass
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold



# 导入必要的类
@dataclass
class BiometricData:
    registration: Dict[str, Dict[str, Dict[str, str]]]  # {用户ID: {pair: {文件名: 内容}}}
    test: Dict[str, Dict[str, str]]  # {文件夹名: {文件名: 内容}}

    def __repr__(self):
        reg_count = sum(len(sessions) for sessions in self.registration.values())
        test_count = len(self.test)
        return (f"BiometricData(\n"
                f"  registration: {len(self.registration)} users, {reg_count} sessions,\n"
                f"  test: {test_count} folders\n"
                f")")

class DataReader:
    def __init__(self, registration_folder: str, test_folder: str):
        self.registration_folder = registration_folder
        self.test_folder = test_folder

    def read_registration_files(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        读取注册数据

        参数:
            root_folder: 根文件夹路径

        返回:
            嵌套字典: {
                'HNU32001': {
                    'pair': {
                        'txt': '内容',
                        'txt': '内容',
                        'txt': '内容'
                    },
                    'pair': {...}
                },
                'HNU32002': {...}
            }
        """

        root_path = Path(self.registration_folder)
        registration_data = {}

        # 遍历第一层文件夹
        for level1_folder in root_path.iterdir():
            if level1_folder.is_dir():
                user_id = level1_folder.name
                registration_data[user_id] = {}

                # 遍历第二层文件夹
                for level2_folder in level1_folder.iterdir():
                    if level2_folder.is_dir():
                        session_id = level2_folder.name
                        registration_data[user_id][session_id] = {}

                        # 读取txt文件
                        txt_files = list(level2_folder.glob('*.txt'))
                        for txt_file in txt_files:
                            try:
                                content = txt_file.read_text(encoding='utf-8')
                                registration_data[user_id][session_id][txt_file.stem] = content
                            except Exception as e:
                                print(f"读取 {txt_file} 失败: {e}")

        return registration_data


    def read_test_files(self) -> Dict[str, Dict[str, str]]:
        """
        读取测试数据

        参数:
            root_folder: 根文件夹路径

        返回:
            嵌套字典: {
                'dir': {
                    'ACC_20251121_155735411': '内容',
                    'ECG_20251121_155735407': '内容',
                    'PPG_20251117_190549029': '内容'
                },
                'dir2': {
                    'ACC': '内容',
                    ...
                }
            }
        """
        root_path = Path(self.test_folder)
        test_data = {}

        # 遍历第一层文件夹
        for level1_folder in root_path.iterdir():
            if level1_folder.is_dir():
                folder_name = level1_folder.name
                test_data[folder_name] = {}

                # 读取txt文件
                txt_files = list(level1_folder.glob('*.txt'))
                for txt_file in txt_files:
                    try:
                        content = txt_file.read_text(encoding='utf-8')
                        test_data[folder_name][txt_file.stem] = content
                    except Exception as e:
                        print(f"读取 {txt_file} 失败: {e}")

        return test_data


    def load_data(self) -> BiometricData:
        """
        统一接口：加载所有数据

        参数:
            registration_folder: 注册数据文件夹路径
            test_folder: 测试数据文件夹路径

        返回:
            BiometricData 对象（嵌套字典结构）
        """

        registration_data = self.read_registration_files()
        test_data = self.read_test_files()

        return BiometricData(
            registration=registration_data,
            test=test_data
        )

class DataOutput:
    def __init__(self, results: List[List], output_file: str = "./测试结果.xlsx"):
        self.results = results
        self.output_file = output_file

    def save_results_to_xlsx(self):
        """
        将结果保存为xlsx文件

        参数:
            results: 二维列表 [[user_id, data_pairs_id, prob], [...], ...]
            output_file: 输出文件名
        """
        # 创建 DataFrame
        df = pd.DataFrame(self.results, columns=['user_id', 'data_pairs_id', 'prob'])

        # 确保数据类型
        df['user_id'] = df['user_id'].astype(str)
        df['data_pairs_id'] = df['data_pairs_id'].astype(str)
        df['prob'] = df['prob'].astype(float)

        # 保存为 xlsx
        df.to_excel(self.output_file, index=False, engine='openpyxl')
        print(f"✓ 已保存 {len(df)} 条记录到 {self.output_file}")


# ECG 部分代码
def load_ecg_from_content(content):
    ecg_values = []
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if "voltage" in data:  # Assuming ECG is under "voltage"
            ecg_values.extend([v["value"] for v in data["voltage"]])
    return np.array(ecg_values, dtype=float)

def _pad_or_trim(x, target_len):
    """保证 beat 长度一致（零填充或截断）"""
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")
    return x[:target_len]

def _safe_entropy(hist):
    p = hist / (np.sum(hist) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def _dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )
    return dtw[n, m] / (n + m)

def extract_ecg_identity_features(info, waves, ecg, FS_ECG=500):
    feats = {
        "mean_RR": np.nan,
        "std_RR": np.nan,
        "RMSSD": np.nan,
        "pNN50": np.nan,
        "QRS_width": np.nan,
        "PR_interval": np.nan,
        "QT_interval": np.nan,
        "R_amp": np.nan,
        "Q_R_ratio": np.nan,
        "R_S_ratio": np.nan,
        "P_R_ratio": np.nan,
        "T_R_ratio": np.nan,
        "R_upstroke_slope": np.nan,
    }
    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            return feats
        rr = np.diff(rpeaks) / FS_ECG
        feats["mean_RR"] = np.mean(rr)
        feats["std_RR"] = np.std(rr)
        feats["RMSSD"] = np.sqrt(np.mean(np.diff(rr) ** 2))
        feats["pNN50"] = np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr)
        for key in ["ECG_R_Offsets", "ECG_R_Onsets", "ECG_P_Onsets", "ECG_T_Offsets",
                    "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Peaks"]:
            if key in waves:
                waves[key] = np.asarray(waves[key])
        feats["QRS_width"] = np.nanmean((waves["ECG_R_Offsets"] - waves["ECG_R_Onsets"]) / FS_ECG)
        feats["PR_interval"] = np.nanmean((waves["ECG_R_Onsets"] - waves["ECG_P_Onsets"]) / FS_ECG)
        feats["QT_interval"] = np.nanmean((waves["ECG_T_Offsets"] - waves["ECG_R_Onsets"]) / FS_ECG)
        r_amp = ecg[rpeaks]
        feats["R_amp"] = np.mean(r_amp)
        q_peaks = waves["ECG_Q_Peaks"]
        s_peaks = waves["ECG_S_Peaks"]
        t_peaks = waves["ECG_T_Peaks"]
        p_peaks = waves["ECG_P_Peaks"]
        valid = (~np.isnan(q_peaks)) & (~np.isnan(s_peaks)) & (~np.isnan(p_peaks)) & (~np.isnan(t_peaks))
        if not np.any(valid):
            return feats
        q_idx = q_peaks[valid].astype(int)
        r_idx = rpeaks[valid].astype(int)
        s_idx = s_peaks[valid].astype(int)
        p_idx = p_peaks[valid].astype(int)
        t_idx = t_peaks[valid].astype(int)
        q_amp = np.abs(ecg[q_idx])
        r_amp = np.abs(ecg[r_idx])
        s_amp = np.abs(ecg[s_idx])
        p_amp = np.abs(ecg[p_idx])
        t_amp = np.abs(ecg[t_idx])
        feats["Q_R_ratio"] = np.mean(q_amp / (r_amp + 1e-6))
        feats["R_S_ratio"] = np.mean(r_amp / (s_amp + 1e-6))
        feats["P_R_ratio"] = np.mean(p_amp / (r_amp + 1e-6))
        feats["T_R_ratio"] = np.mean(t_amp / (r_amp + 1e-6))
        time_diff = (r_idx - q_idx) / FS_ECG
        valid_diff = time_diff > 0
        if np.any(valid_diff):
            slopes = (r_amp[valid_diff] - q_amp[valid_diff]) / time_diff[valid_diff]
            feats["R_upstroke_slope"] = np.mean(slopes)
    except Exception:
        pass
    return feats

def extract_ecg_psd_features(ecg_clean, FS_ECG):
    feats = {
        "PSD_peak_freq": np.nan,
        "PSD_total_power": np.nan,
        "PSD_LF_power": np.nan,
        "PSD_MF_power": np.nan,
        "PSD_HF_power": np.nan,
        "PSD_LF_HF_ratio": np.nan,
    }
    try:
        nperseg = min(len(ecg_clean), int(2 * FS_ECG))
        freqs, psd = welch(ecg_clean, fs=FS_ECG, nperseg=nperseg)
        mask_total = (freqs >= 0.5) & (freqs <= 40)
        f_val, p_val = freqs[mask_total], psd[mask_total]
        feats["PSD_peak_freq"] = f_val[np.argmax(p_val)]
        feats["PSD_total_power"] = np.trapz(p_val, f_val)
        def band_power(f, p, low, high):
            m = (f >= low) & (f < high)
            if not np.any(m):
                return np.nan
            return np.trapz(p[m], f[m])
        feats["PSD_LF_power"] = band_power(f_val, p_val, 0.5, 5)
        feats["PSD_MF_power"] = band_power(f_val, p_val, 5, 15)
        feats["PSD_HF_power"] = band_power(f_val, p_val, 15, 40)
        if feats["PSD_HF_power"] > 0:
            feats["PSD_LF_HF_ratio"] = feats["PSD_LF_power"] / feats["PSD_HF_power"]
    except Exception:
        pass
    return feats

def extract_ecg_template_features(ecg_clean, info, FS_ECG):
    feats = {
        "template_std": np.nan,
        "template_skew": np.nan,
        "template_kurtosis": np.nan,
        "template_corr_mean": np.nan,
        "template_corr_std": np.nan,
    }
    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats
        rr = np.diff(rpeaks) / FS_ECG
        median_rr = np.median(rr)
        win_pre = min(0.25, 0.35 * median_rr)
        win_post = min(0.45, 0.65 * median_rr)
        n_pre, n_post = int(win_pre * FS_ECG), int(win_post * FS_ECG)
        beats = []
        for r in rpeaks:
            if r - n_pre < 0 or r + n_post >= len(ecg_clean):
                continue
            beat = ecg_clean[r - n_pre: r + n_post]
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
            beats.append(beat)
        if len(beats) < 5:
            return feats
        beats = np.asarray(beats)
        template = beats.mean(axis=0)
        feats["template_std"] = np.std(template)
        feats["template_skew"] = skew(template)
        feats["template_kurtosis"] = kurtosis(template)
        corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
        feats["template_corr_mean"] = np.nanmean(corrs)
        feats["template_corr_std"] = np.nanstd(corrs)
    except Exception:
        pass
    return feats

def extract_ecg_global_morphology(ecg_clean):
    feats = {
        "ECG_skewness": np.nan,
        "ECG_kurtosis": np.nan,
        "ECG_rms": np.nan,
        "ECG_iqr": np.nan,
    }
    try:
        ecg_clean = ecg_clean - np.mean(ecg_clean)
        feats["ECG_skewness"] = skew(ecg_clean)
        feats["ECG_kurtosis"] = kurtosis(ecg_clean)
        feats["ECG_rms"] = np.sqrt(np.mean(ecg_clean ** 2))
        q95, q05 = np.percentile(ecg_clean, [95, 5])
        feats["ECG_iqr"] = q95 - q05
    except Exception:
        pass
    return feats

def extract_ecg_qrs_geometry(ecg_clean, waves):
    feats = {
        "QRS_area": np.nan,
        "QRS_asymmetry": np.nan,
    }
    try:
        q = np.asarray(waves.get("ECG_Q_Peaks", []))
        s = np.asarray(waves.get("ECG_S_Peaks", []))
        valid = (~np.isnan(q)) & (~np.isnan(s))
        if not np.any(valid):
            return feats
        q, s = q[valid].astype(int), s[valid].astype(int)
        areas = []
        skews = []
        for qi, si in zip(q, s):
            if qi < 0 or si >= len(ecg_clean):
                continue
            seg = ecg_clean[qi:si]
            areas.append(np.trapz(np.abs(seg)))
            skews.append(skew(seg))
        feats["QRS_area"] = np.nanmean(areas)
        feats["QRS_asymmetry"] = np.nanmean(skews)
    except Exception:
        pass
    return feats

def extract_ecg_r_height_variability(ecg_clean, info):
    feats = {
        "R_height_cv": np.nan,
    }
    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats
        r_amps = ecg_clean[rpeaks]
        feats["R_height_cv"] = np.std(r_amps) / (np.mean(np.abs(r_amps)) + 1e-6)
    except Exception:
        pass
    return feats

def extract_ecg_beat_consistency_features(ecg_clean, info, FS_ECG):
    feats = {
        "beat_rms_mean": np.nan,
        "beat_rms_std": np.nan,
        "beat_corr_mean": np.nan,
        "beat_corr_std": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats

        rr = np.diff(rpeaks) / FS_ECG
        median_rr = np.median(rr)

        win_pre = int(min(0.25, 0.35 * median_rr) * FS_ECG)
        win_post = int(min(0.45, 0.65 * median_rr) * FS_ECG)
        beat_len = win_pre + win_post

        beats = []
        for r in rpeaks:
            if r - win_pre < 0 or r + win_post >= len(ecg_clean):
                continue
            beat = ecg_clean[r - win_pre:r + win_post]
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
            beat = _pad_or_trim(beat, beat_len)
            beats.append(beat)

        if len(beats) < 5:
            return feats

        beats = np.asarray(beats)
        template = np.nanmean(beats, axis=0)

        rms = np.sqrt(np.nanmean((beats - template) ** 2, axis=1))
        corrs = [np.corrcoef(b, template)[0, 1] for b in beats]

        feats["beat_rms_mean"] = np.nanmean(rms)
        feats["beat_rms_std"] = np.nanstd(rms)
        feats["beat_corr_mean"] = np.nanmean(corrs)
        feats["beat_corr_std"] = np.nanstd(corrs)

    except Exception:
        pass

    return feats

def extract_ecg_energy_ratio_features(ecg_clean, waves, info):
    feats = {
        "QRS_energy_ratio": np.nan,
        "ST_energy_ratio": np.nan,
        "T_energy_ratio": np.nan,
    }

    try:
        q = np.asarray(waves.get("ECG_Q_Peaks", []), dtype=float)
        s = np.asarray(waves.get("ECG_S_Peaks", []), dtype=float)
        t = np.asarray(waves.get("ECG_T_Peaks", []), dtype=float)
        t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)
        r = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)

        valid = (~np.isnan(q)) & (~np.isnan(s)) & (~np.isnan(t)) & (~np.isnan(r))
        if not np.any(valid):
            return feats

        has_toff = len(t_off) == len(q)
        if has_toff:
            valid = valid & (~np.isnan(t_off))

        q, r, s, t = q[valid].astype(int), r[valid].astype(int), \
                     s[valid].astype(int), t[valid].astype(int)

        t_off = t_off[valid].astype(int) if has_toff else None

        qrs_energy, st_energy, t_energy = [], [], []

        for i in range(len(q)):
            qi, ri, si, ti = q[i], r[i], s[i], t[i]
            if qi < 0 or si <= qi or ti <= si or ti >= len(ecg_clean):
                continue

            qrs_energy.append(np.sum(ecg_clean[qi:si] ** 2))
            st_energy.append(np.sum(ecg_clean[si:ti] ** 2))

            if t_off is not None and t_off[i] > ti and t_off[i] <= len(ecg_clean):
                t_seg = ecg_clean[ti:t_off[i]]
            else:
                t_seg = ecg_clean[ti:min(ti + (si - qi), len(ecg_clean))]

            t_energy.append(np.sum(t_seg ** 2))

        total = np.sum(qrs_energy) + np.sum(st_energy) + np.sum(t_energy)
        if total <= 0:
            return feats

        feats["QRS_energy_ratio"] = np.sum(qrs_energy) / total
        feats["ST_energy_ratio"] = np.sum(st_energy) / total
        feats["T_energy_ratio"] = np.sum(t_energy) / total

    except Exception:
        pass

    return feats

def extract_ecg_spectral_shape_features(ecg_clean, FS_ECG):
    feats = {
        "spec_centroid": np.nan,
        "spec_flatness": np.nan,
        "spec_skew": np.nan,
        "spec_kurtosis": np.nan,
    }

    try:
        freqs, psd = welch(ecg_clean, fs=FS_ECG,
                           nperseg=min(len(ecg_clean), FS_ECG * 2))
        mask = (freqs >= 0.5) & (freqs <= 40)
        f, p = freqs[mask], psd[mask]

        if len(f) < 2 or np.all(p <= 0):
            return feats

        p = p + 1e-12
        logp = np.log(p)

        feats["spec_centroid"] = np.sum(f * p) / np.sum(p)
        feats["spec_flatness"] = np.exp(np.mean(logp)) / np.mean(p)
        feats["spec_skew"] = skew(logp)
        feats["spec_kurtosis"] = kurtosis(logp)

    except Exception:
        pass

    return feats

def extract_ecg_entropy_features(info, FS_ECG):
    feats = {
        "RR_entropy": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats

        rr = np.diff(rpeaks) / FS_ECG
        bins = min(10, max(3, len(rr) // 2))
        hist, _ = np.histogram(rr, bins=bins)

        feats["RR_entropy"] = _safe_entropy(hist)

    except Exception:
        pass

    return feats

def extract_ecg_r_amplitude_features(ecg_clean, info):
    feats = {
        "R_amp_mean": np.nan,
        "R_amp_std": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            return feats

        r_amp = ecg_clean[rpeaks]
        feats["R_amp_mean"] = np.mean(r_amp)
        feats["R_amp_std"] = np.std(r_amp)

    except Exception:
        pass

    return feats

def extract_ecg_template_dtw_features(ecg_clean, info, FS_ECG):
    feats = {
        "template_dtw_mean": np.nan,
        "template_dtw_std": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats

        rr = np.diff(rpeaks) / FS_ECG
        median_rr = np.median(rr)

        win_pre = int(min(0.25, 0.35 * median_rr) * FS_ECG)
        win_post = int(min(0.45, 0.65 * median_rr) * FS_ECG)
        beat_len = win_pre + win_post

        beats = []
        for r in rpeaks:
            if r - win_pre < 0 or r + win_post >= len(ecg_clean):
                continue
            beat = ecg_clean[r - win_pre:r + win_post]
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
            beat = _pad_or_trim(beat, beat_len)
            beat = beat[::4]
            beats.append(beat)

        if len(beats) < 5:
            return feats

        beats = np.asarray(beats)
        template = np.mean(beats, axis=0)
        template = template[::4]

        dtws = [_dtw_distance(b, template) for b in beats]

        feats["template_dtw_mean"] = np.mean(dtws)
        feats["template_dtw_std"] = np.std(dtws)

    except Exception:
        pass

    return feats

def extract_ecg_qrs_time_asymmetry(waves, info, FS_ECG):
    feats = {
        "QRS_time_asymmetry": np.nan,
    }

    try:
        q = np.asarray(waves.get("ECG_Q_Peaks", []))
        r = np.asarray(info.get("ECG_R_Peaks", []))
        s = np.asarray(waves.get("ECG_S_Peaks", []))

        valid = (~np.isnan(q)) & (~np.isnan(r)) & (~np.isnan(s))
        if not np.any(valid):
            return feats

        q = q[valid].astype(int)
        r = r[valid].astype(int)
        s = s[valid].astype(int)

        up = (r - q) / FS_ECG
        down = (s - r) / FS_ECG

        valid_t = (up > 0) & (down > 0)
        if not np.any(valid_t):
            return feats

        feats["QRS_time_asymmetry"] = np.mean(up[valid_t] / down[valid_t])

    except Exception:
        pass

    return feats

def extract_ecg_r_peak_jitter(info, FS_ECG):
    feats = {
        "R_peak_jitter": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 5:
            return feats

        rr = np.diff(rpeaks) / FS_ECG
        mean_rr = np.mean(rr)

        expected = np.arange(len(rpeaks)) * mean_rr
        actual = rpeaks / FS_ECG

        jitter = actual - expected
        feats["R_peak_jitter"] = np.std(jitter)

    except Exception:
        pass

    return feats

def extract_ecg_valid_beat_ratio(info, signal_len, FS_ECG):
    feats = {
        "valid_beat_ratio": np.nan,
    }

    try:
        rpeaks = info["ECG_R_Peaks"]
        duration = signal_len / FS_ECG

        if duration <= 0:
            return feats

        # 理论心搏数（假设 40–180 bpm）
        min_beats = duration * 40 / 60
        max_beats = duration * 180 / 60

        valid = len(rpeaks)
        if min_beats <= valid <= max_beats:
            feats["valid_beat_ratio"] = valid / max_beats
        else:
            feats["valid_beat_ratio"] = valid / (max_beats + 1e-6)

    except Exception:
        pass

    return feats

def ecg_quality_check(ecg_clean, FS_ECG, thresh=0.3):
    try:
        quality = nk.ecg_quality(ecg_clean, sampling_rate=FS_ECG)
        return np.nanmean(quality) >= thresh
    except Exception:
        return False

def ecg_features_summary(ecg, FS_ECG=500):
    # 初始化所有可能的特征为 NaN
    feats = {
        "mean_RR": np.nan,
        "std_RR": np.nan,
        "RMSSD": np.nan,
        "pNN50": np.nan,
        "QRS_width": np.nan,
        "PR_interval": np.nan,
        "QT_interval": np.nan,
        "R_amp": np.nan,
        "Q_R_ratio": np.nan,
        "R_S_ratio": np.nan,
        "P_R_ratio": np.nan,
        "T_R_ratio": np.nan,
        "R_upstroke_slope": np.nan,
        "PSD_peak_freq": np.nan,
        "PSD_total_power": np.nan,
        "PSD_LF_power": np.nan,
        "PSD_MF_power": np.nan,
        "PSD_HF_power": np.nan,
        "PSD_LF_HF_ratio": np.nan,
        "template_std": np.nan,
        "template_skew": np.nan,
        "template_kurtosis": np.nan,
        "template_corr_mean": np.nan,
        "template_corr_std": np.nan,
        "ECG_skewness": np.nan,
        "ECG_kurtosis": np.nan,
        "ECG_rms": np.nan,
        "ECG_iqr": np.nan,
        "QRS_area": np.nan,
        "QRS_asymmetry": np.nan,
        "R_height_cv": np.nan,
        "beat_rms_mean": np.nan,
        "beat_rms_std": np.nan,
        "beat_corr_mean": np.nan,
        "beat_corr_std": np.nan,
        "QRS_energy_ratio": np.nan,
        "ST_energy_ratio": np.nan,
        "T_energy_ratio": np.nan,
        "spec_centroid": np.nan,
        "spec_flatness": np.nan,
        "spec_skew": np.nan,
        "spec_kurtosis": np.nan,
        "RR_entropy": np.nan,
        "R_amp_mean": np.nan,
        "R_amp_std": np.nan,
        "template_dtw_mean": np.nan,
        "template_dtw_std": np.nan,
        "QRS_time_asymmetry": np.nan,
        "R_peak_jitter": np.nan,
        "valid_beat_ratio": np.nan
    }
    ecg_clean = None
    info = None
    waves = None
    rpeaks = None
    rr = None

    try:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=FS_ECG)
    except Exception:
        print('! ECG 清洗失败')
        return None  # 如果清洗失败，直接扔掉

    if ecg_clean is None:
        return None

    quality_pass = ecg_quality_check(ecg_clean, FS_ECG, thresh=0.3)
    if not quality_pass:
        print('! ECG 信号质量差')
        return None  # 未通过质量检查，直接扔掉

    try:
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=FS_ECG)
        rpeaks = info["ECG_R_Peaks"]
        rr = np.diff(rpeaks) / FS_ECG
    except Exception:
        pass

    if rpeaks is not None:
        try:
            _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=FS_ECG, method="dwt")
        except Exception:
            pass

    # 现在分别尝试更新每个特征组，即使依赖项缺失，也只影响该组（用NaN表示未计算）
    if ecg_clean is not None and info is not None and waves is not None:
        try:
            id_feats = extract_ecg_identity_features(info, waves, ecg_clean, FS_ECG)
            feats.update(id_feats)
        except Exception:
            pass

        try:
            feats.update(extract_ecg_psd_features(ecg_clean, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_global_morphology(ecg_clean))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_template_features(ecg_clean, info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_r_height_variability(ecg_clean, info))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_qrs_geometry(ecg_clean, waves))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_beat_consistency_features(ecg_clean, info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_energy_ratio_features(ecg_clean, waves, info))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_spectral_shape_features(ecg_clean, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_entropy_features(info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_r_amplitude_features(ecg_clean, info))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_template_dtw_features(ecg_clean, info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_qrs_time_asymmetry(waves, info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_r_peak_jitter(info, FS_ECG))
        except Exception:
            pass

        try:
            feats.update(extract_ecg_valid_beat_ratio(info, len(ecg_clean), FS_ECG))
        except Exception:
            pass


    return feats

def extract_ecg_features_sliding(ecg, feat_manifest, win_sec=10, step_sec=2, FS_ECG=500):
    win_len = int(win_sec * FS_ECG)
    step_len = int(step_sec * FS_ECG)
    starts = list(range(0, len(ecg) - win_len + 1, step_len))
    def compute_window(start):
        ecg_win = ecg[start:start + win_len]
        feats = ecg_features_summary(ecg_win, FS_ECG)
        return feats
    feats_list = []
    for s in starts:
        feats = compute_window(s)
        feats_list.append(feats)
    feats_list = [f for f in feats_list if f is not None]
    if not feats_list:
        print('! 该样本未计算出 ECG 特征，返回空数组')
        return {feat: np.array([], dtype=float) for feat in feat_manifest}
    feats_dict = {k: np.array([f.get(k, np.nan) for f in feats_list], dtype=float) for k in feats_list[0].keys()}
    return feats_dict

def ecg_compute_sample_features(pair_data, ecg_feat_manifest):
    feats_dict = {}
    for file_name, content in pair_data.items():
        if not file_name.lower().startswith("ecg"):
            continue
        ecg = load_ecg_from_content(content)
        feats_dict = extract_ecg_features_sliding(ecg, ecg_feat_manifest)
    if not any(len(v) > 0 for v in feats_dict.values()):
        print('! ECG 特征层级返回 None')
        return None

    distributions = {}
    vector = []

    for feat in ecg_feat_manifest:
        values = np.asarray(feats_dict.get(feat, []), dtype=float)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            print(f'! 计算出的特征为空-{feat}')
            print('! ECG 特征层级返回 None')
            return None

        elif len(values) < 3:
            print('! 计算出的特征有效值不足')
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            sk = np.nan
            kur = np.nan
            hist, _ = np.histogram(values, bins=20, range=(np.min(values), np.max(values) + 1e-6), density=True)
        else:
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            sk = skew(values)
            kur = kurtosis(values)
            hist, _ = np.histogram(values, bins=20, range=(np.min(values), np.max(values) + 1e-6), density=True)

        distributions[feat] = {"mean": mean, "std": std, "median": median, "skew": sk, "kurtosis": kur, "hist": hist}

        vector.extend([mean, std, median])

    vector = np.asarray(vector, dtype=float)
    return {"vector": vector, "distributions": distributions}

# PPG部分代码
def load_ppg_from_content(content):
    ppg = {"green": [], "red": [], "infrared": []}
    lines = content.splitlines()
    line_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        line_count += 1

        if "greenLight" in data:
            ppg["green"].append(data["greenLight"][0])
        if "redLight" in data:
            ppg["red"].append(data["redLight"][0])
        if "infraredLight" in data:
            ppg["infrared"].append(data["infraredLight"][0])

    if line_count < 1600:
        print('! PPG数据量不足1600行，返回None')
        return None

    return {k: np.array(v, dtype=float) for k, v in ppg.items()}

def remove_motion_artifacts(ppg_dict, fs, win_sec=2.0, step_sec=1.0,
                                          amp_thresh=10.0, diff_thresh=10.0, min_sec=16.0):
    """
    根据 green 通道检测伪迹，并在所有通道上同步切除坏段
    ppg_dict: {"green": ndarray, "red": ndarray, "infrared": ndarray}
    返回清理后的 ppg_dict_cleaned
    """
    min_len = min(len(ppg_dict[ch]) for ch in ppg_dict)
    for ch in ppg_dict:
        ppg_dict[ch] = ppg_dict[ch][:min_len]

    green = ppg_dict["green"]
    n = len(green)
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    good = np.ones(n, dtype=bool)
    d1 = np.diff(green, prepend=green[0])

    # 遍历滑窗检测伪迹
    for start in range(0, n - win, step):
        seg = green[start:start + win]
        seg_d1 = d1[start:start + win]

        mad = np.median(np.abs(seg - np.median(seg)))
        mad_d1 = np.median(np.abs(seg_d1 - np.median(seg_d1)))

        if mad == 0 or mad_d1 == 0:
            continue

        if (
            np.max(np.abs(seg - np.median(seg))) > amp_thresh * mad or
            np.max(np.abs(seg_d1)) > diff_thresh * mad_d1
        ):
            good[start:start + win] = False

    # 检查有效段长度
    if np.sum(good) < min_sec * fs:
        print("！有效段太短，信号跳过")
        return None

    # 拼接所有通道的有效段
    ppg_dict_cleaned = {}
    for ch, sig in ppg_dict.items():
        ppg_dict_cleaned[ch] = sig[good]

    return ppg_dict_cleaned

def ppg_quality_check(ppg_clean, FS_PPG, thresh=0.3):
    try:
        quality = nk.ppg_quality(ppg_clean, sampling_rate=FS_PPG)
        return np.nanmean(quality) >= thresh
    except Exception:
        return False

def ppg_features_summary(ppg, FS_PPG=100):
    feats = {
        "mean_IBI": np.nan,
        "RMSSD": np.nan,

        "PSD_freq_0": np.nan,
        "PSD_freq_1": np.nan,
        "PSD_freq_2": np.nan,
        "PSD_pow_0": np.nan,
        "PSD_pow_1": np.nan,
        "PSD_pow_2": np.nan,

        "Max_Upstroke_Slope_mean": np.nan,
        "Max_Upstroke_Slope_std": np.nan,

        "Pulse_Width_50_mean": np.nan,
        "Pulse_Width_50_std": np.nan,

        "Rise_time_mean": np.nan,
        "Rise_time_std": np.nan,

        "IPA_mean": np.nan,
        "IPA_std": np.nan,

        "AI_mean": np.nan,
        "AI_std": np.nan,
    }

    # ---------- 清洗 ----------
    try:
        ppg_z = (ppg - np.nanmean(ppg)) / (np.nanstd(ppg) + 1e-6)
        ppg_clean = nk.ppg_clean(ppg_z, sampling_rate=FS_PPG, method="elgendi")
    except Exception:
        print('! PPG 清洗失败')
        return None

    if not ppg_quality_check(ppg_clean, FS_PPG):
        print('! PPG 信号质量差')
        return None

    ppg_smooth = savgol_filter(ppg_clean, 7, 3)

    # ---------- 找峰 / 谷 ----------
    try:
        peaks = nk.ppg_findpeaks(ppg_clean, FS_PPG)["PPG_Peaks"]
        troughs = nk.ppg_findpeaks(-ppg_clean, FS_PPG)["PPG_Peaks"]
    except Exception:
        return feats

    if len(peaks) < 3 or len(troughs) < 3:
        return feats

    # ---------- 节律 ----------
    ibi = np.diff(peaks) / FS_PPG
    if len(ibi) > 1:
        feats["mean_IBI"] = np.mean(ibi)
        feats["RMSSD"] = np.sqrt(np.mean(np.diff(ibi) ** 2))

    # ---------- PSD ----------
    try:
        freqs, psd = welch(ppg_clean, fs=FS_PPG, nperseg=min(256, len(ppg_clean)))
        idx = np.argsort(psd)[-3:]
        for i, j in enumerate(idx):
            feats[f"PSD_freq_{i}"] = freqs[j]
            feats[f"PSD_pow_{i}"] = psd[j]
    except Exception:
        pass

    # ---------- 形态 ----------
    slopes, widths, rises, ipa_list, ai_list = [], [], [], [], []
    d1 = np.gradient(ppg_smooth)
    d2 = np.gradient(d1)

    for pk in peaks:
        t0 = troughs[troughs < pk]
        t1 = troughs[troughs > pk]
        if len(t0) == 0 or len(t1) == 0:
            continue

        p0, p1, p2 = t0[-1], pk, t1[0]

        slopes.append(np.max(d1[p0:p1]))
        rises.append((p1 - p0) / FS_PPG)

        baseline = ppg_smooth[p0]
        amp = ppg_smooth[p1] - baseline
        amp50 = baseline + 0.5 * amp

        l = np.where(ppg_smooth[p0:p1] >= amp50)[0]
        r = np.where(ppg_smooth[p1:p2] <= amp50)[0]
        if len(l) > 0 and len(r) > 0:
            widths.append((p1 + r[0] - (p0 + l[0])) / FS_PPG)

        seg2 = d2[p1:p2]
        if len(seg2) > 3:
            notch = p1 + np.argmax(seg2)
            sys = np.trapz(ppg_smooth[p0:p1] - baseline)
            dia = np.trapz(ppg_smooth[p1:p2] - ppg_smooth[p2])
            if dia > 0:
                ipa_list.append(sys / dia)

            ai = (ppg_smooth[notch] - baseline) / (amp + 1e-6)
            ai_list.append(ai)

    def stats(x, name):
        if len(x) > 0:
            feats[f"{name}_mean"] = np.nanmean(x)
            feats[f"{name}_std"] = np.nanstd(x)

    stats(slopes, "Max_Upstroke_Slope")
    stats(widths, "Pulse_Width_50")
    stats(rises, "Rise_time")
    stats(ipa_list, "IPA")
    stats(ai_list, "AI")

    return feats

def extract_ppg_features_single_channel(ppg, win_sec=10, step_sec=2, FS_PPG=100):
    win_len = int(win_sec * FS_PPG)
    step_len = int(step_sec * FS_PPG)

    feats_list = []
    for s in range(0, len(ppg) - win_len + 1, step_len):
        feats = ppg_features_summary(ppg[s:s + win_len], FS_PPG)
        feats_list.append(feats)

    feats_list = [f for f in feats_list if f is not None]
    if not feats_list:
        return {}

    return {k: np.array([f.get(k, np.nan) for f in feats_list], dtype=float) for k in feats_list[0]}

def extract_ppg_features_multichannel(ppg_dict, ppg_feat_manifest, win_sec=10, step_sec=2, FS_PPG=100):
    all_feats = {}
    ch_feats = {}

    # ---------- 单通道 ----------
    for ch, sig in ppg_dict.items():
        feats = extract_ppg_features_single_channel(sig, win_sec, step_sec, FS_PPG)
        ch_feats[ch] = feats
        for k, v in feats.items():
            all_feats[f"{ch}-{k}"] = v

    if not all_feats:
        print('! 该样本未计算出 PPG 特征，返回空数组')
        return {k: np.array([], dtype=float) for k in ppg_feat_manifest}

    return {k: all_feats.get(k, np.array([], dtype=float)) for k in ppg_feat_manifest}

def ppg_compute_sample_features(pair_data, ppg_feat_manifest):
    feats_dict = {}
    for file_name, content in pair_data.items():
        if not file_name.lower().startswith("ppg"):
            continue
        ppg = load_ppg_from_content(content)
        if ppg is None:
            return None
        ppg_cleaned = remove_motion_artifacts(ppg, 100)
        if ppg_cleaned is None:
            return None
        feats_dict = extract_ppg_features_multichannel(ppg_cleaned, ppg_feat_manifest)
    if not any(len(v) > 0 for v in feats_dict.values()):
        print('! PPG 特征层级返回 None')
        return None

    distributions = {}
    vector = []

    for feat in ppg_feat_manifest:
        values = np.asarray(feats_dict.get(feat, []), dtype=float)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            print(f'! 计算出的特征为空-{feat}')
            print('! PPG 特征层级返回 None')
            return None

        elif len(values) < 3:
            print('! 计算出的特征有效值不足')
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            sk = np.nan
            kur = np.nan
            hist, _ = np.histogram(values, bins=20, range=(np.min(values), np.max(values) + 1e-6), density=True)
        else:
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            sk = skew(values)
            kur = kurtosis(values)
            hist, _ = np.histogram(values, bins=20, range=(np.min(values), np.max(values) + 1e-6), density=True)

        distributions[feat] = {"mean": mean, "std": std, "median": median, "skew": sk, "kurtosis": kur, "hist": hist}

        vector.extend([mean, std, median])

    vector = np.asarray(vector, dtype=float)
    return {"vector": vector, "distributions": distributions}

# 公共代码
def compute_eer_safe(y_true, scores):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr

        if len(fpr) == 0:
            return np.nan, None

        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        return eer, thresholds[idx]
    except Exception:
        return np.nan, None

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
    return np.asarray(vec, dtype=float)

def build_reg_finetune_pairs_safe(data_list, feat_manifest, max_neg_users=15):
    try:
        X, y = [], []

        user_groups = defaultdict(list)
        for d in data_list:
            if "externalid" not in d:
                continue
            user_groups[d["externalid"]].append(d)

        uids = list(user_groups.keys())
        if len(uids) < 2:
            print("[Pair] 用户数不足")
            return None, None

        for uid in uids:
            regs = user_groups[uid]
            if len(regs) < 2:
                continue

            # ---------- 正样本 ----------
            for i in range(len(regs)):
                for j in range(i + 1, len(regs)):
                    try:
                        feat = get_pair_features(regs[i], regs[j], feat_manifest)
                        if feat is None:
                            continue
                        X.append(feat)
                        y.append(1)
                    except Exception:
                        continue

            # ---------- 负样本 ----------
            neg_uids = [u for u in uids if u != uid]
            random.shuffle(neg_uids)
            neg_uids = neg_uids[:max_neg_users]

            for neg_uid in neg_uids:
                neg_regs = user_groups.get(neg_uid, [])
                if len(neg_regs) == 0:
                    continue

                for r1 in regs:
                    try:
                        r2 = random.choice(neg_regs)
                        feat = get_pair_features(r1, r2, feat_manifest)
                        if feat is None:
                            continue
                        X.append(feat)
                        y.append(0)
                    except Exception:
                        continue

        if len(X) == 0:
            print("[Pair] 未构造出任何样本")
            return None, None

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # NaN 兜底（你这个点做得本来就对）
        X = np.nan_to_num(X)

        return X, y

    except Exception as e:
        print("[Pair] 构造样本异常，已保护")
        print(" ", repr(e))
        return None, None

def safe_finetune_xgb(base_model, base_scaler, data_list, feat_manifest, min_samples=20):
    try:
        # ---------- pair 构造（安全版） ----------
        X_ft, y_ft = build_reg_finetune_pairs_safe(data_list, feat_manifest)

        if X_ft is None or y_ft is None:
            print("[Fine-tune] pair 构造失败，回退原模型")
            return base_model

        if len(y_ft) < min_samples or len(np.unique(y_ft)) < 2:
            print("[Fine-tune] 样本不足，跳过")
            return base_model

        # ---------- scaler ----------
        try:
            X_ft = base_scaler.transform(X_ft)
        except Exception as e:
            print("[Fine-tune] scaler.transform 失败")
            print(" ", repr(e))
            return base_model

        # ---------- 交叉验证 ----------
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        base_scores = []
        ft_scores = []

        for fold_id, (tr, va) in enumerate(skf.split(X_ft, y_ft)):
            Xtr, Xva = X_ft[tr], X_ft[va]
            ytr, yva = y_ft[tr], y_ft[va]

            # 原模型
            try:
                p0 = base_model.predict_proba(Xva)[:, 1]
                eer0, _ = compute_eer_safe(yva, p0)
                if np.isnan(eer0):
                    print(f"[Fine-tune] Fold {fold_id} base EER 计算失败")
                    return base_model
                base_scores.append(eer0)

            except Exception:
                print(f"[Fine-tune] Fold {fold_id} 原模型预测失败")
                return base_model

            # 微调模型
            try:
                params = base_model.get_params()
                params['n_estimators'] = 10  # 强制限制新增树的数量
                params['learning_rate'] = 0.01  # 极小的步长
                ft_model = xgb.XGBClassifier(**params)
                ft_model.fit(Xtr, ytr, xgb_model=base_model.get_booster(), verbose=False)

                p1 = ft_model.predict_proba(Xva)[:, 1]
                eer1, _ = compute_eer_safe(yva, p1)
                if np.isnan(eer1):
                    print(f"[Fine-tune] Fold {fold_id} ft EER 计算失败")
                    return base_model
                ft_scores.append(eer1)
            except Exception as e:
                print(f"[Fine-tune] Fold {fold_id} 微调失败")
                print(" ", repr(e))
                return base_model

        # ---------- 决策 ----------
        if np.mean(ft_scores) <= np.mean(base_scores) - 1e-4:
            print("[Fine-tune] EER 接受微调模型")
            params = base_model.get_params()
            params['n_estimators'] = 10  # 强制限制新增树的数量
            params['learning_rate'] = 0.01  # 极小的步长
            final_model = xgb.XGBClassifier(**params)
            final_model.fit(X_ft, y_ft, xgb_model=base_model.get_booster(), verbose=False)
            return final_model
        else:
            print("[Fine-tune] EER 性能下降，回退原模型")
            return base_model

    except Exception as e:
        print("[Fine-tune] 全局异常，已保护回退")
        print(" ", repr(e))
        return base_model





def main():
    reader = DataReader('registration_data', 'test_data')
    data = reader.load_data()
    registration = data.registration
    test = data.test

    ecg_model_path = r"ecg_xgb.pkl"
    ecg_scaler_path = r"ecg_scaler.pkl"
    ecg_model_base = joblib.load(ecg_model_path)
    ecg_scaler = joblib.load(ecg_scaler_path)

    ppg_model_path = r"ppg_xgb.pkl"
    ppg_scaler_path = r"ppg_scaler.pkl"
    ppg_model_base = joblib.load(ppg_model_path)
    ppg_scaler = joblib.load(ppg_scaler_path)

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
    ppg_feat_manifest = ['green-AI_mean', 'green-AI_std', 'green-IPA_mean', 'green-IPA_std', 'green-Max_Upstroke_Slope_mean',
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

    # Build data_list from registration (each sample is per data_pair)
    ecg_data_list = []
    for user_id, reg_data in registration.items():
        for pair_id, pair_files in reg_data.items():
            print(f'(registration) Processing ECG {user_id}-{pair_id}...')
            d = ecg_compute_sample_features(pair_files, ecg_feat_manifest)
            if d is None:
                print(f'!! registration 样本 {user_id}-{pair_id} 无有效 ECG 特征，跳过')
                continue
            d["externalid"] = user_id
            ecg_data_list.append(d)

    ppg_data_list = []
    for user_id, reg_data in registration.items():
        for pair_id, pair_files in reg_data.items():
            print(f'(registration) Processing PPG {user_id}-{pair_id}...')
            d = ppg_compute_sample_features(pair_files, ppg_feat_manifest)
            if d is None:
                print(f'!! registration 样本 {user_id}-{pair_id} 无有效 PPG 特征，跳过')
                continue
            d["externalid"] = user_id
            ppg_data_list.append(d)

    # Collect test probes
    ecg_tests = {}
    ppg_tests = {}
    none_tests = []
    for data_pair_id in sorted(test.keys()):
        pair_files = test[data_pair_id]
        print(f'(test) Processing ECG {data_pair_id}...')
        ecg_probe = ecg_compute_sample_features(pair_files, ecg_feat_manifest)
        if ecg_probe is not None:
            ecg_tests[data_pair_id] = ecg_probe
        else:
            print(f'(test) Processing PPG {data_pair_id}...')
            ppg_probe = ppg_compute_sample_features(pair_files, ppg_feat_manifest)
            if ppg_probe is not None:
                ppg_tests[data_pair_id] = ppg_probe
            else:
                none_tests.append(data_pair_id)
    # Fine_tune model
    ecg_model = safe_finetune_xgb(ecg_model_base, ecg_scaler, ecg_data_list, ecg_feat_manifest)

    ppg_model = safe_finetune_xgb(ppg_model_base, ppg_scaler, ppg_data_list, ppg_feat_manifest)

    # For each test data_pair (probe), compute prob for each user
    probs = {}
    users = sorted(registration.keys())

    # First, process ECG computable tests
    for data_pair_id, ecg_probe in ecg_tests.items():
        probs[data_pair_id] = {}
        raw_scores = []
        for user_id in users:
            reg_samples = [d for d in ecg_data_list if d["externalid"] == user_id]
            if len(reg_samples) == 0:
                prob = 0.0
            else:
                pair_probs = []
                for r in reg_samples:
                    feat = get_pair_features(ecg_probe, r, ecg_feat_manifest)
                    feat = np.nan_to_num(feat)
                    feat_scaled = ecg_scaler.transform([feat])
                    p = ecg_model.predict_proba(feat_scaled)[0][1]
                    pair_probs.append(p)

                pair_probs = np.array(pair_probs)
                k = min(3, len(pair_probs))
                prob = np.mean(np.sort(pair_probs)[-k:]) - 0.3 * np.std(np.sort(pair_probs)[-k:])

            raw_scores.append(prob)
            probs[data_pair_id][user_id] = prob

        raw_scores = np.array(raw_scores)
        mu = np.mean(raw_scores)
        sigma = np.std(raw_scores) + 1e-6
        for user_id in users:
            probs[data_pair_id][user_id] = (probs[data_pair_id][user_id] - mu) / sigma

    # Then, process PPG computable tests (where ECG not computable)
    for data_pair_id, ppg_probe in ppg_tests.items():
        probs[data_pair_id] = {}
        raw_scores = []
        for user_id in users:
            reg_samples = [d for d in ppg_data_list if d["externalid"] == user_id]
            if len(reg_samples) == 0:
                prob = 0.0
            else:
                pair_probs = []
                for r in reg_samples:
                    feat = get_pair_features(ppg_probe, r, ppg_feat_manifest)
                    feat = np.nan_to_num(feat)
                    feat_scaled = ppg_scaler.transform([feat])
                    p = ppg_model.predict_proba(feat_scaled)[0][1]
                    pair_probs.append(p)

                pair_probs = np.array(pair_probs)
                k = min(3, len(pair_probs))
                prob = np.mean(np.sort(pair_probs)[-k:]) - 0.3 * np.std(np.sort(pair_probs)[-k:])

            raw_scores.append(prob)
            probs[data_pair_id][user_id] = prob

        raw_scores = np.array(raw_scores)
        mu = np.mean(raw_scores)
        sigma = np.std(raw_scores) + 1e-6
        for user_id in users:
            probs[data_pair_id][user_id] = (probs[data_pair_id][user_id] - mu) / sigma


    # For none computable tests
    for data_pair_id in none_tests:
        print(f'(test) Processing none {data_pair_id} with uniform distribution...')
        probs[data_pair_id] = {}
        for user_id in users:
            probs[data_pair_id][user_id] = 0.0

    # Collect results
    results = []
    for data_pair_id in sorted(test.keys()):
        for user_id in users:
            prob = probs[data_pair_id][user_id]
            results.append([user_id, data_pair_id, prob])
    results.sort(key=lambda x: x[0])

    # Save to parent directory
    reg_path = os.path.abspath('registration_data')
    parent_dir = os.path.dirname(reg_path)
    os.chdir(parent_dir)

    dataoutput = DataOutput(results)
    dataoutput.save_results_to_xlsx()

if __name__ == "__main__":
    main()
