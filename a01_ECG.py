import os
import logging
import json
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from pathlib import Path
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean



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

def setup_logger(log_path):
    logger = logging.getLogger("ECG_PROCESS")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def is_ecg_already_processed(txt_path):
    base_dir = os.path.dirname(txt_path)
    feat_dir = os.path.join(base_dir, "ECG_Features")

    return os.path.exists(feat_dir)

def load_ecg_from_txt(txt_path):
    ecg_values = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            data = json.loads(line)  # 每行是一个 JSON 对象
            ecg_values.extend([v["value"] for v in data["voltage"]])

    return np.array(ecg_values, dtype=float)

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
        return None  # 如果清洗失败，直接扔掉

    if ecg_clean is None:
        return None

    quality_pass = ecg_quality_check(ecg_clean, FS_ECG, thresh=0.3)
    if not quality_pass:
        print('! 信号质量差')
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

def extract_ecg_features_sliding_parallel(ecg, win_sec=10, step_sec=2, FS_ECG=500, n_jobs=4):
    win_len = int(win_sec * FS_ECG)
    step_len = int(step_sec * FS_ECG)
    starts = list(range(0, len(ecg) - win_len + 1, step_len))
    def compute_window(start):
        ecg_win = ecg[start:start + win_len]
        feats = ecg_features_summary(ecg_win, FS_ECG)
        return feats  # 可能为 None
    feats_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(compute_window)(s) for s in starts)
    # 过滤掉 None（质量差的窗口）
    feats_list = [f for f in feats_list if f is not None]
    if not feats_list:
        print('! 该样本未计算出特征，返回空数组')
        # 返回空数组 dict，如果无有效窗口
        return {k: np.array([], dtype=float) for k in ecg_features_summary(np.array([]), FS_ECG).keys() or []}
    feats_dict = {k: np.array([f.get(k, np.nan) for f in feats_list], dtype=float) for k in feats_list[0].keys()}
    return feats_dict

def save_ecg_features_per_feature(ecg, txt_path):
    feats_dict = extract_ecg_features_sliding_parallel(ecg)
    base_dir = os.path.dirname(txt_path)
    feat_dir = os.path.join(base_dir, "ECG_Features")
    os.makedirs(feat_dir, exist_ok=True)
    for feat_name, values in feats_dict.items():
        # 检查是否全为 NaN，如果是，则不存储该 npy
        if np.all(np.isnan(values)):
            print(f'! 特征 {feat_name} 全为 NaN，不存储')
            continue
        save_path = os.path.join(
            feat_dir, f"{feat_name}.npy"
        )
        np.save(save_path, values)






root_dir = r"data"

FS_ECG = 500


log_path = os.path.join(root_dir, "ecg_processing.log")
logger = setup_logger(log_path)

data_dir = Path(root_dir)

for external_dir in tqdm([d for d in data_dir.iterdir() if d.is_dir()]):
    for group_dir in [d for d in external_dir.iterdir() if d.is_dir()]:

        files = list(group_dir.iterdir())

        ecg_txt_files = [
            f for f in files
            if f.is_file()
            and f.suffix == ".txt"
            and f.name.split("_")[0].lower() == "ecg"
        ]

        if not ecg_txt_files:
            logger.warning(f"目录跳过：{group_dir} 中未发现 ECG txt 文件")
            continue

        if len(ecg_txt_files) != 1:
            logger.error(
                f"目录错误：{group_dir} 中发现 {len(ecg_txt_files)} 个 ECG 文件: {ecg_txt_files}"
            )
            continue

        txt_path = ecg_txt_files[0]
        logger.info(f"Processing {txt_path}")

        if is_ecg_already_processed(txt_path):
            logger.info(f"Skip (already processed): {txt_path}")
            continue

        try:
            ecg = load_ecg_from_txt(txt_path)
            save_ecg_features_per_feature(ecg, txt_path)
            logger.info(f"Success: {txt_path}")

        except Exception:
            logger.exception(f"Failed: {txt_path}")

