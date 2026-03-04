import os
import logging
import json
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, welch


#############################################
# Logger
#############################################
def setup_logger(log_path):
    logger = logging.getLogger("PPG_PROCESS")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def is_ppg_already_processed(txt_path):
    base_dir = os.path.dirname(txt_path)
    feat_dir = os.path.join(base_dir, "PPG_Features")

    return os.path.exists(feat_dir)

def remove_motion_artifacts(ppg_dict, fs, win_sec=2.0, step_sec=1.0,
                                          amp_thresh=10.0, diff_thresh=10.0, min_sec=10.0):
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
        raise ValueError("！有效段太短，信号跳过")

    # 拼接所有通道的有效段
    ppg_dict_cleaned = {}
    for ch, sig in ppg_dict.items():
        ppg_dict_cleaned[ch] = sig[good]

    return ppg_dict_cleaned

def load_ppg_from_txt(txt_path):
    ppg = {"green": [], "red": [], "infrared": []}
    line_count = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
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

    if line_count < 1000:
        raise ValueError(f"数据量过少: {txt_path} 仅有 {line_count} 行，低于阈值 {1000}")

    return {k: np.array(v, dtype=float) for k, v in ppg.items()}

def extract_ppg_identity_features(ppg_raw, FS_PPG=100):
    feats = {}
    ppg_zscore = (ppg_raw - np.mean(ppg_raw)) / (np.std(ppg_raw) + 1e-6)

    # 显式清洗
    ppg_clean = nk.ppg_clean(ppg_zscore, sampling_rate=FS_PPG, method='elgendi')
    ppg_smooth = savgol_filter(ppg_clean, 7, 3)  # 减小窗口以保留特征细节

    try:
        peaks_info = nk.ppg_findpeaks(ppg_clean, sampling_rate=FS_PPG, method='elgendi')
        peaks = peaks_info["PPG_Peaks"]
        trough_info = nk.ppg_findpeaks(-ppg_clean, sampling_rate=FS_PPG, method='elgendi')
        troughs = trough_info["PPG_Peaks"]
    except Exception as e:
        print('！ppg_findpeaks失败')
        print(e)
        return None

    # 过滤点数不足的情况
    if len(peaks) < 3 or len(troughs) < 3:
        print('！检测出的峰值不够')
        return None

    # ---------- 2. 节律特征 ----------
    ibi = np.diff(peaks) / FS_PPG
    feats["mean_IBI"] = np.mean(ibi)
    feats["RMSSD"] = np.sqrt(np.mean(np.diff(ibi) ** 2))

    # ---------- 3. 频域特征 (PSD) 优化建议 ----------
    try:
        nperseg = min(256, len(ppg_clean))
        freqs, psd = welch(ppg_clean, fs=FS_PPG, nperseg=nperseg)
        idx_top3 = np.argsort(psd)[-3:]
        for i, idx in enumerate(idx_top3):
            feats[f"PSD_freq_{i}"] = freqs[idx]
            feats[f"PSD_pow_{i}"] = psd[idx]
    except:
        pass

    # ---------- 4. 逐周期形态学特征 (Trough-Peak-Trough) ----------
    rise_times, widths_50, slopes, ipa_list, ai_list = [], [], [], [], []
    d1 = np.gradient(ppg_smooth)
    d2 = np.gradient(d1)

    # 确保 troughs 和 peaks 对齐：每个周期定义为 p0(谷) -> p1(峰) -> p2(下个谷)
    for i in range(len(peaks)):
        # 寻找该峰值前后的谷值
        t_before = troughs[troughs < peaks[i]]
        t_after = troughs[troughs > peaks[i]]

        if len(t_before) == 0 or len(t_after) == 0:
            continue

        p0 = t_before[-1]  # 最近的前谷
        p1 = peaks[i]  # 当前峰
        p2 = t_after[0]  # 最近的后谷

        # A. 斜率与上升时间
        slopes.append(np.max(d1[p0:p1]))
        rise_times.append((p1 - p0) / FS_PPG)

        # B. 完整 Pulse Width 50% (考虑左右两侧)
        baseline = ppg_smooth[p0]
        peak_amp = ppg_smooth[p1] - baseline
        amp_50 = baseline + 0.5 * peak_amp
        # 搜索左侧
        left_idx = np.where(ppg_smooth[p0:p1] >= amp_50)[0]
        # 搜索右侧
        right_idx = np.where(ppg_smooth[p1:p2] <= amp_50)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            t_start = p0 + left_idx[0]
            t_end = p1 + right_idx[0]
            widths_50.append((t_end - t_start) / FS_PPG)

        # C. Notch 定位与 AI/IPA
        # 限制在波峰到下个波谷之间寻找切迹点
        search_range = ppg_smooth[p1:p2]
        d2_range = d2[p1:p2]

        if len(d2_range) > 3:
            # 切迹点定位：二阶导数的第一个局部最大值点（反映斜率变化剧减点）
            notch_rel = np.argmax(d2_range)
            notch = p1 + notch_rel

            # IPA (Area Ratio)
            sys_area = np.trapz(ppg_smooth[p0:p1] - ppg_smooth[p0])
            dia_area = np.trapz(ppg_smooth[p1:p2] - ppg_smooth[p2])
            if dia_area > 0:
                ipa_list.append(sys_area / dia_area)

            # AI (Augmentation Index) 修正：收集列表
            ai = (ppg_smooth[notch] - ppg_smooth[p0]) / (ppg_smooth[p1] - ppg_smooth[p0] + 1e-6)
            ai_list.append(ai)

    # ---------- 5. 统计特征汇总与标准差补充 ----------
    # 辅助函数：安全均值/标差
    def cal_stats(data_list, name, target_dict):
        if len(data_list) > 0:
            target_dict[f"{name}_mean"] = np.nanmean(data_list)
            target_dict[f"{name}_std"] = np.nanstd(data_list)
        else:
            target_dict[f"{name}_mean"] = np.nan
            target_dict[f"{name}_std"] = np.nan

    cal_stats(slopes, "Max_Upstroke_Slope", feats)
    cal_stats(widths_50, "Pulse_Width_50", feats)
    cal_stats(rise_times, "Rise_time", feats)
    cal_stats(ipa_list, "IPA", feats)
    cal_stats(ai_list, "AI", feats)

    return feats

def extract_ppg_features_sliding(ppg, win_sec=10, step_sec=2, FS_PPG=100):
    win_len = int(win_sec * FS_PPG)
    step_len = int(step_sec * FS_PPG)

    feats_list = []

    for start in range(0, len(ppg) - win_len + 1, step_len):
        ppg_win = ppg[start:start + win_len]
        feats = extract_ppg_identity_features(ppg_win, FS_PPG)
        if feats is not None:
            feats_list.append(feats)

    if len(feats_list) == 0:
        return None

    keys = feats_list[0].keys()
    feats_dict = {k: np.array([f[k] for f in feats_list]) for k in keys}

    return feats_dict

def save_ppg_features(ppg_dict, txt_path):
    all_channel_feats = {}
    for channel, ppg in ppg_dict.items():
        feats_dict = extract_ppg_features_sliding(ppg)

        if feats_dict is None:
            print('！特征列表为空')
            return
        all_channel_feats[channel] = feats_dict

        base_dir = os.path.dirname(txt_path)
        feat_dir = os.path.join(base_dir, "PPG_Features")
        os.makedirs(feat_dir, exist_ok=True)

        for feat_name, values in feats_dict.items():
            save_path = os.path.join(
                feat_dir, f"{channel}-{feat_name}.npy"
            )
            np.save(save_path, values)





root_dir = r"data"

FS_PPG = 100

log_path = os.path.join(root_dir, "ppg_processing.log")
logger = setup_logger(log_path)

data_dir = Path(root_dir)

for external_dir in tqdm([d for d in data_dir.iterdir() if d.is_dir()]):
    for group_dir in [d for d in external_dir.iterdir() if d.is_dir()]:

        files = list(group_dir.iterdir())

        ppg_txt_files = [
            f for f in files
            if f.is_file()
            and f.suffix == ".txt"
            and f.name.split("_")[0].lower() == "ppg"
        ]

        if not ppg_txt_files:
            logger.warning(f"目录跳过：{group_dir} 中未发现 PPG txt 文件")
            continue

        if len(ppg_txt_files) != 1:
            logger.error(f"目录错误：{group_dir} 中发现 {len(ppg_txt_files)} 个 PPG 文件: {ppg_txt_files}")
            continue

        txt_path = ppg_txt_files[0]
        logger.info(f"Processing {txt_path}")

        if is_ppg_already_processed(txt_path):
            logger.info(f"Skip (already processed): {txt_path}")
            continue

        try:
            ppg_dict = load_ppg_from_txt(txt_path)
            ppg_dict_cleaned = remove_motion_artifacts(ppg_dict, FS_PPG)
            save_ppg_features(ppg_dict_cleaned, txt_path)
            logger.info(f"Success {txt_path}")
        except Exception:
            logger.exception(f"Failed {txt_path}")