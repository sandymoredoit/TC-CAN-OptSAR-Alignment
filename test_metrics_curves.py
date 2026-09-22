

import os
import random
import numpy as np
import math
import warnings
import tifffile
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch


from model import TemporalStabilityPredictor

warnings.filterwarnings("ignore")


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})


DATASET_DIR = "./processed_patchs_select"


CKPT_PATH = "./checkpoints_SS13V2_T6/epoch68-val_loss3.2424.ckpt"
# CKPT_PATH = "./checkpoints_SS06_T6/epoch67-val_loss1.7068.ckpt"

NUM_TEST_SAMPLES = 500
CROP_SIZE = 224
MAX_SHIFT = 16
TRAIN_SEQ_LEN = 6

MAX_PENALTY_EPE = 40.0
VIZ_OUTPUT_DIR = "./test_visualizations_full_pipeline"

config = {
    'OPT_CHANNELS': 4,
    'SAR_CHANNELS': 2,
    'TRAIN_SEQ_LEN': TRAIN_SEQ_LEN,
}

os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(VIZ_OUTPUT_DIR, "metrics_curves"), exist_ok=True)


def _read_tif_tensor(filepath):
    try:
        img_array = tifffile.imread(filepath).astype(np.float32)
        if img_array.ndim == 3 and img_array.shape[-1] <= 4:
            img_array = img_array.transpose(2, 0, 1)
        elif img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=0)

        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)

        for c in range(img_array.shape[0]):
            channel_data = img_array[c]
            valid_pixels = channel_data[(channel_data > -9000) & (channel_data < 9000) & (channel_data != 0)]
            if len(valid_pixels) > 0:
                vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
                if (vmax - vmin) > 1e-5:
                    channel_data = (channel_data - vmin) / (vmax - vmin)
                    img_array[c] = np.clip(channel_data, 0.0, 1.0)
                else:
                    img_array[c] = np.zeros_like(channel_data)
        return torch.from_numpy(np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0))
    except Exception:
        return None


def _sample_files(files, seq_len):
    if len(files) >= seq_len:
        start = random.randint(0, len(files) - seq_len)
        sampled = files[start: start + seq_len]
    else:
        sampled = files[:]
        while len(sampled) < seq_len:
            sampled.append(random.choice(files))
    sampled.sort(key=lambda x: os.path.basename(x).split('_')[1] if '_' in os.path.basename(x) else x)
    return sampled


def run_evaluation():
    print("正在加载 Deep Learning 模型 (全流水线追踪模式)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalStabilityPredictor.load_from_checkpoint(CKPT_PATH, config=config).eval().to(device)
    print(f"模型加载成功！来源: {CKPT_PATH}")

    print(f"\n正在扫描数据集: {DATASET_DIR} ...")
    valid_grids = []
    for root, dirs, files in os.walk(DATASET_DIR):
        opt_files = [os.path.join(root, f) for f in files if f.startswith('OPT_') and f.endswith(('.tif', '.tiff'))]
        sar_files = [os.path.join(root, f) for f in files if f.startswith('SAR_') and f.endswith(('.tif', '.tiff'))]

        if len(opt_files) >= TRAIN_SEQ_LEN and len(sar_files) >= TRAIN_SEQ_LEN:
            valid_grids.append({"opt": sorted(opt_files), "sar": sorted(sar_files), "id": os.path.basename(root)})

    if not valid_grids:
        raise ValueError(f"未找到有效切片数据！(请确保数据集每个文件夹至少包含 {TRAIN_SEQ_LEN} 个 OPT 和 SAR)")

    valid_grids.sort(key=lambda x: x['id'])
    random.seed(42)
    random.shuffle(valid_grids)
    split_idx = int(len(valid_grids) * 0.9)
    unseen_grids = valid_grids[split_idx:]

    random.seed(999)
    test_grids = random.sample(unseen_grids, min(NUM_TEST_SAMPLES, len(unseen_grids)))
    print(f"选定 {len(test_grids)} 对样本进行评测 (Full Tracking Pipeline)...\n")

    results_db = []

    with torch.no_grad():
        for i, grid_info in enumerate(tqdm(test_grids, desc="Run: Full Tracking Pipeline")):
            full_opt = grid_info["opt"]
            full_sar = grid_info["sar"]

            opt1_seq = torch.stack([_read_tif_tensor(f) for f in _sample_files(full_opt, TRAIN_SEQ_LEN)], dim=0)
            opt2_seq = torch.stack([_read_tif_tensor(f) for f in _sample_files(full_opt, TRAIN_SEQ_LEN)], dim=0)
            sar1_seq = torch.stack([_read_tif_tensor(f) for f in _sample_files(full_sar, TRAIN_SEQ_LEN)], dim=0)
            sar2_seq = torch.stack([_read_tif_tensor(f) for f in _sample_files(full_sar, TRAIN_SEQ_LEN)], dim=0)

            if any(t is None for t in [opt1_seq, opt2_seq, sar1_seq, sar2_seq]): continue

            T_len, _, H, W = opt1_seq.shape
            max_safe_y, max_safe_x = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
            if max_safe_y <= 0 or max_safe_x <= 0: continue

            opt_top, opt_left = max_safe_y, max_safe_x
            bound_y, bound_x = min(MAX_SHIFT, max_safe_y), min(MAX_SHIFT, max_safe_x)


            dx1, dy1 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)
            dx2, dy2 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)
            dx3, dy3 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)

            o1_in = opt1_seq[:, :, opt_top:opt_top + CROP_SIZE, opt_left:opt_left + CROP_SIZE].unsqueeze(0).to(device)
            o2_in = opt2_seq[
                :, :, opt_top + dy3:opt_top + dy3 + CROP_SIZE, opt_left + dx3:opt_left + dx3 + CROP_SIZE].unsqueeze(
                0).to(device)
            s1_in = sar1_seq[
                :, :, opt_top + dy1:opt_top + dy1 + CROP_SIZE, opt_left + dx1:opt_left + dx1 + CROP_SIZE].unsqueeze(
                0).to(device)
            s2_in = sar2_seq[
                :, :, opt_top + dy2:opt_top + dy2 + CROP_SIZE, opt_left + dx2:opt_left + dx2 + CROP_SIZE].unsqueeze(
                0).to(device)

            try:
                res1 = model.forward_and_compute_loss(o1_in, s1_in)
                res2 = model.forward_and_compute_loss(o1_in, s2_in)
                res3 = model.forward_and_compute_loss(o2_in, s1_in)

                p1_x, p1_y = res1['shift_x'].item(), res1['shift_y'].item()
                p2_x, p2_y = res2['shift_x'].item(), res2['shift_y'].item()
                p3_x, p3_y = res3['shift_x'].item(), res3['shift_y'].item()

                trk_dx = res1['opt_track_dx'][0].cpu().numpy()
                trk_dy = res1['opt_track_dy'][0].cpu().numpy()

                jitter_mags = np.sqrt(trk_dx ** 2 + trk_dy ** 2)
                max_internal_jitter = float(np.max(jitter_mags))

                abs1 = min(math.hypot(p1_x - dx1, p1_y - dy1), MAX_PENALTY_EPE)
                sar_cycle = min(math.hypot((p2_x - p1_x) - (dx2 - dx1), (p2_y - p1_y) - (dy2 - dy1)), MAX_PENALTY_EPE)
                opt_cycle = min(math.hypot((p3_x - p1_x) - (-dx3), (p3_y - p1_y) - (-dy3)), MAX_PENALTY_EPE)

                init_shift_mag = (math.hypot(dx1, dy1) + math.hypot(dx2, dy2)) / 2.0
                diff_label = "Small (0-5px)" if init_shift_mag <= 5 else (
                    "Medium (6-10px)" if init_shift_mag <= 10 else "Large (>10px)")

                results_db.append({
                    'abs': abs1,
                    'cycle': sar_cycle,
                    'opt_cycle': opt_cycle,
                    'diff': diff_label,
                    'init_shift': init_shift_mag,
                    'max_internal_jitter': max_internal_jitter
                })

            except Exception as e:
                import traceback
                print(f"在样本 {i} 发生错误:")
                traceback.print_exc()

    total = len(results_db)
    if total == 0: return

    print(f"\n正在生成图表至 {VIZ_OUTPUT_DIR}/metrics_curves ...")
    df = pd.DataFrame(results_db)

    df_valid = df[df['abs'] < MAX_PENALTY_EPE]

    metrics_dir = os.path.join(VIZ_OUTPUT_DIR, "metrics_curves")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    fig.suptitle('Error Distribution and Internal Jitter Analysis (Full Pipeline)', fontsize=20, fontweight='bold')

    errors = np.sort(df_valid['abs'].values)
    cdf = np.arange(1, len(errors) + 1) / len(df) * 100
    axes[0, 0].plot(errors, cdf, linewidth=2.5, color='#D32F2F', label='Full Pipeline Inference')
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title(f'(a) Cumulative Error (PCK) [T={TRAIN_SEQ_LEN}]', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('End-Point Error (EPE) Threshold [px]', fontsize=12)
    axes[0, 0].set_ylabel('Success Rate [%]', fontsize=12)
    succ_2px_val = (np.sum(df['abs'].values < 2.0) / len(df)) * 100
    axes[0, 0].axvline(x=2.0, color='blue', linestyle='--', linewidth=1.5,
                       label=f'Threshold < 2px: {succ_2px_val:.1f}%')
    axes[0, 0].legend(loc='lower right', fontsize=12)
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    sns.scatterplot(data=df_valid, x='max_internal_jitter', y='abs', color='#009688', alpha=0.7, s=50, edgecolor='k',
                    ax=axes[0, 1])
    if len(df_valid) > 10:
        z = np.polyfit(df_valid['max_internal_jitter'], df_valid['abs'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df_valid['max_internal_jitter'], p(df_valid['max_internal_jitter']), "r--", linewidth=2,
                        label="Trend")
    axes[0, 1].set_title('(b) Impact of Internal OPT Jitter on Cross-modal EPE', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Detected Max Internal Jitter (Frame-to-Frame) [px]', fontsize=12)
    axes[0, 1].set_ylabel('Final Cross-modal EPE [px]', fontsize=12)
    axes[0, 1].axhline(y=2.0, color='red', linestyle=':', alpha=0.8, linewidth=1.5, label='2.0px Standard')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)

    sns.kdeplot(df_valid['cycle'], color='#388E3C', fill=True, alpha=0.5, linewidth=2,
                label=f'SAR Cycle EPE (Mean: {df_valid["cycle"].mean():.2f}px)', ax=axes[1, 0])
    sns.kdeplot(df_valid['opt_cycle'], color='#FF9800', fill=True, alpha=0.5, linewidth=2,
                label=f'OPT Cycle EPE (Mean: {df_valid["opt_cycle"].mean():.2f}px)', ax=axes[1, 0])
    axes[1, 0].set_title('(c) Intra-Modal Cycle Consistency: SAR/OPT', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Cycle End-Point Error (EPE) [px]', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_xlim(0, max(5, df_valid['cycle'].max() + 0.5))
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)

    sns.histplot(df_valid['max_internal_jitter'], bins=30, kde=True, color='#673AB7', edgecolor='black', alpha=0.6,
                 ax=axes[1, 1])
    axes[1, 1].set_title('(d) Distribution of Detected Internal Data Jitter', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Detected Max Internal Jitter [px]', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].axvline(df_valid['max_internal_jitter'].mean(), color='red', linestyle='dashed', linewidth=1.5,
                       label=f'Mean Detected Jitter: {df_valid["max_internal_jitter"].mean():.2f}px')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    combined_path = os.path.join(metrics_dir, "00_Combined_Metrics_Panel.png")
    fig.savefig(combined_path)
    plt.close(fig)

    print(f"成功生成 2x2 拼接分析大图，已保存至: {combined_path}\n")
    # =========================================================================

    mean_abs = df['abs'].mean()
    mean_sar_cyc = df['cycle'].mean()
    mean_opt_cyc = df['opt_cycle'].mean()
    mean_internal_jitter = df['max_internal_jitter'].mean()

    model_name = os.path.basename(CKPT_PATH).split('-')[0]

    print("=" * 140)
    print(
        f"| Model (Full Pipe): {model_name:<15} | EPE (Abs): {mean_abs:<5.2f}px | SAR Cycle: {mean_sar_cyc:<5.2f}px | OPT Cycle: {mean_opt_cyc:<5.2f}px | PCK<2px: {succ_2px_val:>5.1f}% | Detected Mean Jitter: {mean_internal_jitter:.1f}px |")
    print("=" * 140)


if __name__ == "__main__":
    run_evaluation()

