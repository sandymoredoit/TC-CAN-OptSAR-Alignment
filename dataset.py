import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import warnings
import tifffile

warnings.filterwarnings("ignore")


class ProxyAlignDataset(Dataset):
    def __init__(self, dataset_dir, is_train=True, train_seq_len=6, crop_size=224, max_shift_px=20, seed=42):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.train_seq_len = train_seq_len
        self.crop_size = crop_size
        self.max_shift_px = max_shift_px

        valid_grids = []
        print(f"正在扫描目录: {dataset_dir} ...")
        for root, dirs, files in os.walk(dataset_dir):
            opt_files = [os.path.join(root, f) for f in files if f.startswith('OPT_') and f.endswith(('.tif', '.tiff'))]
            sar_files = [os.path.join(root, f) for f in files if f.startswith('SAR_') and f.endswith(('.tif', '.tiff'))]

            if len(opt_files) >= 1 and len(sar_files) >= 1:
                valid_grids.append({
                    "opt": sorted(opt_files),
                    "sar": sorted(sar_files),
                    "id": os.path.basename(root)
                })

        if len(valid_grids) == 0:
            raise ValueError(f"未找到有效数据")

        valid_grids.sort(key=lambda x: x['id'])

        random.seed(seed)
        random.shuffle(valid_grids)
        split_idx = int(len(valid_grids) * 0.9)
        if split_idx == 0: split_idx = 1
        if split_idx == len(valid_grids): split_idx = max(1, len(valid_grids) - 1)

        self.grids = valid_grids[:split_idx] if self.is_train else valid_grids[split_idx:]
        print(f"{'训练集' if is_train else '验证集'} 加载完毕，共 {len(self.grids)} 个网格。 (Max Shift: {self.max_shift_px}px)")

    def __len__(self):
        return len(self.grids) * (10 if self.is_train else 2)

    def _read_tif(self, filepath):
        try:
            img_array = tifffile.imread(filepath).astype(np.float32)

            if img_array.ndim == 3 and img_array.shape[-1] <= 4:
                img_array = img_array.transpose(2, 0, 1)

            img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)

            for c in range(img_array.shape[0]):
                channel_data = img_array[c]
                valid_pixels = channel_data[(channel_data > -9000) & (channel_data < 9000) & (channel_data != 0)]

                num_valid = len(valid_pixels)
                if num_valid > 1000:
                    stride = num_valid // 1000
                    sub_pixels = valid_pixels[::stride][:1000]
                    vmin = np.percentile(sub_pixels, 2)
                    vmax = np.percentile(sub_pixels, 98)
                elif num_valid > 0:
                    vmin = np.percentile(valid_pixels, 2)
                    vmax = np.percentile(valid_pixels, 98)
                else:
                    vmin, vmax = 0.0, 0.0

                if (vmax - vmin) < 1e-5:
                    img_array[c] = np.zeros_like(channel_data)
                else:
                    channel_data = (channel_data - vmin) / (vmax - vmin)
                    img_array[c] = np.clip(channel_data, 0.0, 1.0)

            img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.from_numpy(img_array)

        except Exception as e:
            print(f"读取失败: {filepath}, Error: {e}")
            return None

    def _sample_files(self, files, active_len):
        if len(files) >= active_len:
            start = random.randint(0, len(files) - active_len)
            sampled = files[start: start + active_len]
        else:
            sampled = files[:]
            while len(sampled) < active_len:
                sampled.append(random.choice(files))

        sampled.sort(key=lambda x: os.path.basename(x).split('_')[1] if '_' in os.path.basename(x) else x)
        return sampled

    def __getitem__(self, idx):
        real_idx = idx % len(self.grids)
        grid_info = self.grids[real_idx]

        active_len = self.train_seq_len

        opt_paths = self._sample_files(grid_info["opt"], active_len)
        sar_paths = self._sample_files(grid_info["sar"], active_len)

        opt_seq = torch.stack([self._read_tif(f) for f in opt_paths], dim=0)
        sar_seq = torch.stack([self._read_tif(f) for f in sar_paths], dim=0)

        T, _, H, W = opt_seq.shape

        if self.is_train:
            if random.random() < 0.5:
                opt_seq = torch.flip(opt_seq, dims=[3])
                sar_seq = torch.flip(sar_seq, dims=[3])
            if random.random() < 0.5:
                opt_seq = torch.flip(opt_seq, dims=[2])
                sar_seq = torch.flip(sar_seq, dims=[2])

            rot_k = random.randint(0, 3)
            if rot_k > 0:
                opt_seq = torch.rot90(opt_seq, k=rot_k, dims=[2, 3])
                sar_seq = torch.rot90(sar_seq, k=rot_k, dims=[2, 3])

            max_offset_y = H - self.crop_size
            max_offset_x = W - self.crop_size

            if max_offset_y <= 0 or max_offset_x <= 0:
                opt_y, opt_x, sar_y, sar_x = 0, 0, 0, 0
                actual_shift_y, actual_shift_x = 0, 0
            else:
                physical_limit_y = max_offset_y // 2
                physical_limit_x = max_offset_x // 2
                bound_y = min(self.max_shift_px, physical_limit_y)
                bound_x = min(self.max_shift_px, physical_limit_x)

                shift_y = random.randint(-bound_y, bound_y)
                shift_x = random.randint(-bound_x, bound_x)

                min_opt_y = max(0, -shift_y)
                max_opt_y = min(max_offset_y, max_offset_y - shift_y)
                min_opt_x = max(0, -shift_x)
                max_opt_x = min(max_offset_x, max_offset_x - shift_x)

                opt_y = random.randint(min_opt_y, max_opt_y)
                opt_x = random.randint(min_opt_x, max_opt_x)

                sar_y = opt_y + shift_y
                sar_x = opt_x + shift_x

                actual_shift_y, actual_shift_x = shift_y, shift_x

            opt_seq = opt_seq[:, :, opt_y:opt_y + self.crop_size, opt_x:opt_x + self.crop_size]
            sar_seq = sar_seq[:, :, sar_y:sar_y + self.crop_size, sar_x:sar_x + self.crop_size]

            if random.random() < 0.5:
                b_factor = torch.empty(T, 1, 1, 1).uniform_(-0.2, 0.2)
                c_factor = torch.empty(T, 1, 1, 1).uniform_(0.8, 1.2)
                opt_seq = (opt_seq - 0.5) * c_factor + 0.5 + b_factor
                opt_seq = torch.clamp(opt_seq, 0.0, 1.0)

            if random.random() < 0.5:
                noise = torch.randn_like(sar_seq) * 0.1
                sar_seq = sar_seq * (1.0 + noise)
                sar_seq = torch.clamp(sar_seq, 0.0, 1.0)

            if opt_seq.shape[2] < self.crop_size or opt_seq.shape[3] < self.crop_size:
                pad_y = max(0, self.crop_size - opt_seq.shape[2])
                pad_x = max(0, self.crop_size - opt_seq.shape[3])
                opt_seq = torch.nn.functional.pad(opt_seq, (0, pad_x, 0, pad_y))
                sar_seq = torch.nn.functional.pad(sar_seq, (0, pad_x, 0, pad_y))

            return {
                'grid_id': grid_info['id'],
                'opt_crop': opt_seq.contiguous(),
                'sar_crop': sar_seq.contiguous(),
                'shift_x': actual_shift_x,
                'shift_y': actual_shift_y
            }

        else:
            max_offset_y = (H - self.crop_size) // 2
            max_offset_x = (W - self.crop_size) // 2

            if max_offset_y > 0 and max_offset_x > 0:
                bound_y = min(self.max_shift_px, max_offset_y)
                bound_x = min(self.max_shift_px, max_offset_x)
                cy, cx = H // 2, W // 2
                base_top, base_left = cy - self.crop_size // 2, cx - self.crop_size // 2
                dx1, dy1 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)
                dx2, dy2 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)
                dx3, dy3 = random.randint(-bound_x, bound_x), random.randint(-bound_y, bound_y)
                o1_crop = opt_seq[:, :, base_top:base_top + self.crop_size, base_left:base_left + self.crop_size]
                o2_crop = opt_seq[
                    :, :, base_top + dy3:base_top + dy3 + self.crop_size, base_left + dx3:base_left + dx3 + self.crop_size]
                s1_crop = sar_seq[
                    :, :, base_top + dy1:base_top + dy1 + self.crop_size, base_left + dx1:base_left + dx1 + self.crop_size]
                s2_crop = sar_seq[
                    :, :, base_top + dy2:base_top + dy2 + self.crop_size, base_left + dx2:base_left + dx2 + self.crop_size]
            else:
                dx1, dy1, dx2, dy2, dx3, dy3 = 0, 0, 0, 0, 0, 0
                base_top, base_left = 0, 0
                o1_crop = opt_seq[:, :, base_top:base_top + self.crop_size, base_left:base_left + self.crop_size]
                o2_crop = o1_crop
                s1_crop = sar_seq[:, :, base_top:base_top + self.crop_size, base_left:base_left + self.crop_size]
                s2_crop = s1_crop

            if o1_crop.shape[2] < self.crop_size or o1_crop.shape[3] < self.crop_size:
                pad_y = max(0, self.crop_size - o1_crop.shape[2])
                pad_x = max(0, self.crop_size - o1_crop.shape[3])
                o1_crop = torch.nn.functional.pad(o1_crop, (0, pad_x, 0, pad_y))
                o2_crop = torch.nn.functional.pad(o2_crop, (0, pad_x, 0, pad_y))
                s1_crop = torch.nn.functional.pad(s1_crop, (0, pad_x, 0, pad_y))
                s2_crop = torch.nn.functional.pad(s2_crop, (0, pad_x, 0, pad_y))

            return {
                'grid_id': grid_info['id'],
                'opt1_crop': o1_crop.contiguous(),
                'opt2_crop': o2_crop.contiguous(),
                'sar1_crop': s1_crop.contiguous(),
                'sar2_crop': s2_crop.contiguous(),
                'dx1': dx1, 'dy1': dy1,
                'dx2': dx2, 'dy2': dy2,
                'dx3': dx3, 'dy3': dy3,
            }


class ProxyAlignDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        max_shift = self.config.get('MAX_SHIFT_PX', 20)

        if stage == 'fit' or stage is None:
            self.train_ds = ProxyAlignDataset(
                dataset_dir=self.config['DATASET_DIR'],
                is_train=True,
                train_seq_len=self.config['TRAIN_SEQ_LEN'],
                crop_size=224,
                max_shift_px=max_shift,
                seed=self.config.get('SEED', 42)
            )
            self.val_ds = ProxyAlignDataset(
                dataset_dir=self.config['DATASET_DIR'],
                is_train=False,
                train_seq_len=self.config['TRAIN_SEQ_LEN'],
                crop_size=224,
                max_shift_px=max_shift,
                seed=self.config.get('SEED', 42)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.config.get('DATA_WORKER_NUM', 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            num_workers=self.config.get('DATA_WORKER_NUM', 4),
            pin_memory=True,
            persistent_workers=True
        )