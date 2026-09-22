

import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
import timm




class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, feat_dim=64, backbone_name='hrnet_w18'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            out_indices=(0, 1), in_chans=in_channels
        )
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, in_channels, 224, 224))
            ch_0, ch_1 = [out.shape[1] for out in dummy]

        self.proj = nn.Conv2d(ch_0 + ch_1, feat_dim, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feat_dim, affine=True)

    def forward(self, img_seq):
        B, T, C, H, W = img_seq.shape
        img_flat = img_seq.view(B * T, C, H, W)

        feats = self.backbone(img_flat)
        f0 = feats[0]
        f1 = F.interpolate(feats[1], size=f0.shape[2:], mode='bilinear', align_corners=True)

        feat = self.norm(self.proj(torch.cat([f0, f1], dim=1)))

        return feat.view(B, T, -1, f0.shape[2], f0.shape[3])


class StructuralAnchorExtractor(nn.Module):
    def __init__(self, in_channels=1, cloud_thresh=0.75):
        super().__init__()
        self.in_channels = in_channels
        self.cloud_thresh = cloud_thresh

    @torch.no_grad()
    def forward(self, img_seq):
        img_seq = img_seq.float()
        B, T, C, H, W = img_seq.shape

        intensity = img_seq[:, :, :self.in_channels].mean(dim=2)

        temporal_var = intensity.var(dim=1, keepdim=True)
        temporal_mask = torch.exp(-temporal_var * 15.0)

        mean_intensity = intensity.mean(dim=1, keepdim=True)
        brightness_mask = torch.clamp(1.0 - (mean_intensity - self.cloud_thresh) * 4.0, min=0.1, max=1.0)

        intensity_flat = intensity.view(B * T, 1, H, W)
        dy = intensity_flat[:, :, 1:, :] - intensity_flat[:, :, :-1, :]
        dx = intensity_flat[:, :, :, 1:] - intensity_flat[:, :, :, :-1]

        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

        grad_mag = grad_mag.view(B, T, H, W)
        grad_mag = grad_mag * temporal_mask * brightness_mask

        border = 16
        mask = torch.zeros_like(grad_mag)
        mask[:, :, border:-border, border:-border] = 1.0
        grad_mag = grad_mag * mask

        valid_area = grad_mag[:, :, border:-border, border:-border].contiguous().view(B * T, -1)
        mean_val = valid_area.mean(dim=1).view(B * T, 1, 1, 1)
        std_val = valid_area.std(dim=1).view(B * T, 1, 1, 1)

        grad_mag_flat = grad_mag.view(B * T, 1, H, W)
        score = torch.relu(grad_mag_flat - mean_val) / (std_val + 1e-5)
        score = torch.clamp(score, 0.0, 1.0)

        mask_flat = mask.view(B * T, 1, H, W)
        score = score * mask_flat

        return score.view(B, T, H, W)


class TemporalStabilityPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat_scale = 4.0
        self.patch_radius = 5

        self.opt_extractor = SpatioTemporalFeatureExtractor(in_channels=config.get('OPT_CHANNELS', 4), feat_dim=64)
        self.sar_extractor = SpatioTemporalFeatureExtractor(in_channels=config.get('SAR_CHANNELS', 2), feat_dim=64)
        self.opt_anchor_finder = StructuralAnchorExtractor(in_channels=config.get('OPT_CHANNELS', 4))

        self.ce_loss = nn.CrossEntropyLoss()

        for param in self.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def compute_phase_correlation(self, ref_img, tgt_img):
        B, H, W = ref_img.shape
        device = ref_img.device

        window_y = torch.hann_window(H, device=device).view(1, H, 1)
        window_x = torch.hann_window(W, device=device).view(1, 1, W)
        window = window_y * window_x

        ref_w = ref_img * window
        tgt_w = tgt_img * window

        F_ref = torch.fft.fft2(ref_w)
        F_tgt = torch.fft.fft2(tgt_w)

        F_ref_conj = torch.conj(F_ref)
        cross_power = F_tgt * F_ref_conj
        cross_power = cross_power / (torch.abs(cross_power) + 1e-8)

        r = torch.fft.ifft2(cross_power)
        r = torch.real(r)

        r_flat = r.view(B, -1)
        max_idx = torch.argmax(r_flat, dim=1)

        y_max = max_idx // W
        x_max = max_idx % W

        dy = torch.where(y_max > H // 2, y_max - H, y_max)
        dx = torch.where(x_max > W // 2, x_max - W, x_max)

        peak_val = r_flat.max(dim=1).values
        mean_val = r_flat.mean(dim=1)
        std_val = r_flat.std(dim=1)
        conf = (peak_val - mean_val) / (std_val + 1e-6)

        return dx, dy, conf

    def get_hard_anchor(self, sae_map):
        B, H, W = sae_map.shape
        safe_margin = self.patch_radius + 6

        valid_sae = sae_map.clone()
        valid_sae[:, :safe_margin, :] = -1
        valid_sae[:, -safe_margin:, :] = -1
        valid_sae[:, :, :safe_margin] = -1
        valid_sae[:, :, -safe_margin:] = -1

        flat_idx = valid_sae.view(B, -1).argmax(dim=1)
        y = flat_idx // W
        x = flat_idx % W
        return x, y


    def extract_patch(self, feat, x, y, radius):
        B, C, H, W = feat.shape

        px = torch.clamp(torch.round(x).long(), radius, W - radius - 1)
        py = torch.clamp(torch.round(y).long(), radius, H - radius - 1)


        batch_idx = torch.arange(B, device=feat.device).view(B, 1, 1, 1)
        channel_idx = torch.arange(C, device=feat.device).view(1, C, 1, 1)

        dy = torch.arange(-radius, radius + 1, device=feat.device).view(1, 1, 2 * radius + 1, 1)
        dx = torch.arange(-radius, radius + 1, device=feat.device).view(1, 1, 1, 2 * radius + 1)

        grid_y = py.view(B, 1, 1, 1) + dy
        grid_x = px.view(B, 1, 1, 1) + dx


        return feat[batch_idx, channel_idx, grid_y, grid_x]

    def forward_and_compute_loss(self, opt_seq, sar_seq, gt_shift_x=None, gt_shift_y=None, force_anchor_feat=None,
                                 force_best_t=None):
        B, T, _, H, W = opt_seq.shape
        device = opt_seq.device
        b_idx = torch.arange(B, device=device)

        opt_feat_seq = F.normalize(self.opt_extractor(opt_seq), p=2, dim=2)
        sar_feat_seq = F.normalize(self.sar_extractor(sar_seq), p=2, dim=2)

        C_f, H_f, W_f = opt_feat_seq.shape[2:]
        dynamic_scale_y = H / H_f
        dynamic_scale_x = W / W_f

        if force_anchor_feat is not None:
            if force_best_t is not None:
                best_t = force_best_t
            else:
                best_t = torch.zeros(B, dtype=torch.long, device=device)

            opt_ref_feat = opt_feat_seq[b_idx, best_t]

            with torch.no_grad():
                opt_gray_ref = opt_seq[b_idx, best_t, :].mean(dim=1)
                opt_sae = None

            ref_template = force_anchor_feat
            match_0 = F.conv2d(
                opt_ref_feat.view(1, B * C_f, H_f, W_f),
                ref_template,
                groups=B
            ).squeeze(0)
            W_m0 = match_0.shape[2]
            flat_idx_0 = match_0.view(B, -1).argmax(dim=1)
            ref_x = (flat_idx_0 % W_m0) + self.patch_radius
            ref_y = (flat_idx_0 // W_m0) + self.patch_radius

        else:
            with torch.no_grad():
                opt_sae_all = self.opt_anchor_finder(opt_seq)

                opt_sae_mean = opt_sae_all.mean(dim=1)
                opt_sae_f = F.interpolate(opt_sae_mean.unsqueeze(1), size=(H_f, W_f),
                                          mode='bilinear', align_corners=True).squeeze(1)
                ref_x, ref_y = self.get_hard_anchor(opt_sae_f)

                frame_scores = opt_sae_all.view(B, T, -1).sum(dim=2)  # shape: [B, T]
                best_t = frame_scores.argmax(dim=1)  # shape: [B]

                opt_sae = opt_sae_all[b_idx, best_t]

                opt_gray_ref = opt_seq[b_idx, best_t, :].mean(dim=1)

            opt_ref_feat = opt_feat_seq[b_idx, best_t]  # [B, C_f, H_f, W_f]
            ref_template = self.extract_patch(opt_ref_feat, ref_x, ref_y, self.patch_radius)

        aligned_opt_patches = []
        cycle_weights = []
        opt_track_dx, opt_track_dy = [], []
        patch_area = (2 * self.patch_radius + 1) ** 2

        for t in range(0, T):
            search_space = opt_feat_seq[:, t]

            with torch.no_grad():
                tgt_gray = opt_seq[:, t, :].mean(dim=1)
                dx_img, dy_img, fft_conf = self.compute_phase_correlation(opt_gray_ref, tgt_gray)

                dx_feat = torch.round(dx_img.float() / dynamic_scale_x)
                dy_feat = torch.round(dy_img.float() / dynamic_scale_y)

                valid_fft = (fft_conf > 3.0).float()
                dx_feat = dx_feat * valid_fft
                dy_feat = dy_feat * valid_fft

                fallback_x = torch.clamp(ref_x + dx_feat, self.patch_radius, W_f - self.patch_radius - 1).long()
                fallback_y = torch.clamp(ref_y + dy_feat, self.patch_radius, H_f - self.patch_radius - 1).long()

                match_map = F.conv2d(search_space.reshape(1, B * C_f, H_f, W_f), ref_template,
                                     groups=B).squeeze(0)
                m_flat = match_map.reshape(B, -1)
                peak_val, peak_idx = m_flat.max(dim=1)

                conf = (peak_val - m_flat.mean(dim=1)) / (m_flat.std(dim=1) + 1e-6)

                W_m = match_map.shape[2]
                tx_raw = ((peak_idx % W_m) + self.patch_radius).float()
                ty_raw = ((peak_idx // W_m) + self.patch_radius).float()

                cx = torch.clamp(tx_raw.long() - self.patch_radius, 1, match_map.shape[2] - 2)
                cy = torch.clamp(ty_raw.long() - self.patch_radius, 1, match_map.shape[1] - 2)

                c_val = match_map[b_idx, cy, cx]
                l_val = match_map[b_idx, cy, cx - 1]
                r_val = match_map[b_idx, cy, cx + 1]
                u_val = match_map[b_idx, cy - 1, cx]
                d_val = match_map[b_idx, cy + 1, cx]

                denom_x = 2 * (l_val - 2 * c_val + r_val)
                denom_y = 2 * (u_val - 2 * c_val + d_val)

                denom_x = torch.where(denom_x.abs() < 1e-6, torch.sign(denom_x + 1e-12) * 1e-6, denom_x)
                denom_y = torch.where(denom_y.abs() < 1e-6, torch.sign(denom_y + 1e-12) * 1e-6, denom_y)

                is_true_peak_x = ((c_val > l_val) & (c_val > r_val)).float()
                is_true_peak_y = ((c_val > u_val) & (c_val > d_val)).float()

                dx_sub = torch.clamp((l_val - r_val) / denom_x, -0.5, 0.5) * is_true_peak_x
                dy_sub = torch.clamp((u_val - d_val) / denom_y, -0.5, 0.5) * is_true_peak_y

                valid_mask = ((tx_raw > self.patch_radius) & (tx_raw < W_f - self.patch_radius - 1) &
                              (ty_raw > self.patch_radius) & (ty_raw < H_f - self.patch_radius - 1)).float()

                tx_refined = tx_raw + dx_sub * valid_mask
                ty_refined = ty_raw + dy_sub * valid_mask

                temp_back_template = self.extract_patch(search_space, tx_refined, ty_refined, self.patch_radius)
                back_match_map = F.conv2d(opt_ref_feat.reshape(1, B * C_f, H_f, W_f), temp_back_template,
                                          groups=B).squeeze(0)
                b_flat = back_match_map.reshape(B, -1)
                b_peak_idx = b_flat.argmax(dim=1)

                bx_raw = ((b_peak_idx % W_m) + self.patch_radius).float()
                by_raw = ((b_peak_idx // W_m) + self.patch_radius).float()

                cycle_error = torch.sqrt((bx_raw - ref_x.float()) ** 2 + (by_raw - ref_y.float()) ** 2)

                cycle_is_valid = (cycle_error < 1.5).float()
                is_reliable = ((conf > 3.5) & (cycle_is_valid == 1.0)).float()

                track_x = is_reliable * tx_refined + (1 - is_reliable) * fallback_x.float()
                track_y = is_reliable * ty_refined + (1 - is_reliable) * fallback_y.float()

                frame_weight = torch.exp(-0.5 * cycle_error)
                cycle_weights.append(frame_weight)

                opt_track_dx.append((track_x - ref_x.float()) * dynamic_scale_x)
                opt_track_dy.append((track_y - ref_y.float()) * dynamic_scale_y)

            aligned_opt_patches.append(self.extract_patch(search_space, track_x, track_y, self.patch_radius))

        stacked_patches = torch.stack(aligned_opt_patches, dim=0)
        stacked_weights = torch.stack(cycle_weights, dim=0).view(T, B, 1, 1, 1)

        norm_weights = stacked_weights / (stacked_weights.sum(dim=0, keepdim=True) + 1e-6)

        opt_template_fused = (stacked_patches * norm_weights).sum(dim=0)
        opt_template_fused = F.normalize(opt_template_fused, p=2, dim=1)

        sar_img = F.normalize(sar_feat_seq.mean(dim=1), p=2, dim=1)
        sar_img_flat = sar_img.view(1, B * C_f, H_f, W_f)

        corr_surface = F.conv2d(sar_img_flat, opt_template_fused, groups=B).squeeze(0)
        corr_surface = (corr_surface / patch_area).unsqueeze(1)

        temperature = 0.05
        logits = corr_surface.view(B, -1) / temperature
        loss_nce = torch.tensor(0.0, device=device)

        if gt_shift_x is not None and gt_shift_y is not None:
            dx_feat_gt = torch.round(gt_shift_x.float() / dynamic_scale_x).long()
            dy_feat_gt = torch.round(gt_shift_y.float() / dynamic_scale_y).long()

            target_y = ref_y - dy_feat_gt - self.patch_radius
            target_x = ref_x - dx_feat_gt - self.patch_radius
            target_y = torch.clamp(target_y, 0, corr_surface.shape[2] - 1)
            target_x = torch.clamp(target_x, 0, corr_surface.shape[3] - 1)
            target_idx = target_y * corr_surface.shape[3] + target_x
            loss_nce = self.ce_loss(logits, target_idx)

        with torch.no_grad():
            B_c, _, H_c, W_c = corr_surface.shape

            pred_idx = logits.argmax(dim=1)
            py = pred_idx // W_c
            px = pred_idx % W_c

            pred_y_base = (py + self.patch_radius).float()
            pred_x_base = (px + self.patch_radius).float()

            cy_c = torch.clamp(py, 1, H_c - 2)
            cx_c = torch.clamp(px, 1, W_c - 2)
            val_3d = logits.view(B_c, H_c, W_c)

            c_val_c = val_3d[b_idx, cy_c, cx_c]
            l_val_c = val_3d[b_idx, cy_c, cx_c - 1]
            r_val_c = val_3d[b_idx, cy_c, cx_c + 1]
            u_val_c = val_3d[b_idx, cy_c - 1, cx_c]
            d_val_c = val_3d[b_idx, cy_c + 1, cx_c]

            denom_x_c = 2 * (l_val_c - 2 * c_val_c + r_val_c)
            denom_y_c = 2 * (u_val_c - 2 * c_val_c + d_val_c)
            denom_x_c = torch.where(denom_x_c.abs() < 1e-6, torch.sign(denom_x_c + 1e-12) * 1e-6, denom_x_c)
            denom_y_c = torch.where(denom_y_c.abs() < 1e-6, torch.sign(denom_y_c + 1e-12) * 1e-6, denom_y_c)

            is_true_peak_x_c = ((c_val_c > l_val_c) & (c_val_c > r_val_c)).float()
            is_true_peak_y_c = ((c_val_c > u_val_c) & (c_val_c > d_val_c)).float()

            dx_sub_c = torch.clamp((l_val_c - r_val_c) / denom_x_c, -0.5, 0.5) * is_true_peak_x_c
            dy_sub_c = torch.clamp((u_val_c - d_val_c) / denom_y_c, -0.5, 0.5) * is_true_peak_y_c

            valid_mask_c = ((px > 0) & (px < W_c - 1) & (py > 0) & (py < H_c - 1)).float()

            pred_x = pred_x_base + dx_sub_c * valid_mask_c
            pred_y = pred_y_base + dy_sub_c * valid_mask_c

            opt_mean_feat = F.normalize(opt_feat_seq.mean(dim=1), p=2, dim=1)
            opt_mean_flat = opt_mean_feat.view(1, B * C_f, H_f, W_f)
            opt_mean_corr = F.conv2d(opt_mean_flat, opt_template_fused, groups=B).squeeze(0)
            opt_mean_corr = (opt_mean_corr / patch_area).unsqueeze(1)

            opt_m_idx = opt_mean_corr.view(B, -1).argmax(dim=1)
            m_py = opt_m_idx // W_c
            m_px = opt_m_idx % W_c

            opt_m_y_base = (m_py + self.patch_radius).float()
            opt_m_x_base = (m_px + self.patch_radius).float()

            cy_m = torch.clamp(m_py, 1, H_c - 2)
            cx_m = torch.clamp(m_px, 1, W_c - 2)
            val_3d_m = opt_mean_corr.view(B_c, H_c, W_c)

            c_val_m = val_3d_m[b_idx, cy_m, cx_m]
            l_val_m = val_3d_m[b_idx, cy_m, cx_m - 1]
            r_val_m = val_3d_m[b_idx, cy_m, cx_m + 1]
            u_val_m = val_3d_m[b_idx, cy_m - 1, cx_m]
            d_val_m = val_3d_m[b_idx, cy_m + 1, cx_m]

            denom_x_m = 2 * (l_val_m - 2 * c_val_m + r_val_m)
            denom_y_m = 2 * (u_val_m - 2 * c_val_m + d_val_m)
            denom_x_m = torch.where(denom_x_m.abs() < 1e-6, torch.sign(denom_x_m + 1e-12) * 1e-6, denom_x_m)
            denom_y_m = torch.where(denom_y_m.abs() < 1e-6, torch.sign(denom_y_m + 1e-12) * 1e-6, denom_y_m)

            is_true_peak_x_m = ((c_val_m > l_val_m) & (c_val_m > r_val_m)).float()
            is_true_peak_y_m = ((c_val_m > u_val_m) & (c_val_m > d_val_m)).float()

            dx_sub_m = torch.clamp((l_val_m - r_val_m) / denom_x_m, -0.5, 0.5) * is_true_peak_x_m
            dy_sub_m = torch.clamp((u_val_m - d_val_m) / denom_y_m, -0.5, 0.5) * is_true_peak_y_m

            valid_mask_m = ((m_px > 0) & (m_px < W_c - 1) & (m_py > 0) & (m_py < H_c - 1)).float()

            opt_mean_x = opt_m_x_base + dx_sub_m * valid_mask_m
            opt_mean_y = opt_m_y_base + dy_sub_m * valid_mask_m

            shift_x = (opt_mean_x - pred_x) * dynamic_scale_x
            shift_y = (opt_mean_y - pred_y) * dynamic_scale_y

        return {
            'loss': loss_nce,
            'metrics': {'loss_nce': loss_nce.detach() if loss_nce.requires_grad else loss_nce},
            'stable_opt_sae': opt_sae if opt_sae is not None else torch.zeros(B, H, W, device=device),
            'best_t': best_t,
            'raw_opt_x': ref_x,
            'raw_opt_y': ref_y,
            'opt_x': ref_x.float() * dynamic_scale_x,
            'opt_y': ref_y.float() * dynamic_scale_y,
            'shift_x': shift_x,
            'shift_y': shift_y,
            'corr_surface': corr_surface,
            'anchor_feature': ref_template.detach(),
            'opt_track_dx': torch.stack(opt_track_dx, dim=1),
            'opt_track_dy': torch.stack(opt_track_dy, dim=1)
        }

    def training_step(self, batch, batch_idx):
        res = self.forward_and_compute_loss(
            batch['opt_crop'], batch['sar_crop'],
            batch['shift_x'], batch['shift_y']
        )
        self.log('train/Loss_NCE', res['loss'], prog_bar=True)
        return res['loss']

    def validation_step(self, batch, batch_idx):
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        o1, o2 = batch['opt1_crop'], batch['opt2_crop']
        s1, s2 = batch['sar1_crop'], batch['sar2_crop']
        B, T, C_opt, H_img, W_img = o1.shape
        device = o1.device

        dx1, dy1 = batch['dx1'].float(), batch['dy1'].float()
        dx2, dy2 = batch['dx2'].float(), batch['dy2'].float()
        dx3, dy3 = batch['dx3'].float(), batch['dy3'].float()

        res1 = self.forward_and_compute_loss(o1, s1, dx1, dy1)
        res2 = self.forward_and_compute_loss(o1, s2)
        res3 = self.forward_and_compute_loss(o2, s1)

        p1_x, p1_y = res1['shift_x'], res1['shift_y']
        p2_x, p2_y = res2['shift_x'], res2['shift_y']
        p3_x, p3_y = res3['shift_x'], res3['shift_y']

        self.log('val_loss', res1['loss'], sync_dist=True)

        abs_epe = torch.sqrt((p1_x - dx1) ** 2 + (p1_y - dy1) ** 2)

        sar_cyc = torch.sqrt(((p2_x - p1_x) - (dx2 - dx1)) ** 2 + ((p2_y - p1_y) - (dy2 - dy1)) ** 2)

        opt_cyc = torch.sqrt(((p3_x - p1_x) - (-dx3)) ** 2 + ((p3_y - p1_y) - (-dy3)) ** 2)

        succ_2px = (abs_epe < 2.0).float() * 100.0

        self.log('val_metrics/1_Abs_EPE_px', abs_epe.mean(), sync_dist=True)
        self.log('val_metrics/2_SAR_Cycle_px', sar_cyc.mean(), sync_dist=True)
        self.log('val_metrics/3_OPT_Cycle_px', opt_cyc.mean(), sync_dist=True)
        self.log('val_metrics/4_Success_rate_under_2px', succ_2px.mean(), sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero and getattr(self, "logger", None) is not None:
            import wandb
            num_viz = min(4, B)
            wandb_images = []

            def norm01(tensor):
                t = tensor.cpu().float()
                return (t - t.min()) / (t.max() - t.min() + 1e-6)

            for i in range(num_viz):
                ox, oy = res1['opt_x'][i].item(), res1['opt_y'][i].item()
                dx_p, dy_p = p1_x[i].item(), p1_y[i].item()
                sx_p, sy_p = ox - dx_p, oy - dy_p
                t_val = res1['best_t'][i].item()

                opt_rgb = norm01(o1[i, t_val, :3]).numpy().transpose(1, 2, 0)
                sar_np = norm01(s1[i, 0, 0]).numpy()
                sar_rgb = np.stack([sar_np] * 3, axis=-1)
                sae_map = norm01(res1['stable_opt_sae'][i]).numpy()

                fig = Figure(figsize=(18, 6), dpi=120)
                canvas = FigureCanvasAgg(fig)
                ax = fig.subplots(1, 3)

                ax[0].imshow(opt_rgb)
                ax[0].plot(ox, oy, 'r+', markersize=15, markeredgewidth=3)
                ax[0].set_title(f"OPT Frame {t_val} (Best) & Anchor")

                ax[1].imshow(sar_rgb)
                ax[1].plot(ox, oy, 'r+', markersize=12, markeredgewidth=2, alpha=0.5)
                ax[1].plot(sx_p, sy_p, 'g+', markersize=15, markeredgewidth=3)
                ax[1].annotate('', xy=(sx_p, sy_p), xytext=(ox, oy),
                               arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.5))
                ax[1].set_title(f"SAR Match | EPE: {abs_epe[i]:.1f}px")

                ax[2].imshow(sae_map, cmap='magma')
                ax[2].set_title("Structural Anchor (SAE) Map")

                for sub_ax in ax.flatten(): sub_ax.axis('off')

                fig.tight_layout(pad=1.5)
                canvas.draw()
                img_arr = np.asarray(canvas.buffer_rgba())

                caption = f"B{i} | EPE:{abs_epe[i]:.1f} | SCyc:{sar_cyc[i]:.1f} | OCyc:{opt_cyc[i]:.1f}"
                wandb_images.append(wandb.Image(img_arr, caption=caption))
                fig.clear();
                del fig, canvas, ax

            self.logger.experiment.log({"Validation/Cycle_Consistency_Panel": wandb_images})

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.get('BASE_LR', 3e-4), weight_decay=0.01)
        T_max = self.trainer.max_epochs if self.trainer.max_epochs else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
