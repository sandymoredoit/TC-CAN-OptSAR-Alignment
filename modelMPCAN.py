

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
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels

    @torch.no_grad()
    def forward(self, img_seq):
        img_seq = img_seq.float()
        B, T, C, H, W = img_seq.shape

        intensity = img_seq[:, :, :self.in_channels].mean(dim=2, keepdim=True).view(B * T, 1, H, W)

        dy = intensity[:, :, 1:, :] - intensity[:, :, :-1, :]
        dx = intensity[:, :, :, 1:] - intensity[:, :, :, :-1]

        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

        border = 16
        mask = torch.zeros_like(grad_mag)
        mask[:, :, border:-border, border:-border] = 1.0
        grad_mag = grad_mag * mask

        valid_area = grad_mag[:, :, border:-border, border:-border].contiguous().view(B * T, -1)
        mean_val = valid_area.mean(dim=1).view(B * T, 1, 1, 1)
        std_val = valid_area.std(dim=1).view(B * T, 1, 1, 1)

        score = torch.relu(grad_mag - mean_val) / (std_val + 1e-5)
        score = torch.clamp(score, 0.0, 1.0)
        score = score * mask

        return score.view(B, T, H, W).mean(dim=1)


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
        patches = []
        for b in range(B):
            px, py = x[b].item(), y[b].item()
            # 边界安全保护
            px = max(radius, min(W - radius - 1, px))
            py = max(radius, min(H - radius - 1, py))
            patch = feat[b:b + 1, :, py - radius:py + radius + 1, px - radius:px + radius + 1]
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def forward_and_compute_loss(self, opt_seq, sar_seq, gt_shift_x=None, gt_shift_y=None, force_anchor_feat=None):
        B, T, _, H, W = opt_seq.shape
        device = opt_seq.device

        opt_feat_seq = F.normalize(self.opt_extractor(opt_seq), p=2, dim=2)
        sar_feat_seq = F.normalize(self.sar_extractor(sar_seq), p=2, dim=2)

        C_f, H_f, W_f = opt_feat_seq.shape[2:]
        dynamic_scale_y = H / H_f
        dynamic_scale_x = W / W_f

        if force_anchor_feat is not None:
            ref_template = force_anchor_feat
            match_0 = F.conv2d(
                opt_feat_seq.mean(dim=1).view(1, B * C_f, H_f, W_f),
                ref_template,
                groups=B
            ).squeeze(0)
            W_m0 = match_0.shape[2]
            flat_idx_0 = match_0.view(B, -1).argmax(dim=1)
            ref_x = (flat_idx_0 % W_m0) + self.patch_radius
            ref_y = (flat_idx_0 // W_m0) + self.patch_radius
            opt_sae = None
        else:
            opt_sae = self.opt_anchor_finder(opt_seq)
            opt_sae_f = F.interpolate(opt_sae.unsqueeze(1), size=(H_f, W_f),
                                      mode='bilinear', align_corners=True).squeeze(1)
            ref_x, ref_y = self.get_hard_anchor(opt_sae_f)

            opt_feat_avg = opt_feat_seq.mean(dim=1)
            ref_template = self.extract_patch(opt_feat_avg, ref_x, ref_y, self.patch_radius)

        aligned_opt_patches = []
        opt_track_dx, opt_track_dy = [], []
        patch_area = (2 * self.patch_radius + 1) ** 2

        for t in range(0, T):
            search_space = opt_feat_seq[:, t]
            match_map = F.conv2d(search_space.reshape(1, B * C_f, H_f, W_f), ref_template, groups=B).squeeze(0)

            m_flat = match_map.reshape(B, -1)
            peak_val, peak_idx = m_flat.max(dim=1)
            conf = (peak_val - m_flat.mean(dim=1)) / (m_flat.std(dim=1) + 1e-6)
            is_reliable = (conf > 3.5).float()

            W_m = match_map.shape[2]
            tx_raw = (peak_idx % W_m) + self.patch_radius
            ty_raw = (peak_idx // W_m) + self.patch_radius
            track_x = (is_reliable * tx_raw + (1 - is_reliable) * ref_x).long()
            track_y = (is_reliable * ty_raw + (1 - is_reliable) * ref_y).long()
            opt_track_dx.append((track_x - ref_x).float() * dynamic_scale_x)
            opt_track_dy.append((track_y - ref_y).float() * dynamic_scale_y)
            aligned_opt_patches.append(self.extract_patch(search_space, track_x, track_y, self.patch_radius))


        opt_template_fused = torch.stack(aligned_opt_patches, dim=0).mean(dim=0)

        sar_img = sar_feat_seq.mean(dim=1)
        sar_img_flat = sar_img.view(1, B * C_f, H_f, W_f)

        corr_surface = F.conv2d(sar_img_flat, opt_template_fused, groups=B).squeeze(0)
        corr_surface = (corr_surface / patch_area).unsqueeze(1)

        temperature = 0.05
        logits = corr_surface.view(B, -1) / temperature
        loss_nce = torch.tensor(0.0, device=device)

        if gt_shift_x is not None and gt_shift_y is not None:
            dx_feat = torch.round(gt_shift_x.float() / dynamic_scale_x).long()
            dy_feat = torch.round(gt_shift_y.float() / dynamic_scale_y).long()

            target_y = ref_y - dy_feat - self.patch_radius
            target_x = ref_x - dx_feat - self.patch_radius
            target_y = torch.clamp(target_y, 0, corr_surface.shape[2] - 1)
            target_x = torch.clamp(target_x, 0, corr_surface.shape[3] - 1)
            target_idx = target_y * corr_surface.shape[3] + target_x
            loss_nce = self.ce_loss(logits, target_idx)

        with torch.no_grad():
            B_c, C_c, H_c, W_c = corr_surface.shape
            pred_idx = logits.argmax(dim=1)
            py = pred_idx // W_c
            px = pred_idx % W_c

            pred_y = (py + self.patch_radius).float()
            pred_x = (px + self.patch_radius).float()

            for b in range(B_c):
                x, y = px[b].item(), py[b].item()
                if 0 < x < W_c - 1 and 0 < y < H_c - 1:
                    val = logits[b].view(H_c, W_c)
                    c = val[y, x]
                    l, r = val[y, x - 1], val[y, x + 1]
                    u, d = val[y - 1, x], val[y + 1, x]
                    dx = (l - r) / (2 * (l - 2 * c + r) - 1e-6)
                    dy = (u - d) / (2 * (u - 2 * c + d) - 1e-6)
                    pred_x[b] += torch.clamp(dx, -0.5, 0.5)
                    pred_y[b] += torch.clamp(dy, -0.5, 0.5)

            shift_x = (ref_x - pred_x).float() * dynamic_scale_x
            shift_y = (ref_y - pred_y).float() * dynamic_scale_y

        return {
            'loss': loss_nce,
            'metrics': {'loss_nce': loss_nce.detach() if loss_nce.requires_grad else loss_nce},
            'stable_opt_sae': opt_sae if opt_sae is not None else torch.zeros(B, H, W, device=device),
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

        opt_seq, sar_seq = batch['opt_crop'], batch['sar_crop']
        B, _, _, H_img, W_img = opt_seq.shape
        device = opt_seq.device

        gt_x = batch['shift_x'].float()
        gt_y = batch['shift_y'].float()

        res = self.forward_and_compute_loss(opt_seq, sar_seq, gt_x, gt_y)
        self.log('val_loss', res['loss'], sync_dist=True)

        pred_x, pred_y = res['shift_x'], res['shift_y']
        epe = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        self.log('val/1_EPE_Error_px', epe.mean(), sync_dist=True)

        corr_surface = res['corr_surface']
        corr_flat = corr_surface.view(B, -1)
        peak_val, _ = corr_flat.max(dim=1)
        mean_val = corr_flat.mean(dim=1)
        std_val = corr_flat.std(dim=1)
        cpz_scores = (peak_val - mean_val) / (std_val + 1e-6)
        self.log('val/2_Match_CPZ_Score', cpz_scores.mean(), sync_dist=True)

        if res['opt_track_dx'] is not None:
            track_mag = torch.sqrt(res['opt_track_dx'] ** 2 + res['opt_track_dy'] ** 2)
            self.log('val/3_Tracker_Jitter_Mean_px', track_mag.mean(), sync_dist=True)
            self.log('val/3_Tracker_Jitter_Max_px', track_mag.max(dim=1)[0].mean(), sync_dist=True)

        tx = -pred_x / (W_img / 2.0)
        ty = -pred_y / (H_img / 2.0)
        matrix = torch.zeros((B, 2, 3), device=device)
        matrix[:, 0, 0] = 1.0
        matrix[:, 1, 1] = 1.0
        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        grid = F.affine_grid(matrix, [B, 1, H_img, W_img], align_corners=True)

        mean_opt_gray = opt_seq[:, 0, :3].mean(dim=1)
        mean_sar_gray = sar_seq[:, :, 0].mean(dim=1)

        warped_sar_gray = F.grid_sample(
            mean_sar_gray.unsqueeze(1), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(1)

        def compute_nmi(img1, img2, bins=32):
            i1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-6)
            i2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6)
            i1 = torch.clamp((i1 * bins).long(), 0, bins - 1).view(-1)
            i2 = torch.clamp((i2 * bins).long(), 0, bins - 1).view(-1)
            hist_2d = torch.bincount(i1 * bins + i2, minlength=bins ** 2).float()
            hist_2d = hist_2d.view(bins, bins) / (hist_2d.sum() + 1e-8)
            p_i1, p_i2 = hist_2d.sum(dim=1), hist_2d.sum(dim=0)
            H1 = -torch.sum(p_i1 * torch.log2(p_i1 + 1e-8))
            H2 = -torch.sum(p_i2 * torch.log2(p_i2 + 1e-8))
            H12 = -torch.sum(hist_2d * torch.log2(hist_2d + 1e-8))
            return (H1 + H2) / (H12 + 1e-8)

        safe_margin = 20
        safe_opt = mean_opt_gray[:, safe_margin:-safe_margin, safe_margin:-safe_margin]
        safe_sar = warped_sar_gray[:, safe_margin:-safe_margin, safe_margin:-safe_margin]

        nmi_scores = [compute_nmi(safe_opt[b], safe_sar[b]) for b in range(B)]
        self.log('val/4_Aligned_NMI', sum(nmi_scores) / B, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero and getattr(self, "logger", None) is not None:
            import wandb
            num_viz = min(4, B)
            wandb_images = []

            step = 32
            yy_c, xx_c = np.mgrid[0:H_img, 0:W_img]
            checker_mask = np.expand_dims(((xx_c // step) + (yy_c // step)) % 2 == 0, axis=-1)

            def norm01(tensor):
                t = tensor.cpu().float()
                return (t - t.min()) / (t.max() - t.min() + 1e-6)

            for i in range(num_viz):
                ox, oy = res['opt_x'][i].item(), res['opt_y'][i].item()
                dx_p, dy_p = pred_x[i].item(), pred_y[i].item()
                sx_p, sy_p = ox - dx_p, oy - dy_p

                opt_rgb = norm01(opt_seq[i, 0, :3]).numpy().transpose(1, 2, 0)
                sar_np = norm01(mean_sar_gray[i]).numpy()
                sar_rgb = np.stack([sar_np] * 3, axis=-1)
                warped_sar_np = norm01(warped_sar_gray[i]).numpy()
                warped_sar_rgb = np.stack([warped_sar_np] * 3, axis=-1)

                opt_gray_np = opt_rgb.mean(axis=-1)
                sae_map = norm01(res['stable_opt_sae'][i]).numpy()

                checker_before = opt_rgb * checker_mask + sar_rgb * (1 - checker_mask)
                checker_after = opt_rgb * checker_mask + warped_sar_rgb * (1 - checker_mask)

                overlay_before = np.zeros_like(opt_rgb)
                overlay_before[..., 0], overlay_before[..., 1], overlay_before[..., 2] = opt_gray_np, sar_np, sar_np

                overlay_after = np.zeros_like(opt_rgb)
                overlay_after[..., 0], overlay_after[..., 1], overlay_after[
                    ..., 2] = opt_gray_np, warped_sar_np, warped_sar_np

                fig = Figure(figsize=(24, 12), dpi=120)
                canvas = FigureCanvasAgg(fig)
                ax = fig.subplots(2, 4)

                ax[0, 0].imshow(opt_rgb)
                ax[0, 0].plot(ox, oy, 'r+', markersize=15, markeredgewidth=3)
                ax[0, 0].set_title("1. OPT_0 & Extract Anchor")
                ax[0, 0].set_xlim(0, W_img)
                ax[0, 0].set_ylim(H_img, 0)

                ax[0, 1].imshow(sar_rgb)
                ax[0, 1].plot(ox, oy, 'r+', markersize=12, markeredgewidth=2, alpha=0.5)
                ax[0, 1].plot(sx_p, sy_p, 'g+', markersize=15, markeredgewidth=3)
                ax[0, 1].annotate('', xy=(sx_p, sy_p), xytext=(ox, oy),
                                  arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.5, mutation_scale=20))
                ax[0, 1].set_title(f"2. SAR Match | EPE: {epe[i]:.1f}px")
                ax[0, 1].set_xlim(0, W_img)
                ax[0, 1].set_ylim(H_img, 0)

                ax[0, 2].imshow(sae_map, cmap='magma')
                ax[0, 2].set_title("3. Structural Anchor (SAE) Map")

                corr_viz = corr_surface[i, 0].float().cpu().numpy()
                ax[0, 3].imshow(corr_viz, cmap='jet', interpolation='nearest')
                ax[0, 3].set_title(f"4. Match Energy (CPZ: {cpz_scores[i]:.1f})")

                ax[1, 0].imshow(checker_before)
                ax[1, 0].set_title("5. Checkerboard BEFORE")

                ax[1, 1].imshow(checker_after)
                ax[1, 1].set_title("6. Checkerboard AFTER")

                ax[1, 2].imshow(overlay_before)
                ax[1, 2].set_title("7. Overlay BEFORE (R:OPT, GB:SAR)")

                ax[1, 3].imshow(overlay_after)
                ax[1, 3].set_title(f"8. Overlay AFTER (NMI: {nmi_scores[i]:.2f})")

                for sub_ax in ax.flatten():
                    sub_ax.axis('off')

                fig.tight_layout(pad=1.5)

                canvas.draw()
                img_arr = np.asarray(canvas.buffer_rgba())
                wandb_images.append(wandb.Image(
                    img_arr,
                    caption=f"B{i} | EPE: {epe[i]:.2f}px | NMI: {nmi_scores[i]:.2f} | CPZ: {cpz_scores[i]:.2f}"
                ))
                fig.clear()
                del fig, canvas, ax

            self.logger.experiment.log({"Validation/Matching_Analysis_Panel": wandb_images})

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.get('BASE_LR', 3e-4), weight_decay=0.01)
        T_max = self.trainer.max_epochs if self.trainer.max_epochs else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
