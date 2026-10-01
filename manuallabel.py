import os
import json
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import MultiCursor

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

DATASET_DIR = "./processed_patchs_select"
MODEL_EXPORT_JSON = "./model_exported_points.json"

def load_exact_samples(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}! Please run test_metrics_curves.py first")

    with open(json_path, 'r', encoding='utf-8') as f:
        exported_data = json.load(f)

    valid_grids = []
    for item in exported_data:
        valid_grids.append({
            "id": item['grid_id'],
            "opt_files": [item['opt1_file'], item['opt2_file']],
            "sar_files": [item['sar1_file'], item['sar2_file']],
            "model_info": item
        })
    print(f"Successfully loaded {len(valid_grids)} exact adversarial samples!")
    return valid_grids

def read_and_stretch(filepath, is_opt=True):
    img_array = tifffile.imread(filepath).astype(np.float32)
    if img_array.ndim == 3 and img_array.shape[-1] <= 4:
        img_array = img_array.transpose(2, 0, 1)
    elif img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=0)

    img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
    vis_img = []
    for c in range(min(3, img_array.shape[0]) if is_opt else 1):
        channel_data = img_array[c]
        valid_pixels = channel_data[(channel_data > -9000) & (channel_data < 9000) & (channel_data != 0)]
        vmin, vmax = (np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)) if len(valid_pixels) > 0 else (
            0.0, 0.0)
        channel_data = (channel_data - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-5 else np.zeros_like(channel_data)
        vis_img.append(np.clip(channel_data, 0.0, 1.0))

    vis_img = np.stack(vis_img, axis=-1)
    if not is_opt and vis_img.shape[-1] == 1: vis_img = np.repeat(vis_img, 3, axis=-1)
    return vis_img

class ExactMatchUI:
    def __init__(self, val_grids):
        self.val_grids = val_grids
        self.current_grid_idx = 0
        self.cur_idx = 0
        self.auto_bright = False
        self.show_metrics = False
        self.current_metrics = ""

        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.subplots_adjust(top=0.82)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.load_current_grid()

    def load_current_grid(self):
        if self.current_grid_idx >= len(self.val_grids):
            print("\nThe 4 samples exported by the model have been tested!")
            plt.close(self.fig)
            return

        grid = self.val_grids[self.current_grid_idx]
        print(f"\nLoading exact adversarial sample {self.current_grid_idx + 1}/4 | ID: {grid['id']}")
        self.model_info = grid['model_info']

        self.opt_file_names = [os.path.basename(f) for f in grid['opt_files']]
        self.sar_file_names = [os.path.basename(f) for f in grid['sar_files']]

        self.opt_imgs = [read_and_stretch(f, True) for f in grid['opt_files']]
        self.sar_imgs = [read_and_stretch(f, False) for f in grid['sar_files']]

        self.cur_idx = 0
        self.show_metrics = False
        self.opt_points = [None, None]
        self.sar_points = [None, None]
        self.redraw()

    def _apply_auto_brightness(self, img):
        res = np.copy(img)
        for c in range(res.shape[-1]):
            valid_mask = res[..., c] > 0
            if np.any(valid_mask):
                mean_val = np.mean(res[..., c][valid_mask])
                if 0 < mean_val < 0.4:
                    res[..., c] = np.power(res[..., c], np.clip(np.log(0.5) / np.log(mean_val), 0.3, 1.0))
        return res

    def _compare_metrics(self):
        if any(p is None for p in self.opt_points + self.sar_points):
            print("\nCannot calculate! You must mark points on all 4 images!")
            return

        o1x, o1y = self.opt_points[0]
        o2x, o2y = self.opt_points[1]
        s1x, s1y = self.sar_points[0]
        s2x, s2y = self.sar_points[1]
        m = self.model_info

        if 'dx1' not in m:
            print("\nError: Your JSON file is missing prediction parameters! Please modify test.py according to the tutorial to regenerate JSON!")
            return

        h_s1_dx, h_s1_dy = s1x - o1x, s1y - o1y
        h_s2_dx, h_s2_dy = s2x - o1x, s2y - o1y
        h_o2_dx, h_o2_dy = o2x - o1x, o2y - o1y

        m_s1_dx, m_s1_dy = m['dx1'] - m['p1_x'], m['dy1'] - m['p1_y']
        m_s2_dx, m_s2_dy = m['dx2'] - m['p2_x'], m['dy2'] - m['p2_y']

        m_o2_dx, m_o2_dy = m['dx3'] - (m['p1_x'] - m['p3_x']), m['dy3'] - (m['p1_y'] - m['p3_y'])

        epe_s1 = np.hypot(m_s1_dx - h_s1_dx, m_s1_dy - h_s1_dy)
        epe_s2 = np.hypot(m_s2_dx - h_s2_dx, m_s2_dy - h_s2_dy)
        epe_o2 = np.hypot(m_o2_dx - h_o2_dx, m_o2_dy - h_o2_dy)

        human_sar_cycle = np.hypot(s2x - s1x, s2y - s1y)
        human_opt_cycle = np.hypot(o2x - o1x, o2y - o1y)

        print("\n" + "=" * 65)
        print(f"Absolute Evaluation using [You (Human)] as GT (Sample: {self.model_info['grid_id']})")
        print("=" * 65)
        print(f"Final Model Error (EPE)")
        print(f" ├─ SAR_1 Matching Error : {epe_s1:>6.2f} px")
        print(f" ├─ SAR_2 Matching Error : {epe_s2:>6.2f} px")
        print(f" └─ OPT_2 Matching Error : {epe_o2:>6.2f} px")
        print("-" * 65)
        print(f"Internal Structure Jitter Detection (Cycle) | Your GT Judgment | Model's Judgment")
        print(f" ├─ SAR Modality Internal Offset        | {human_sar_cycle:>8.2f} px   | {m['metrics']['sar_cycle']:>8.2f} px")
        print(f" └─ OPT Modality Internal Offset        | {human_opt_cycle:>8.2f} px   | {m['metrics']['opt_cycle']:>8.2f} px")
        print("=" * 65)

        self.current_metrics = f"EPE (Based on You): S1={epe_s1:.1f}px | S2={epe_s2:.1f}px | O2={epe_o2:.1f}px"
        self.show_metrics = True

    def redraw(self):
        self.axes[0].clear(); self.axes[1].clear()

        opt_disp = self._apply_auto_brightness(self.opt_imgs[self.cur_idx]) if self.auto_bright else self.opt_imgs[
            self.cur_idx]
        sar_disp = self._apply_auto_brightness(self.sar_imgs[self.cur_idx]) if self.auto_bright else self.sar_imgs[
            self.cur_idx]

        self.axes[0].imshow(opt_disp)
        self.axes[0].set_title(f"OPT T{self.cur_idx + 1} \n{self.opt_file_names[self.cur_idx]}", fontsize=12,
                               color='cyan')
        self.axes[0].axis('off')

        self.axes[1].imshow(sar_disp)
        self.axes[1].set_title(f"SAR T{self.cur_idx + 1} \n{self.sar_file_names[self.cur_idx]}", fontsize=12,
                               color='cyan')
        self.axes[1].axis('off')

        if self.cur_idx == 0:
            mx, my = self.model_info['anchor_x'], self.model_info['anchor_y']
            gap, length = 12, 20
            self.axes[0].plot([mx - gap - length, mx - gap], [my, my], color='yellow', lw=2)
            self.axes[0].plot([mx + gap, mx + gap + length], [my, my], color='yellow', lw=2)
            self.axes[0].plot([mx, mx], [my - gap - length, my - gap], color='yellow', lw=2)
            self.axes[0].plot([mx, mx], [my + gap, my + gap + length], color='yellow', lw=2)
            self.axes[0].text(mx + gap + 5, my - gap - 5, "Target Feature", color='yellow', fontsize=12, weight='bold',
                              bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

        if self.opt_points[self.cur_idx]:
            ox, oy = self.opt_points[self.cur_idx]
            self.axes[0].plot(ox, oy, 'rx', markersize=10, markeredgewidth=2)
        if self.sar_points[self.cur_idx]:
            sx, sy = self.sar_points[self.cur_idx]
            self.axes[1].plot(sx, sy, 'yx', markersize=10, markeredgewidth=2)
            if self.opt_points[self.cur_idx]:
                con = ConnectionPatch(xyA=(sx, sy), xyB=(ox, oy), coordsA="data", coordsB="data", axesA=self.axes[1],
                                      axesB=self.axes[0], color="lime", lw=1.5, ls=':')
                self.axes[1].add_artist(con)

        title_str = (f"Sample: {self.current_grid_idx + 1}/4 | {self.model_info['grid_id']}\n"
                     f"Task: Find the feature at the center of the yellow [hollow cross] in T1, and click on all 4 images!\n"
                     f"Operations: [Left Click] Mark Point | [Up/Down] Turn Page T1/T2 | [E] Calculate Result | [N] Next")
        if self.show_metrics: title_str += f"\n{self.current_metrics}"

        self.fig.suptitle(title_str, fontsize=12, weight='bold')
        self.cursor = MultiCursor(self.fig.canvas, self.axes, color='cyan', lw=1, ls='--', horizOn=True, vertOn=True,
                                  useblit=True)
        self.fig.canvas.draw()

    def onscroll(self, event):
        if event.button == 'up':
            self.cur_idx = max(0, self.cur_idx - 1)
        elif event.button == 'down':
            self.cur_idx = min(1, self.cur_idx + 1)
        self.redraw()

    def onclick(self, event):
        if event.button == 1:
            if event.inaxes == self.axes[0]:
                self.opt_points[self.cur_idx] = (event.xdata, event.ydata)
            elif event.inaxes == self.axes[1]:
                self.sar_points[self.cur_idx] = (event.xdata, event.ydata)
            self.redraw()

    def onkey(self, event):
        if event.key in ['c', 'C']:
            self.auto_bright = not self.auto_bright; self.redraw()
        elif event.key in ['e', 'E']:
            self._compare_metrics(); self.redraw()
        elif event.key == 'up':
            self.cur_idx = max(0, self.cur_idx - 1); self.redraw()
        elif event.key == 'down':
            self.cur_idx = min(1, self.cur_idx + 1); self.redraw()
        elif event.key in ['n', 'N']:
            self.current_grid_idx += 1; self.load_current_grid()
        elif event.key in ['q', 'Q']:
            plt.close(self.fig)


if __name__ == "__main__":
    app = ExactMatchUI(load_exact_samples(MODEL_EXPORT_JSON))
    plt.show()
