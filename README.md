**1. Install Dependencies**
Ensure the required packages (including PyTorch) are installed in your environment:
```bash
pip install torch numpy pandas matplotlib seaborn tifffile tqdm
```

**2. Configure Paths**
Before running, manually update the following hardcoded variables in `test_metrics_curves.py` (around lines 31-41) to match your local file paths:
*   `DATASET_DIR`: Path to your test dataset directory (currently `"./processed_patchs_select"`).
*   `CKPT_PATH`: Path to your trained model weights (currently `"./checkpoints_SS13V2_T6/epoch68-val_loss3.2424.ckpt"`).
*   `NUM_TEST_SAMPLES`: The number of samples to evaluate (default is `500`).
*   `VIZ_OUTPUT_DIR`: The directory where the output charts will be saved (default is `"./test_visualizations_full_pipeline"`).

**3. Run the Script**
Execute the script in your terminal:
```bash
python test_metrics_curves.py
```

**4. Check Results**
*   **Console Output**: Upon completion, the script prints the quantitative alignment metrics to the terminal, including `EPE (Abs)`, `SAR Cycle`, `OPT Cycle`, `PCK<2px`, and `Detected Mean Jitter`.
*   **Visualizations**: A comprehensive 2x2 metrics chart named `00_Combined_Metrics_Panel.png` will be generated and saved in your specified `VIZ_OUTPUT_DIR/metrics_curves/` folder.
*   
Pre-trained Weights: Download the checkpoint from GitHub Releases and place it in the checkpoints_SS13_T6/ folder.
