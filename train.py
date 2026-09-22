import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
import yaml
import warnings
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from dataset import ProxyAlignDataModule
from model import TemporalStabilityPredictor

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


torch.set_float32_matmul_precision('medium')

os.environ['WANDB_API_KEY'] = 'YOUR_WANDB_API_KEY'

def load_config(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"成功加载配置文件: {path}")
    else:
        print(f"未找到配置文件 {path}，使用内置默认参数！")
        config = {
            'DATASET_DIR': './data',
            'OPT_CHANNELS': 4,
            'SAR_CHANNELS': 2,
            'BATCH_SIZE': 16,
            'TRAIN_SEQ_LEN': 6,
            'DATA_WORKER_NUM': 4,
            'TRAIN_EPOCHS': 100,
            'BASE_LR': 3e-4,
            'SEED': 42
        }
    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="./config/config.yml", help="配置文件路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="开启后只跑 1 个 Batch，用于调试")
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config.get('SEED', 42), workers=True)
    torch.backends.cudnn.benchmark = True

    print("\n正在初始化 数据模块 和 密集时空对比网络...")
    datamodule = ProxyAlignDataModule(config)
    model = TemporalStabilityPredictor(config)


    wandb_logger = WandbLogger(
        project="SAR_OPT_Proxy_Alignment",
        name="Phase2_Spatiotemporal_Neighborhood_Consistency",
        config=config,
        mode='disabled' if args.fast_dev_run else 'online'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints_decoupled_contrastive',
        filename='epoch{epoch:02d}-val_loss{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=10)

    trainer = pl.Trainer(
        max_epochs=config.get('TRAIN_EPOCHS', 100),
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        logger=wandb_logger,
        precision="bf16-mixed",
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        num_sanity_val_steps=2
    )

    trainer.fit(model, datamodule=datamodule)
    print("\n训练结束！")

if __name__ == "__main__":
    main()