import os

import lightning as L

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from .model import B2T_Model
from .data import B2T_DataModule

import wandb
from jsonargparse import ArgumentParser
import shutil


def train(config=None, project_name="sweep"):
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(49, workers=True)

    if config is None:
        wandb.init(project="sweep")
        config = wandb.config

    # load datamodule
    data_module = B2T_DataModule(**config)

    model = B2T_Model(**config)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [lr_monitor, checkpoint_callback]

    wandb_logger = WandbLogger(project=project_name)

    wandb_logger.watch(model)

    trainer = L.Trainer(
        accelerator="gpu",
        val_check_interval=config["val_check_interval"],
        max_epochs=config["max_epochs"],
        gradient_clip_val=config["gradient_clip_val"],
        logger=[wandb_logger],
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":

    train()