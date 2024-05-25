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


# def main():

base_config = {
    "conv_size": 1024,
    "hidden_size": 512,
    "encoder_n_layer": 2,
    "decoder_n_layer": 2,
    "update_probs": 0.5,
    "al_loss_weight": 0.5,
    "peak_lr": 1e-4,
    "last_lr": 1e-6,
    "beta_1": 0.9,
    "beta_2": 0.95,
    "weight_decay": 0.1,
    "eps": 1e-08,
    "lr_warmup_perc": 0.1,
    "add_noises": True,
    "train_data_dir": "./dataset/train",
    "val_data_dir": "./dataset/valid",
    "test_data_dir": "./dataset/test",
    "word_level": False,
    "use_addtional_corpus": False,
    "sp_noise_std": 0.2,
    "gaussian_filter_sigma": 0.8,
    "debugging": False,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "num_workers": 4,
    "max_epochs": 10,
    "val_check_interval": 0.5,
    "gradient_clip_val": 1.0,
}


def train(config=None, project_name="sweep"):
    L.seed_everything(49, workers=True)

    if config is None:
        wandb.init(project="sweep")
        config=wandb.config

    # load datamodule
    data_module = B2T_DataModule(**base_config)

    model = B2T_Model(**base_config)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True
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
