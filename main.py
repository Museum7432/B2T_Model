import hydra
from omegaconf import DictConfig, OmegaConf
import os

import lightning as L

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from src.model import B2T_Phonemes_CTC
from src.data import B2T_DataModule

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

import wandb


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):

    working_dir = os.getcwd()

    original_cwd = hydra.utils.get_original_cwd()

    print(f"The current working directory is {working_dir}")

    if config.others.get("seed"):
        L.seed_everything(config.others.seed, workers=True)

    torch.autograd.set_detect_anomaly(True)

    if config.others.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.others.float32_matmul_precision)

    # load model
    model = B2T_Phonemes_CTC(config.model)

    # load datamodule
    data_module = B2T_DataModule(config.data)

    # loggers
    loggers = []

    if config.others.get("wandb") and config.others.wandb:
        wdb = WandbLogger(
            project=config.others.experiment_name,
            settings=wandb.Settings(code_dir=original_cwd),
        )
        loggers.append(wdb)

    tb = TensorBoardLogger(save_dir="./", name="", default_hp_metric=False)
    loggers.append(tb)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_weights_only=True,
        dirpath="ckpts",
    )

    callbacks = [lr_monitor, checkpoint_callback]

    trainer = L.Trainer(
        **config.others.trainer,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    # trainer.test(ckpt_path="best", datamodule=data_module)

    if config.others.get("wandb") and config.others.wandb:
        artifact = wandb.Artifact(name="backup", type="configs")
        artifact.add_file(local_path="./valid.txt")
        artifact.add_file(local_path="./test.txt")
        artifact.add_dir(local_path="./.hydra")

        wdb.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
