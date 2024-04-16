import hydra
from omegaconf import DictConfig, OmegaConf
import os

import lightning as L

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from src.model import B2T_CTC, B2T_Model
from src.data import B2T_DataModule

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

import wandb


def get_model(config):
    print(f"Training {config.training_stage}")

    if "ctc" in config.training_stage:
        Model = B2T_CTC
    else:
        Model = B2T_Model
    
    if "phonetic" in config.training_stage:
        phoneme_rec = True
    else:
        phoneme_rec = False
    
    if config.get("from_ckpt") and config.from_ckpt != 0:
        print(f"load parameters from {config.from_ckpt}")
        model = Model.load_from_checkpoint(
            config.from_ckpt, config=config.model, phoneme_rec=phoneme_rec, strict=False
        )
    else:
        model = Model(config.model, phoneme_rec=phoneme_rec)
    
    return model


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):

    working_dir = os.getcwd()

    original_cwd = hydra.utils.get_original_cwd()

    print(f"The current working directory is {working_dir}")

    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)

    torch.autograd.set_detect_anomaly(True)

    if config.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.float32_matmul_precision)



    model = get_model(config)

    # load datamodule
    data_module = B2T_DataModule(config.data_module)

    # loggers
    loggers = []

    if config.get("wandb") and config.wandb:
        wdb = WandbLogger(
            project=config.experiment_name,
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
        **config.trainer,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)

    # trainer.test(model, datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)

    if config.get("wandb") and config.wandb:
        artifact = wandb.Artifact(name="backup", type="configs")
        artifact.add_file(local_path="./valid.txt")
        artifact.add_file(local_path="./test.txt")
        artifact.add_dir(local_path="./.hydra")

        wdb.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
