import hydra
from omegaconf import DictConfig, OmegaConf
import os

import lightning as L

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from src.model import joint_Model
from src.data import B2T_DataModule

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

import wandb

import shutil


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):

    working_dir = os.getcwd()

    original_cwd = hydra.utils.get_original_cwd()

    for f in ["data", "model", "utils"]:
        shutil.copyfile(
            os.path.join(original_cwd, f"src/{f}.py"),
            os.path.join(working_dir, f".hydra/{f}.py")
        )

    print(f"The current working directory is {working_dir}")

    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)

    torch.autograd.set_detect_anomaly(True)

    if config.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

    if config.get("from_ckpt") and config.from_ckpt != 0:
        model = joint_Model.load_from_checkpoint(config.from_ckpt, strict=False, config=config.model, decoders_conf=None)
    else:
        model = joint_Model(config.model)
    
    if config.model.get("max_epochs"):
        config.trainer.max_epochs=config.model.max_epochs

    if config.model.get("val_check_interval"):
        config.trainer.val_check_interval=config.model.val_check_interval
    
    if config.model.get("word_level"):
        config.data_module.word_level=config.model.word_level

    if config.model.get("use_addtional_corpus"):
        config.data_module.use_addtional_corpus=config.model.use_addtional_corpus

    # load datamodule
    data_module = B2T_DataModule(config.data_module)

    # loggers
    loggers = []

    if config.get("wandb") and config.wandb:
        wdb = WandbLogger(
            project=config.experiment_name,
            settings=wandb.Settings(code_dir=original_cwd)
        )
        loggers.append(wdb)

        artifact = wandb.Artifact(name="configs", type="configs")
        artifact.add_dir(local_path="./.hydra")
        wdb.experiment.log_artifact(artifact)


    tb = TensorBoardLogger(save_dir="./", name="", default_hp_metric=False)
    loggers.append(tb)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="wer",
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

    # if config.get("wandb") and config.wandb:
    #     artifact = wandb.Artifact(name="backup", type="configs")
    #     # artifact.add_file(local_path="./valid.txt")
    #     # artifact.add_file(local_path="./test.txt")
    #     artifact.add_dir(local_path="./.hydra")

    #     wdb.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
