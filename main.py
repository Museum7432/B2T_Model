import hydra
from omegaconf import DictConfig, OmegaConf
import os

import lightning as L

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from src.model import B2T_Phonemes_CTC
from src.data import B2T_DataModule


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(config: DictConfig):

    # print(print(OmegaConf.to_yaml(config)))
    # return

    working_dir = os.getcwd()
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

    trainer = L.Trainer(**config.others.trainer)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
