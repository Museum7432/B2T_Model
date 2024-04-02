import lightning as L

from argparse import ArgumentParser
from model import B2T_Model
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from data import B2T_DataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch


def main(args):
    L.seed_everything(69, workers=True)

    torch.autograd.set_detect_anomaly(True)

    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    data_module = B2T_DataModule(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        test_data_dir=args.test_data_dir,
        # tokenizer_name=args.seq2seq_model,
        train_batch_size = args.train_batch_size,
        valid_batch_size= args.valid_batch_size,
        num_workers = args.num_workers,
        debugging=args.debugging
    )

    loggers = []
    if args.wandb:
        loggers.append(WandbLogger(project=args.exp_name))

    tb = TensorBoardLogger("./")
    loggers.append(tb)

    log_dir = tb.log_dir

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        save_weights_only=True,
        dirpath=log_dir + "/checkpoint",
    )

    callbacks = [lr_monitor, checkpoint_callback]

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=loggers,
        callbacks=callbacks,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
        val_check_interval=args.val_check_interval
    )

    model = B2T_Model(
        lr=args.lr,
        log_dir=log_dir,
        input_channels=256,
        vocab_size = 150
    )

    trainer.fit(model, datamodule=data_module)

    # trainer.test(model, datamodule=data_module)

    # trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--float32_matmul_precision", type=str, default="highest")
    parser.add_argument("--val_check_interval", type=float, default=0.5)

    parser.add_argument("--exp_name", type=str, default="brain2text")

    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=-1)

    parser.add_argument("--seq2seq_model", type=str, default="google-t5/t5-small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--conv_hidden_size", type=int, default=None)

    parser.add_argument("--block_size", type=int, default=16)
    
    parser.add_argument("--train_data_dir", type=str, default="competitionData/train")
    parser.add_argument("--val_data_dir", type=str, default="competitionData/test")
    parser.add_argument("--test_data_dir", type=str, default="competitionData/competitionHoldOut")

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--debugging", action="store_true")

    args = parser.parse_args()

    main(args)
