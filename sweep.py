from src.train import train
import wandb
from argparse import ArgumentParser

sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_al_loss"},
    "parameters": {
        "conv_size": {"values": [512, 768, 1024, 2048]},
        "hidden_size": {"values": [512, 768]},
        "encoder_n_layer": {"values": [6, 7, 8, 9, 10, 11, 12, 13]},
        "decoder_n_layer": {"values": [2, 3, 4, 5, 6]},
        "peak_lr": {"max": 0.1, "min": 2e-5},
        "update_probs": {"max": 0.95, "min": 0.5},
        "al_loss_weight": {"value": 0.5},
        "last_lr": {"value": 1e-6},
        "beta_1": {"value": 0.9},
        "beta_2": {"value": 0.95},
        "weight_decay": {"value": 0.1},
        "eps": {"value": 1e-08},
        "lr_warmup_perc": {"max": 0.5, "min": 0.1},
        "add_noises": {"value": True},
        "train_data_dir": {"value": "./dataset/train"},
        "val_data_dir": {"value": "./dataset/valid"},
        "test_data_dir": {"value": "./dataset/test"},
        "word_level": {"value": False},
        "use_addtional_corpus": {"value": False},
        "sp_noise_std": {"max": 1.0, "min": 0.05},
        "gaussian_filter_sigma": {"max": 3.0, "min": 0.01},
        "debugging": {"value": False},
        "train_batch_size": {"values": [8, 16]},
        # change this
        "valid_batch_size": {"value": 16},
        "num_workers": {"value": 8},
        "val_check_interval": {"value": 0.5},
        "max_epochs": {"values": [5, 7, 10, 12, 15]},
        "gradient_clip_val": {"max": 1.0, "min": 0.1},
    },
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    args = parser.parse_args()

    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_configuration, project="test_sweep")
    else:
        sweep_id = args.sweep_id
    wandb.agent(sweep_id=sweep_id, function=train)
