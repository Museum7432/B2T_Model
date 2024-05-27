from src.train import train
import wandb
from argparse import ArgumentParser

sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "conv_size": {"values": [512, 768, 1024]},
        "conv_kernel1": {"values": [3, 5, 7, 9, 11, 13, 15, 19, 21]},
        "conv_kernel2": {"values": [3, 5, 7]},
        "conv_g1": {"values": [256, 4, 1]},
        "conv_g2": {"values": [256, 4, 1]},
        "hidden_size": {"value": 512},
        "encoder_n_layer": {"min": 5, "max": 10},
        "decoder_n_layer": {"min": 2, "max": 5},
        "peak_lr": {"max": 1e-3, "min": 3e-6},
        "update_probs": {"max": 0.9, "min": 0.5},
        "al_loss_weight": {"value": 0.5},
        "last_lr": {"value": 1e-6},
        "beta_1": {"max": 0.1, "min": 1.},
        "beta_2": {"max": 0.1, "min": 1.},
        "eps": {"value": 1e-08},
        "weight_decay": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},

        "lr_warmup_perc": {"max": 0.5, "min": 0.1},
        "add_noises": {"value": True},
        "train_data_dir": {"value": "./dataset/train"},
        "val_data_dir": {"value": "./dataset/valid"},
        "test_data_dir": {"value": "./dataset/test"},
        "word_level": {"value": False},
        "use_addtional_corpus": {"value": False},
        "sp_noise_std": {"max": 1.0, "min": 0.01},
        "gaussian_filter_sigma": {"max": 2.0, "min": 0.1},
        "debugging": {"value": False},
        "train_batch_size": {"value": 8},
        # change this
        "valid_batch_size": {"value": 16},
        "num_workers": {"value": 10},
        "val_check_interval": {"value": 0.9},
        # "max_epochs": {"min": 4, "max":10},
        "max_epochs": {"value": 10},
        "gradient_clip_val": {"max": 1.5, "min": 0.1},
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
