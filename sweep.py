from src.train import train
import wandb
from argparse import ArgumentParser

sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "conv_size": {"value": 1024},
        "conv_kernel1": {"values": [3, 5, 7]},
        "conv_kernel2": {"value": 3},
        "conv_g1": {"value": 256},
        "conv_g2": {"value": 1},
        "hidden_size": {"value": 768},
        "encoder_n_layer": {"value": 6},
        "decoder_n_layer": {"value": 4},
        "peak_lr": {"values": [3e-4, 2e-4, 1e-4]},
        ##############################
        # this will need another sweep
        # "update_probs": {"max": 0.9, "min": 0.5},
        ##############################
        "al_loss_weight": {"value": 0.5},
        "last_lr": {"value": 1e-6},
        "beta_1": {"value": 0.9},
        "beta_2": {"value": 0.95},
        "eps": {"value": 1e-08},
        "weight_decay": {"value": 0.1},
        "lr_warmup_perc": {"values": [0.2, 0.1, 0.3]},
        "add_noises": {"value": True},
        "train_data_dir": {"value": "./dataset/train"},
        "val_data_dir": {"value": "./dataset/valid"},
        "test_data_dir": {"value": "./dataset/test"},
        "word_level": {"value": False},
        "use_addtional_corpus": {"value": False},
        "sp_noise_std": {"values": [0.2, 0.15, 0.1, 0.05, 0]},
        "features_noise_std": {"values": [0.2, 0.15, 0.1, 0.05, 0]},
        "gaussian_filter_sigma": {"values": [1, 0.8, 0.5]},
        "debugging": {"value": False},
        "train_batch_size": {"value": 8},
        # change this
        "valid_batch_size": {"value": 16},
        "num_workers": {"value": 8},
        "val_check_interval": {"value": 0.5},
        # "max_epochs": {"min": 4, "max":10},
        "max_epochs": {"value": 10},
        "gradient_clip_val": {"value": 1.0},
    },
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    args = parser.parse_args()

    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_configuration, project="test_sweep")
        print(sweep_id)
    else:
        sweep_id = args.sweep_id
    wandb.agent(sweep_id=sweep_id, function=train)
