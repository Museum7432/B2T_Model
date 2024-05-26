from src.train import train
import wandb
from argparse import ArgumentParser

sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_al_loss"},
    "parameters": {
        "conv_size": {"values": [512, 768, 1024, 2048]},

        "conv_kernel1": {
            "values": [3, 5, 7, 9, 11, 13, 15, 19, 21, 23]
        },
        "conv_kernel2": {"values": [3, 5, 7, 9, 11, 13]},
        "conv_g1": {"values": [256, 64, 32, 4, 1]},
        "conv_g2": {"values": [256, 64, 32, 4, 1]},

        "hidden_size": {"values": [512, 768]},

        "encoder_n_layer": {"min": 1, "max":13},
        "decoder_n_layer": {"min": 1, "max":8},
        
        
        "peak_lr": {"max": 0.02, "min": 2e-5},
        "update_probs": {"max": 0.95, "min": 0.5},

        "al_loss_weight": {"max": 0.7, "min": 0.3},

        "last_lr": {"value": 1e-6},
        "beta_1": {"value": 0.9},
        "beta_2": {"value": 0.95},

        "weight_decay": {"max": 0.1, "min": 0.0001},

        "eps": {"value": 1e-08},
        "lr_warmup_perc": {"max": 0.5, "min": 0.1},
        "add_noises": {"value": True},
        "train_data_dir": {"value": "./dataset/train"},
        "val_data_dir": {"value": "./dataset/valid"},
        "test_data_dir": {"value": "./dataset/test"},
        "word_level": {"value": False},
        "use_addtional_corpus": {"value": False},
        "sp_noise_std": {"max": 1.0, "min": 0.01},
        "gaussian_filter_sigma": {"max": 3.0, "min": 0.01},
        "debugging": {"value": False},
        "train_batch_size": {"values": [8, 16]},
        # change this
        "valid_batch_size": {"value": 16},
        "num_workers": {"value": 8},
        "val_check_interval": {"value": 0.5},
        "max_epochs": {"min": 4, "max":25},

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
