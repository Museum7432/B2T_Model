from src.train import train
import wandb


sweep_configuration = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_al_loss"},
    "parameters": {
        "conv_size": {"values": [512, 768, 1024, 2048]},
        "hidden_size": {"values": [512, 768, 1024, 1536, 2048]},
        "encoder_n_layer": {"values": [6, 8, 12, 14, 16, 20, 24, 26, 28, 30]},
        "decoder_n_layer": {"values": [4, 6, 8, 10, 12]},
        "lr": {"max": 0.1, "min": 2e-5},
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
        "sp_noise_std": {"max": 1., "min": 0.05},
        "gaussian_filter_sigma": {"max": 3., "min": 0.01},
        "debugging": {"value": False},
        "train_batch_size": {"values": [4, 8, 16, 32]},

        # change this
        "valid_batch_size": {"value": 16},
        "num_workers": {"value": 8},
        "val_check_interval": {"value": 0.5},

        "max_epochs": {"values": [5, 10, 15, 20]},

        "gradient_clip_val": {"value": 0.8},

    },
}

if __name__ == '__main__':

    sweep_id=wandb.sweep(sweep_configuration, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=train)
