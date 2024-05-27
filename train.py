from src.train import train


base_config = {
    "conv_size": 1024,
    "conv_kernel1": 7,
    "conv_kernel2": 3,
    "conv_g1": 256,
    "conv_g2": 4,
    "hidden_size": 512,
    "encoder_n_layer": 7,
    "decoder_n_layer": 2,
    "update_probs": 0.85,
    "al_loss_weight": 0.5,
    "peak_lr": 1e-4,
    "last_lr": 1e-6,
    "beta_1": 0.9,
    "beta_2": 0.95,
    "weight_decay": 0.1,
    "eps": 1e-08,
    "lr_warmup_perc": 0.4,
    "add_noises": True,
    "train_data_dir": "./dataset/train",
    "val_data_dir": "./dataset/valid",
    "test_data_dir": "./dataset/test",
    "word_level": False,
    "use_addtional_corpus": False,
    "sp_noise_std": 0.1,
    "features_noise_std": 0.1,
    "gaussian_filter_sigma": 0.8,
    "debugging": False,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "num_workers": 4,
    "max_epochs": 10,
    "val_check_interval": 0.5,
    "gradient_clip_val": 1.0,
}

if __name__ == "__main__":

    train(base_config, save_ckpt=True)
