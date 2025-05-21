"""
Configuration for β-VAE-GAN model
"""

class Config:
    # Data settings
    dataset = 'kodak'  # options: 'kodak', 'clic', 'div2k'
    image_size = 256
    batch_size = 16
    num_workers = 4
    data_dir = './data'
    
    # Model architecture
    latent_dim = 128
    channels = 3
    enc_channels = [64, 128, 256, 512]
    dec_channels = [512, 256, 128, 64]
    
    # Training settings
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    epochs = 100
    save_interval = 5
    beta = 1.0  # KL divergence weight (β parameter)
    lambda_adv = 0.1  # Adversarial loss weight
    
    # Inference and evaluation
    eval_batch_size = 1
    
    # Logging and output
    output_dir = './output/beta_vae_gan'
    log_interval = 100
    tensorboard_log_dir = './runs/beta_vae_gan'
    
    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    
    # Seeds
    seed = 42