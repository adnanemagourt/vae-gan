"""
Configuration for Hierarchical VAE-GAN model
"""

class Config:
    # Data settings
    dataset = 'kodak'  # options: 'kodak', 'clic', 'div2k'
    image_size = 256
    batch_size = 16
    num_workers = 4
    data_dir = './data'
    
    # Model architecture
    channels = 3
    base_channels = 64  # Base number of channels
    
    # Hierarchical latent dimensions
    num_hierarchies = 3  # Number of hierarchical levels
    latent_dims = [32, 64, 128]  # Latent dimensions for each level
    
    # Training settings
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    epochs = 100
    save_interval = 5
    
    # Loss weights
    beta = 1.0  # KL divergence weight
    lambda_adv = 0.1  # Adversarial loss weight
    
    # Hierarchical specific
    level_weights = [1.0, 0.75, 0.5]  # Weights for different hierarchy levels
    
    # Inference and evaluation
    eval_batch_size = 1
    
    # Logging and output
    output_dir = './output/hierarchical_vae_gan'
    log_interval = 100
    tensorboard_log_dir = './runs/hierarchical_vae_gan'
    
    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    
    # Seeds
    seed = 42