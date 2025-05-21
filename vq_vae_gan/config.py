"""
Configuration for VQ-VAE-GAN model
"""

class Config:
    # Data settings
    dataset = 'kodak'  # options: 'kodak', 'clic', 'div2k'
    image_size = 256
    batch_size = 16
    num_workers = 4
    data_dir = './data'
    
    # Model architecture
    embedding_dim = 64  # Size of each embedding vector
    num_embeddings = 512  # Number of embedding vectors (codebook size)
    channels = 3
    enc_channels = [64, 128, 256, 512]
    dec_channels = [512, 256, 128, 64]
    hidden_channels = 256  # Channels in the hidden representation
    
    # VQ-VAE specific parameters
    commitment_cost = 0.25  # Controls the weighting of the commitment loss
    
    # Training settings
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    epochs = 100
    save_interval = 5
    decay = 0.99  # EMA decay for updating codebook
    beta = 1.0  # VQ-VAE stop-gradient parameter
    lambda_adv = 0.1  # Adversarial loss weight
    
    # Inference and evaluation
    eval_batch_size = 1
    
    # Logging and output
    output_dir = './output/vq_vae_gan'
    log_interval = 100
    tensorboard_log_dir = './runs/vq_vae_gan'
    
    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    
    # Seeds
    seed = 42