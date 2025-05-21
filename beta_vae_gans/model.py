"""
β-VAE-GAN model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Skip connection if channel dimensions differ
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.skip(residual)
        out = self.relu(out)
        
        return out


class Encoder(nn.Module):
    """VAE Encoder network"""
    
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256, 512], latent_dim=128):
        super(Encoder, self).__init__()
        
        layers = []
        
        # Initial convolution
        layers.append(nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Residual blocks with downsampling
        for i in range(len(hidden_channels) - 1):
            layers.append(ResidualBlock(hidden_channels[i], hidden_channels[i]))
            layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final residual block
        layers.append(ResidualBlock(hidden_channels[-1], hidden_channels[-1]))
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature map size after downsampling
        # For input 256x256, after 4 downsampling operations: 16x16
        self.feature_map_size = 256 // (2 ** (len(hidden_channels) - 1))
        
        # Fully connected layers for mean and log variance
        fc_input_dim = hidden_channels[-1] * self.feature_map_size * self.feature_map_size
        self.fc_mu = nn.Linear(fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_input_dim, latent_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        features = self.features(x)
        
        # Flatten features
        features = features.view(batch_size, -1)
        
        # Get mean and log variance
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Ensure logvar is bounded for numerical stability
        logvar = torch.clamp(logvar, min=-20, max=20)
        
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder network"""
    
    def __init__(self, latent_dim=128, hidden_channels=[512, 256, 128, 64], output_channels=3):
        super(Decoder, self).__init__()
        
        # Calculate feature map size at the beginning of the decoder
        # For output 256x256, after 4 upsampling operations: 16x16
        self.feature_map_size = 256 // (2 ** len(hidden_channels))
        
        # Initial fully connected layer
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * self.feature_map_size * self.feature_map_size)
        
        # Initial reshaping
        fc_output_dim = hidden_channels[0] * self.feature_map_size * self.feature_map_size
        self.init_size = self.feature_map_size
        
        # Build decoder with upsampling blocks
        layers = []
        
        # Initial convolution from latent projection
        layers.append(nn.BatchNorm2d(hidden_channels[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Upsampling blocks
        for i in range(len(hidden_channels) - 1):
            layers.append(ResidualBlock(hidden_channels[i], hidden_channels[i]))
            layers.append(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1], 
                                           kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        layers.append(ResidualBlock(hidden_channels[-1], hidden_channels[-1]))
        layers.append(nn.Conv2d(hidden_channels[-1], output_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())  # Output in range [0, 1]
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.fc(z)
        x = x.view(batch_size, -1, self.init_size, self.init_size)
        
        # Decode
        x = self.decoder(x)
        
        return x


class Discriminator(nn.Module):
    """GAN Discriminator network"""
    
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        
        layers = []
        
        # Initial convolution
        layers.append(nn.Conv2d(input_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Calculate output feature dimension
        # For input 256x256, after 4 downsampling operations: 16x16
        feature_map_size = 256 // (2 ** len(hidden_channels))
        
        # PatchGAN output - no need to reshape to a single value
        layers.append(nn.Conv2d(hidden_channels[-1], 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class BetaVAEGAN(nn.Module):
    """β-VAE-GAN model for image compression"""
    
    def __init__(self, config):
        super(BetaVAEGAN, self).__init__()
        
        # Get configuration parameters
        self.latent_dim = config.latent_dim
        self.beta = config.beta
        self.input_channels = config.channels
        self.enc_channels = config.enc_channels
        self.dec_channels = config.dec_channels
        self.lambda_adv = config.lambda_adv
        
        # Initialize components
        self.encoder = Encoder(
            input_channels=self.input_channels,
            hidden_channels=self.enc_channels,
            latent_dim=self.latent_dim
        )
        
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_channels=self.dec_channels,
            output_channels=self.input_channels
        )
        
        self.discriminator = Discriminator(
            input_channels=self.input_channels,
            hidden_channels=self.enc_channels
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """Encode input to latent representation"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        # Encode
        z, mu, logvar = self.encode(x)
        
        # Decode
        recon_x = self.decode(z)
        
        return recon_x, z, mu, logvar
    
    def loss_function(self, original, recon_x, mu, logvar, disc_real, disc_fake, train_generator=True):
        """
        Calculate the loss for both VAE and GAN components
        
        Args:
            original: Original input image
            recon_x: Reconstructed image
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake (reconstructed) images
            train_generator: Whether to compute generator loss (True) or discriminator loss (False)
        
        Returns:
            Dictionary of loss components and total loss
        """
        # Reconstruction loss (pixel-wise)
        recon_loss = F.mse_loss(recon_x, original, reduction='mean')
        
        # KL Divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # GAN losses
        if train_generator:
            # Generator wants discriminator to predict reconstructions as real
            adv_loss = -torch.mean(disc_fake)
            
            # Total loss for generator (VAE + GAN)
            total_loss = recon_loss + self.beta * kl_loss + self.lambda_adv * adv_loss
            
            return {
                'total': total_loss,
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'adv': adv_loss.item(),
                'disc': 0.0  # Not calculated when training generator
            }
        else:
            # Discriminator loss
            disc_loss = torch.mean(F.relu(1.0 - disc_real)) + torch.mean(F.relu(1.0 + disc_fake))
            
            return {
                'total': disc_loss,
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'adv': 0.0,  # Not calculated when training discriminator
                'disc': disc_loss.item()
            }