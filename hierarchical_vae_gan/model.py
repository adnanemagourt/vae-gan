"""
Hierarchical VAE-GAN model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange


class SelfAttention(nn.Module):
    """Self-attention module for feature maps"""
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            out: Self-attention output, same shape as input
        """
        batch_size, C, height, width = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        
        attention = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = self.softmax(attention)  # B x (H*W) x (H*W)
        
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)  # B x C x H x W
        
        out = self.gamma * out + x
        
        return out


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


class HierarchicalEncoder(nn.Module):
    """Hierarchical VAE Encoder network"""
    
    def __init__(self, input_channels=3, base_channels=64, latent_dims=[32, 64, 128], num_hierarchies=3):
        super(HierarchicalEncoder, self).__init__()
        
        assert len(latent_dims) == num_hierarchies, "Number of latent dimensions must match number of hierarchies"
        
        self.num_hierarchies = num_hierarchies
        self.latent_dims = latent_dims
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Create encoder blocks for each hierarchy level
        self.encoders = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        
        # Current feature map channels and size
        curr_channels = base_channels
        
        for i in range(num_hierarchies):
            # Compute channels for this level
            level_channels = base_channels * (2 ** i)
            
            # Create encoder block
            encoder_block = nn.Sequential(
                # Downsampling convolution
                nn.Conv2d(curr_channels, level_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(level_channels),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Residual blocks
                ResidualBlock(level_channels, level_channels),
                ResidualBlock(level_channels, level_channels)
            )
            
            # Add self-attention for deeper levels
            if i >= 1:
                encoder_block = nn.Sequential(
                    encoder_block,
                    SelfAttention(level_channels)
                )
            
            self.encoders.append(encoder_block)
            
            # Create mean and log variance layers for this level
            self.mu_layers.append(nn.Conv2d(level_channels, latent_dims[i], kernel_size=1))
            self.logvar_layers.append(nn.Conv2d(level_channels, latent_dims[i], kernel_size=1))
            
            # Update current channels for next level
            curr_channels = level_channels
    
    def forward(self, x):
        """
        Forward pass through hierarchical encoder
        
        Args:
            x: Input image tensor
            
        Returns:
            mu_list: List of mean tensors for each hierarchy level
            logvar_list: List of log variance tensors for each hierarchy level
        """
        mu_list = []
        logvar_list = []
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Process each hierarchy level
        for i in range(self.num_hierarchies):
            # Pass through encoder for this level
            x = self.encoders[i](x)
            
            # Get mean and log variance
            mu = self.mu_layers[i](x)
            logvar = self.logvar_layers[i](x)
            
            # Ensure logvar is bounded for numerical stability
            logvar = torch.clamp(logvar, min=-20, max=20)
            
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        return mu_list, logvar_list


class HierarchicalDecoder(nn.Module):
    """Hierarchical VAE Decoder network"""
    
    def __init__(self, output_channels=3, base_channels=64, latent_dims=[32, 64, 128], num_hierarchies=3):
        super(HierarchicalDecoder, self).__init__()
        
        assert len(latent_dims) == num_hierarchies, "Number of latent dimensions must match number of hierarchies"
        
        self.num_hierarchies = num_hierarchies
        self.latent_dims = latent_dims
        
        # Create decoder blocks for each hierarchy level
        self.decoders = nn.ModuleList()
        
        # Process hierarchy levels in reverse order (deepest first)
        for i in range(num_hierarchies-1, -1, -1):
            # Compute channels for this level
            level_channels = base_channels * (2 ** i)
            
            # For deepest level, convert from latent dimension to feature map
            if i == num_hierarchies - 1:
                decoder_block = nn.Sequential(
                    nn.Conv2d(latent_dims[i], level_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(level_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(level_channels, level_channels)
                )
            else:
                # For other levels, combine current latent with upsampled features from deeper level
                next_level_channels = base_channels * (2 ** (i+1))
                
                decoder_block = nn.Sequential(
                    # Process current level's latent representation
                    nn.Conv2d(latent_dims[i] + next_level_channels, level_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(level_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock(level_channels, level_channels)
                )
            
            # Add self-attention for deeper levels
            if i >= 1:
                decoder_block = nn.Sequential(
                    decoder_block,
                    SelfAttention(level_channels)
                )
            
            self.decoders.append(decoder_block)
        
        # Final output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * (2 ** i), base_channels * (2 ** (i-1)), 
                              kernel_size=4, stride=2, padding=1)
            for i in range(1, num_hierarchies)
        ])
    
    def forward(self, z_list):
        """
        Forward pass through hierarchical decoder
        
        Args:
            z_list: List of latent tensors for each hierarchy level (deepest to shallowest)
            
        Returns:
            x: Reconstructed image tensor
        """
        # Process hierarchy levels in reverse order (deepest first)
        x = None
        
        for i, z in enumerate(reversed(z_list)):
            level_idx = self.num_hierarchies - 1 - i
            
            if i == 0:
                # Deepest level - just process the latent
                x = self.decoders[i](z)
            else:
                # Upsample features from deeper level
                upsampled = self.upsample[i-1](x)
                
                # Concatenate with current level's latent
                combined = torch.cat([z, upsampled], dim=1)
                
                # Process through decoder block
                x = self.decoders[i](combined)
        
        # Final output convolution
        x = self.output_conv(x)
        
        return x


class Discriminator(nn.Module):
    """GAN Discriminator network"""
    
    def __init__(self, input_channels=3, base_channels=64, num_hierarchies=3):
        super(Discriminator, self).__init__()
        
        layers = []
        
        # Initial convolution
        layers.append(nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        curr_channels = base_channels
        for i in range(1, num_hierarchies):
            next_channels = base_channels * (2 ** i)
            
            layers.append(nn.Conv2d(curr_channels, next_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Add self-attention at intermediate layers
            if i >= 1:
                layers.append(SelfAttention(next_channels))
            
            curr_channels = next_channels
        
        # Additional layer for better feature extraction
        layers.append(nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(curr_channels * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # PatchGAN output - no need to reshape to a single value
        layers.append(nn.Conv2d(curr_channels * 2, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class HierarchicalVAEGAN(nn.Module):
    """Hierarchical VAE-GAN model for image compression"""
    
    def __init__(self, config):
        super(HierarchicalVAEGAN, self).__init__()
        
        # Get configuration parameters
        self.latent_dims = config.latent_dims
        self.num_hierarchies = config.num_hierarchies
        self.beta = config.beta
        self.level_weights = config.level_weights
        self.input_channels = config.channels
        self.base_channels = config.base_channels
        self.lambda_adv = config.lambda_adv
        
        # Ensure level weights match number of hierarchies
        if len(self.level_weights) != self.num_hierarchies:
            self.level_weights = [1.0] * self.num_hierarchies
        
        # Initialize components
        self.encoder = HierarchicalEncoder(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            latent_dims=self.latent_dims,
            num_hierarchies=self.num_hierarchies
        )
        
        self.decoder = HierarchicalDecoder(
            output_channels=self.input_channels,
            base_channels=self.base_channels,
            latent_dims=self.latent_dims,
            num_hierarchies=self.num_hierarchies
        )
        
        self.discriminator = Discriminator(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            num_hierarchies=self.num_hierarchies
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """Encode input to latent representation"""
        mu_list, logvar_list = self.encoder(x)
        
        # Apply reparameterization trick at each level
        z_list = [self.reparameterize(mu, logvar) for mu, logvar in zip(mu_list, logvar_list)]
        
        return z_list, mu_list, logvar_list
    
    def decode(self, z_list):
        """Decode latent representation to reconstruction"""
        return self.decoder(z_list)
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        # Encode
        z_list, mu_list, logvar_list = self.encode(x)
        
        # Decode
        recon_x = self.decode(z_list)
        
        return recon_x, z_list, mu_list, logvar_list
    
    def loss_function(self, original, recon_x, mu_list, logvar_list, disc_real, disc_fake, train_generator=True):
        """
        Calculate the loss for both VAE and GAN components
        
        Args:
            original: Original input image
            recon_x: Reconstructed image
            mu_list: List of mean tensors for each hierarchy level
            logvar_list: List of log variance tensors for each hierarchy level
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake (reconstructed) images
            train_generator: Whether to compute generator loss (True) or discriminator loss (False)
        
        Returns:
            Dictionary of loss components and total loss
        """
        # Reconstruction loss (pixel-wise)
        recon_loss = F.mse_loss(recon_x, original, reduction='mean')
        
        # KL Divergence loss for each hierarchy level
        kl_loss = 0.0
        for i, (mu, logvar, weight) in enumerate(zip(mu_list, logvar_list, self.level_weights)):
            kl_i = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss += weight * kl_i
        
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


def estimate_bits_per_pixel(z_list, image_size):
    """
    Calculate an estimate of bits per pixel for hierarchical model
    
    Args:
        z_list: List of latent tensors
        image_size: Size of the original image (assuming square)
    
    Returns:
        Bits per pixel value
    """
    # Sum up bits across all levels
    total_bits = 0
    for z in z_list:
        # Each element in the latent space is assumed to need 32 bits (float)
        level_bits = z.numel() * 32 / 8  # Convert to bytes
        total_bits += level_bits
    
    # Calculate total pixels (assuming RGB image)
    total_pixels = z_list[0].size(0) * image_size * image_size * 3
    
    # Calculate bits per pixel
    bpp = total_bits / total_pixels
    
    return bpp


def compress_hierarchical_model(model, image, bit_allocation=[8, 6, 4], device='cuda'):
    """
    Simulate compression of hierarchical model with quantization
    
    Args:
        model: Trained HierarchicalVAEGAN model
        image: Input image tensor
        bit_allocation: List specifying how many bits to use for each hierarchy level
        device: Device to run the model on
        
    Returns:
        Compressed data size, metrics, and reconstructed image
    """
    model.eval()
    image = image.to(device)
    
    # Encode image
    with torch.no_grad():
        z_list, mu_list, logvar_list = model.encode(image)
    
    # Quantize each latent representation
    quantized_z_list = []
    compression_bytes = 0
    
    for i, (z, bits) in enumerate(zip(z_list, bit_allocation)):
        # Scale to appropriate range for quantization based on bits
        min_val = float(z.min())
        max_val = float(z.max())
        scale = (2 ** bits) - 1
        
        # Quantize
        z_scaled = (z - min_val) / (max_val - min_val)
        z_quantized = torch.round(z_scaled * scale) / scale
        z_dequantized = z_quantized * (max_val - min_val) + min_val
        
        quantized_z_list.append(z_dequantized)
        
        # Calculate compressed size - each value needs 'bits' bits
        level_bytes = z.numel() * bits / 8
        compression_bytes += level_bytes
        
        # Also need to store min and max values (32 bits each)
        compression_bytes += 8  # 2 floats * 4 bytes
    
    # Calculate bits per pixel
    total_pixels = image.size(0) * image.size(2) * image.size(3) * image.size(1)  # batch * height * width * channels
    bpp = (compression_bytes * 8) / total_pixels
    
    # Calculate compression ratio
    original_bytes = total_pixels * 3  # 3 bytes per pixel (assuming 8-bit RGB)
    compression_ratio = original_bytes / compression_bytes
    
    # Decode quantized latent representations
    with torch.no_grad():
        reconstructed = model.decode(quantized_z_list)
    
    # Return metrics
    metrics = {
        'bpp': bpp,
        'compression_ratio': compression_ratio,
        'compressed_bytes': compression_bytes
    }
    
    return metrics, reconstructed, quantized_z_list