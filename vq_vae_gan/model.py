"""
VQ-VAE-GAN model architecture
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


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE
    
    Inputs:
        - num_embeddings: size of the embedding dictionary (codebook size)
        - embedding_dim: dimensionality of each embedding vector
        - commitment_cost: controls the weighting of the commitment loss
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
        # Exponential moving average (EMA) update parameters
        self.decay = decay
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embeddings.weight.data.clone())
    
    def forward(self, inputs, training=True):
        """
        Inputs:
            - inputs: input tensor of shape [batch_size, embedding_dim, height, width]
            - training: whether the model is in training mode (for EMA updates)
            
        Returns:
            - quantized: quantized version of the input
            - loss: VQ loss
            - encoding_indices: indices of the embeddings in the codebook
        """
        # Reshape to [batch_size, embedding_dim, height*width]
        input_shape = inputs.shape
        batch_size, channels, height, width = input_shape
        
        # Flatten input
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat_input = flat_input.view(-1, self.embedding_dim)  # [B*H*W, C]
        
        # Calculate distances to embeddings
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        
        # One-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = self.embeddings(encoding_indices).view(batch_size, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Calculate VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Use EMA to update the embedding vectors
        if training:
            with torch.no_grad():
                # Cluster size EMA update
                encodings_sum = encodings.sum(0)
                self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings_sum
                
                # Embedding EMA update
                encodings_sum = encodings_sum.unsqueeze(1)
                embedded = flat_input.unsqueeze(1) * encodings.unsqueeze(2)
                embedded_sum = embedded.sum(0)
                
                # Update embeddings
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * embedded_sum
                
                # Normalize embeddings
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                
                # Update embedding weights with EMA
                self.embeddings.weight.data = self.ema_w / cluster_size.unsqueeze(1)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.view(batch_size, height, width)


class Encoder(nn.Module):
    """VQ-VAE Encoder network"""
    
    def __init__(self, input_channels=3, hidden_channels=[64, 128, 256, 512], embedding_dim=64):
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
        
        # Final convolution to get the embedding dim
        layers.append(nn.Conv2d(hidden_channels[-1], embedding_dim, kernel_size=1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """VQ-VAE Decoder network"""
    
    def __init__(self, embedding_dim=64, hidden_channels=[512, 256, 128, 64], output_channels=3):
        super(Decoder, self).__init__()
        
        # Build decoder with upsampling blocks
        layers = []
        
        # Initial convolution to expand from embedding_dim to hidden_channels[0]
        layers.append(nn.Conv2d(embedding_dim, hidden_channels[0], kernel_size=3, padding=1))
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
    
    def forward(self, x):
        return self.decoder(x)


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
        
        # PatchGAN output
        layers.append(nn.Conv2d(hidden_channels[-1], 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class VQVAEGAN(nn.Module):
    """VQ-VAE-GAN model for image compression"""
    
    def __init__(self, config):
        super(VQVAEGAN, self).__init__()
        
        # Get configuration parameters
        self.embedding_dim = config.embedding_dim
        self.num_embeddings = config.num_embeddings
        self.commitment_cost = config.commitment_cost
        self.decay = config.decay
        self.input_channels = config.channels
        self.enc_channels = config.enc_channels
        self.dec_channels = config.dec_channels
        self.lambda_adv = config.lambda_adv
        
        # Initialize components
        self.encoder = Encoder(
            input_channels=self.input_channels,
            hidden_channels=self.enc_channels,
            embedding_dim=self.embedding_dim
        )
        
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
            decay=self.decay
        )
        
        self.decoder = Decoder(
            embedding_dim=self.embedding_dim,
            hidden_channels=self.dec_channels,
            output_channels=self.input_channels
        )
        
        self.discriminator = Discriminator(
            input_channels=self.input_channels,
            hidden_channels=self.enc_channels
        )
    
    def encode(self, x, training=True):
        """Encode input to quantized latent representation"""
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vector_quantizer(z, training=training)
        return z_q, z, vq_loss, indices
    
    def decode(self, z_q):
        """Decode quantized latent representation to reconstruction"""
        return self.decoder(z_q)
    
    def forward(self, x, training=True):
        """Forward pass through encoder, VQ, and decoder"""
        # Encode and quantize
        z_q, z, vq_loss, indices = self.encode(x, training=training)
        
        # Decode
        recon_x = self.decode(z_q)
        
        return recon_x, z, z_q, vq_loss, indices
    
    def loss_function(self, original, recon_x, vq_loss, disc_real, disc_fake, train_generator=True):
        """
        Calculate the loss for both VAE and GAN components
        
        Args:
            original: Original input image
            recon_x: Reconstructed image
            vq_loss: Vector quantization loss
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake (reconstructed) images
            train_generator: Whether to compute generator loss (True) or discriminator loss (False)
        
        Returns:
            Dictionary of loss components and total loss
        """
        # Reconstruction loss (pixel-wise)
        recon_loss = F.mse_loss(recon_x, original, reduction='mean')
        
        # GAN losses
        if train_generator:
            # Generator wants discriminator to predict reconstructions as real
            adv_loss = -torch.mean(disc_fake)
            
            # Total loss for generator (VQ-VAE + GAN)
            total_loss = recon_loss + vq_loss + self.lambda_adv * adv_loss
            
            return {
                'total': total_loss,
                'recon': recon_loss.item(),
                'vq': vq_loss.item(),
                'adv': adv_loss.item(),
                'disc': 0.0  # Not calculated when training generator
            }
        else:
            # Discriminator loss
            disc_loss = torch.mean(F.relu(1.0 - disc_real)) + torch.mean(F.relu(1.0 + disc_fake))
            
            return {
                'total': disc_loss,
                'recon': recon_loss.item(),
                'vq': vq_loss.item(),
                'adv': 0.0,  # Not calculated when training discriminator
                'disc': disc_loss.item()
            }


# Entropy coding implementation for VQ-VAE
class ArithmeticEncoder:
    """
    Simple arithmetic coder for entropy coding of VQ-VAE indices
    """
    def __init__(self, num_symbols, precision=32):
        """
        Initialize the arithmetic coder
        
        Args:
            num_symbols: Number of symbols in the codebook
            precision: Number of bits for internal arithmetic precision
        """
        self.num_symbols = num_symbols
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.half = 1 << (precision - 1)
        self.quarter = 1 << (precision - 2)
        
    def encode(self, indices, probs=None):
        """
        Encode indices using arithmetic coding
        
        Args:
            indices: Tensor of indices to encode
            probs: Optional probability distribution for the symbols
            
        Returns:
            Binary representation of the encoded indices
        """
        # Flatten indices
        flat_indices = indices.flatten().cpu().numpy().astype(int)
        
        # If probabilities not provided, use uniform distribution
        if probs is None:
            probs = torch.ones(self.num_symbols) / self.num_symbols
        else:
            probs = probs / probs.sum()  # Normalize
        
        probs = probs.cpu().numpy()
        
        # Calculate cumulative probabilities
        cum_probs = np.zeros(self.num_symbols + 1)
        for i in range(self.num_symbols):
            cum_probs[i+1] = cum_probs[i] + probs[i]
        
        # Initialize encoding range
        low = 0
        high = self.max_val
        
        # Encoded bits
        bits = []
        pending_bits = 0
        
        # Encode each symbol
        for symbol in flat_indices:
            # Calculate new range
            range_size = high - low + 1
            high = low + int(range_size * cum_probs[symbol+1]) - 1
            low = low + int(range_size * cum_probs[symbol])
            
            # Handle range reduction and bit output
            while True:
                if high < self.half:
                    # Output 0 and any pending bits
                    bits.append(0)
                    for _ in range(pending_bits):
                        bits.append(1)
                    pending_bits = 0
                elif low >= self.half:
                    # Output 1 and any pending bits
                    bits.append(1)
                    for _ in range(pending_bits):
                        bits.append(0)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    # Middle range - keep track of pending bits
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                
                # Scale up the range
                low = low * 2
                high = high * 2 + 1
        
        # Output final pending bits
        pending_bits += 1
        if low < self.quarter:
            bits.append(0)
            for _ in range(pending_bits):
                bits.append(1)
        else:
            bits.append(1)
            for _ in range(pending_bits):
                bits.append(0)
        
        # Convert bits to byte array
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits) and bits[i + j]:
                    byte |= 1 << (7 - j)
            byte_array.append(byte)
        
        return byte_array
    
    def decode(self, byte_array, shape, probs=None):
        """
        Decode byte array back to indices
        
        Args:
            byte_array: Bytes to decode
            shape: Shape of the original indices tensor
            probs: Optional probability distribution for the symbols
            
        Returns:
            Tensor of decoded indices
        """
        # If probabilities not provided, use uniform distribution
        if probs is None:
            probs = torch.ones(self.num_symbols) / self.num_symbols
        else:
            probs = probs / probs.sum()  # Normalize
        
        probs = probs.cpu().numpy()
        
        # Calculate cumulative probabilities
        cum_probs = np.zeros(self.num_symbols + 1)
        for i in range(self.num_symbols):
            cum_probs[i+1] = cum_probs[i] + probs[i]
        
        # Convert byte array to bits
        bits = []
        for byte in byte_array:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        # Initialize decoder state
        value = 0
        for i in range(self.precision):
            if i < len(bits):
                value = (value << 1) | bits[i]
        
        low = 0
        high = self.max_val
        
        # Decode symbols
        num_elements = np.prod(shape)
        decoded_indices = np.zeros(num_elements, dtype=int)
        
        for i in range(num_elements):
            # Find symbol
            range_size = high - low + 1
            scaled_value = ((value - low + 1) * self.max_val) // range_size
            
            symbol = 0
            while symbol < self.num_symbols and cum_probs[symbol+1] * self.max_val <= scaled_value:
                symbol += 1
            
            decoded_indices[i] = symbol
            
            # Update range
            high = low + int(range_size * cum_probs[symbol+1]) - 1
            low = low + int(range_size * cum_probs[symbol])
            
            # Read next bit
            while True:
                if high < self.half:
                    # do nothing, bit is 0
                    pass
                elif low >= self.half:
                    # subtract half from both ends and from value
                    low -= self.half
                    high -= self.half
                    value -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    # near middle, subtract quarter from both ends and from value
                    low -= self.quarter
                    high -= self.quarter
                    value -= self.quarter
                else:
                    break
                
                # Scale up the range
                low = low * 2
                high = high * 2 + 1
                
                # Shift in next bit
                value = value * 2
                if i < len(bits):
                    value += bits[i]
                    i += 1
        
        # Reshape back to original tensor shape
        decoded_indices = torch.tensor(decoded_indices.reshape(shape))
        
        return decoded_indices


def calculate_codebook_usage(indices, num_embeddings):
    """
    Calculate the usage of codebook entries
    
    Args:
        indices: Tensor of indices from the encoder
        num_embeddings: Size of the codebook
    
    Returns:
        Tensor of probabilities for each codebook entry
    """
    flat_indices = indices.flatten()
    histogram = torch.histc(flat_indices.float(), bins=num_embeddings, min=0, max=num_embeddings-1)
    probs = histogram / histogram.sum()
    return probs

def compress_vq_indices(model, indices):
    """
    Compress indices using arithmetic coding
    
    Args:
        model: Trained VQ-VAE-GAN model
        indices: Tensor of indices from the encoder
        
    Returns:
        Compressed byte array and probabilities
    """
    # Calculate codebook usage probabilities
    probs = calculate_codebook_usage(indices, model.num_embeddings)
    
    # Create arithmetic encoder
    encoder = ArithmeticEncoder(model.num_embeddings)
    
    # Encode indices
    compressed = encoder.encode(indices, probs)
    
    return compressed, probs

def decompress_vq_indices(model, compressed, shape, probs=None):
    """
    Decompress indices using arithmetic coding
    
    Args:
        model: Trained VQ-VAE-GAN model
        compressed: Compressed byte array
        shape: Shape of the original indices tensor
        probs: Optional probability distribution for decoding
        
    Returns:
        Tensor of decompressed indices
    """
    # Create arithmetic encoder
    encoder = ArithmeticEncoder(model.num_embeddings)
    
    # Decode compressed data
    indices = encoder.decode(compressed, shape, probs)
    
    return indices

def compress_image(model, image, device='cuda'):
    """
    Compress an image using VQ-VAE-GAN
    
    Args:
        model: Trained VQ-VAE-GAN model
        image: Input image tensor
        device: Device to run the model on
        
    Returns:
        Compressed data and metadata
    """
    model.eval()
    image = image.to(device)
    
    # Encode image
    with torch.no_grad():
        _, _, _, indices = model.encode(image, training=False)
    
    # Compress indices
    compressed, probs = compress_vq_indices(model, indices)
    
    # Calculate compression ratio and bits per pixel
    original_size = image.numel() * 8  # 8 bits per channel value
    compressed_size = len(compressed) * 8  # 8 bits per byte
    
    compression_ratio = original_size / compressed_size
    bpp = compressed_size / (image.size(2) * image.size(3) * image.size(0))
    
    metadata = {
        'shape': indices.shape,
        'probs': probs,
        'compression_ratio': compression_ratio,
        'bpp': bpp
    }
    
    return compressed, metadata

def decompress_image(model, compressed, metadata, device='cuda'):
    """
    Decompress an image using VQ-VAE-GAN
    
    Args:
        model: Trained VQ-VAE-GAN model
        compressed: Compressed data
        metadata: Metadata from compression
        device: Device to run the model on
        
    Returns:
        Reconstructed image
    """
    model.eval()
    
    # Get metadata
    shape = metadata['shape']
    probs = metadata['probs']
    
    # Decompress indices
    indices = decompress_vq_indices(model, compressed, shape, probs)
    indices = indices.to(device)
    
    # Convert indices to embeddings
    with torch.no_grad():
        # Look up embeddings directly
        embeddings = model.vector_quantizer.embeddings(indices.view(-1))
        
        # Reshape to original encoded shape
        batch_size, height, width = shape
        z_q = embeddings.view(batch_size, height, width, model.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Decode embeddings
        reconstructed = model.decode(z_q)
    
    return reconstructed