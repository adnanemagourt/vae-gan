"""
Utility functions for VAE-GAN compression models
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from pytorch_fid import fid_score
import math
from torch.utils.tensorboard import SummaryWriter

# Initialize LPIPS model
lpips_fn = lpips.LPIPS(net='alex').cuda()

def calculate_bpp(latent_codes, image_size):
    """
    Calculate bits per pixel (bpp) - theoretical approximation
    
    Args:
        latent_codes: Tensor of latent codes
        image_size: Size of the image (assuming square)
    
    Returns:
        Bits per pixel value
    """
    batch_size = latent_codes.size(0)
    # Assume we use 32 bits (float) per latent dimension
    # For VAE, we need to estimate the entropy of the codes
    # Here, we use a simple approximation based on the KL divergence
    total_bits = latent_codes.numel() * 32 / 8  # in bytes
    pixels = batch_size * image_size * image_size * 3  # RGB images
    bpp = total_bits / pixels
    
    return bpp

def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy().transpose(0, 2, 3, 1)
        reconstructed = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Ensure values are in [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    psnr_values = []
    for i in range(original.shape[0]):
        psnr_values.append(peak_signal_noise_ratio(original[i], reconstructed[i], data_range=1.0))
    
    return np.mean(psnr_values)

def calculate_ssim(original, reconstructed):
    """Calculate Structural Similarity Index"""
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy().transpose(0, 2, 3, 1)
        reconstructed = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Ensure values are in [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    ssim_values = []
    for i in range(original.shape[0]):
        ssim_values.append(structural_similarity(original[i], reconstructed[i], data_range=1.0, multichannel=True))
    
    return np.mean(ssim_values)

def calculate_lpips(original, reconstructed, device='cuda'):
    """Calculate LPIPS perceptual similarity"""
    # Ensure input is tensor and on the right device
    if not isinstance(original, torch.Tensor):
        original = torch.from_numpy(original).permute(0, 3, 1, 2).to(device)
        reconstructed = torch.from_numpy(reconstructed).permute(0, 3, 1, 2).to(device)
    else:
        original = original.to(device)
        reconstructed = reconstructed.to(device)
    
    # LPIPS expects values in [-1, 1]
    original = original * 2 - 1
    reconstructed = reconstructed * 2 - 1
    
    with torch.no_grad():
        lpips_values = lpips_fn(original, reconstructed)
    
    return lpips_values.mean().item()

def save_reconstructed_images(original, reconstructed, filepath, nrow=4):
    """Save grid of original and reconstructed images"""
    # Ensure both are tensors
    if not isinstance(original, torch.Tensor):
        original = torch.from_numpy(original).permute(0, 3, 1, 2)
        reconstructed = torch.from_numpy(reconstructed).permute(0, 3, 1, 2)
    
    # Create a grid
    comparison = torch.cat([original, reconstructed], dim=0)
    grid = make_grid(comparison, nrow=nrow, normalize=True, value_range=(0, 1))
    
    # Save image
    save_image(grid, filepath)
    
    return grid

def prepare_fid_calculation(original_batch, reconstructed_batch, temp_real_dir, temp_fake_dir):
    """
    Prepare images for FID calculation by saving real and reconstructed images to temporary dirs
    """
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_fake_dir, exist_ok=True)
    
    # Convert to numpy if they are tensors
    if isinstance(original_batch, torch.Tensor):
        original_batch = original_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
        reconstructed_batch = reconstructed_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Save images to temporary folders
    for i in range(original_batch.shape[0]):
        # Convert to PIL Image and save
        orig_img = (np.clip(original_batch[i], 0, 1) * 255).astype(np.uint8)
        recon_img = (np.clip(reconstructed_batch[i], 0, 1) * 255).astype(np.uint8)
        
        plt.imsave(os.path.join(temp_real_dir, f"{i:04d}.png"), orig_img)
        plt.imsave(os.path.join(temp_fake_dir, f"{i:04d}.png"), recon_img)

def calculate_fid(real_dir, fake_dir, device='cuda'):
    """
    Calculate Fréchet Inception Distance between real and fake images
    """
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=device, dims=2048)

class MetricLogger:
    """Class to log and display training metrics"""
    
    def __init__(self, log_dir, config):
        """
        Initialize the metric logger
        
        Args:
            log_dir: Directory to save logs
            config: Configuration object
        """
        self.log_dir = log_dir
        self.config = config
        self.writer = SummaryWriter(log_dir)
        
        # Create directories
        os.makedirs(os.path.join(log_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize metric dictionaries
        self.train_metrics = {
            'loss': [], 'recon_loss': [], 'kl_loss': [], 'adv_loss': [],
            'disc_loss': [], 'bpp': []
        }
        
        self.eval_metrics = {
            'psnr': [], 'ssim': [], 'lpips': [], 'fid': [], 'bpp': []
        }
    
    def log_train_metrics(self, epoch, iteration, loss_dict, bpp, images=None, reconstructed=None):
        """Log training metrics"""
        # Update internal metrics
        self.train_metrics['loss'].append(loss_dict['total'])
        self.train_metrics['recon_loss'].append(loss_dict['recon'])
        self.train_metrics['kl_loss'].append(loss_dict['kl'])
        self.train_metrics['adv_loss'].append(loss_dict['adv'])
        self.train_metrics['disc_loss'].append(loss_dict['disc'])
        self.train_metrics['bpp'].append(bpp)
        
        # Log to tensorboard
        total_iter = epoch * len(self.train_metrics['loss']) + iteration
        self.writer.add_scalar('train/total_loss', loss_dict['total'], total_iter)
        self.writer.add_scalar('train/reconstruction_loss', loss_dict['recon'], total_iter)
        self.writer.add_scalar('train/kl_loss', loss_dict['kl'], total_iter)
        self.writer.add_scalar('train/adversarial_loss', loss_dict['adv'], total_iter)
        self.writer.add_scalar('train/discriminator_loss', loss_dict['disc'], total_iter)
        self.writer.add_scalar('train/bits_per_pixel', bpp, total_iter)
        
        # Log images if provided
        if images is not None and reconstructed is not None:
            if total_iter % (self.config.log_interval * 5) == 0:
                grid = save_reconstructed_images(
                    images[:4], reconstructed[:4],
                    os.path.join(self.log_dir, 'images', f'train_recon_epoch{epoch}_iter{iteration}.png')
                )
                self.writer.add_image('train/reconstructions', grid, total_iter)
    
    def log_eval_metrics(self, epoch, psnr, ssim, lpips_value, fid, bpp, images=None, reconstructed=None):
        """Log evaluation metrics"""
        # Update internal metrics
        self.eval_metrics['psnr'].append(psnr)
        self.eval_metrics['ssim'].append(ssim)
        self.eval_metrics['lpips'].append(lpips_value)
        self.eval_metrics['fid'].append(fid)
        self.eval_metrics['bpp'].append(bpp)
        
        # Log to tensorboard
        self.writer.add_scalar('eval/psnr', psnr, epoch)
        self.writer.add_scalar('eval/ssim', ssim, epoch)
        self.writer.add_scalar('eval/lpips', lpips_value, epoch)
        self.writer.add_scalar('eval/fid', fid, epoch)
        self.writer.add_scalar('eval/bits_per_pixel', bpp, epoch)
        
        # Log images if provided
        if images is not None and reconstructed is not None:
            grid = save_reconstructed_images(
                images[:4], reconstructed[:4],
                os.path.join(self.log_dir, 'images', f'eval_recon_epoch{epoch}.png')
            )
            self.writer.add_image('eval/reconstructions', grid, epoch)
    
    def save_metrics(self):
        """Save metrics to disk"""
        np.save(os.path.join(self.log_dir, 'train_metrics.npy'), self.train_metrics)
        np.save(os.path.join(self.log_dir, 'eval_metrics.npy'), self.eval_metrics)
    
    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()