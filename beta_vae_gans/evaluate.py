"""
Evaluation script for β-VAE-GAN model
"""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import tempfile
import shutil

from config import Config
from model import BetaVAEGAN
from data import get_kodak_dataset, set_seed
from utils import (
    calculate_bpp, calculate_psnr, calculate_ssim, calculate_lpips,
    save_reconstructed_images, prepare_fid_calculation, calculate_fid
)


def evaluate_model(model, dataloader, config, save_dir):
    """Evaluate the model on a dataset and save results"""
    
    device = torch.device(config.device)
    model.eval()
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'reconstructions'), exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'psnr': [], 'ssim': [], 'lpips': [], 'bpp': [],
        'image_names': []
    }
    
    # Create temporary directories for FID calculation
    temp_real_dir = os.path.join(tempfile.gettempdir(), 'vae_gan_real_eval')
    temp_fake_dir = os.path.join(tempfile.gettempdir(), 'vae_gan_fake_eval')
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_fake_dir, exist_ok=True)
    
    # Clear temp directories
    for d in [temp_real_dir, temp_fake_dir]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    
    # Iterate over dataset
    with torch.no_grad():
        for i, images in enumerate(tqdm(dataloader, desc="Evaluating")):
            image_name = f"image_{i:04d}"
            metrics['image_names'].append(image_name)
            
            images = images.to(device)
            
            # Get reconstructions
            recon_images, _, mu, logvar = model(images)
            
            # Convert to CPU numpy arrays for metric calculation
            original_np = images.cpu().numpy()
            recon_np = recon_images.cpu().numpy()
            
            # Calculate metrics
            psnr = calculate_psnr(original_np, recon_np)
            ssim = calculate_ssim(original_np, recon_np)
            lpips_value = calculate_lpips(images, recon_images, device=device)
            bpp = calculate_bpp(mu, config.image_size)
            
            # Store metrics
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['lpips'].append(lpips_value)
            metrics['bpp'].append(bpp)
            
            # Save reconstructed images
            save_reconstructed_images(
                images, recon_images,
                os.path.join(save_dir, 'reconstructions', f"{image_name}.png")
            )
            
            # Prepare images for FID calculation
            prepare_fid_calculation(
                original_np, recon_np,
                temp_real_dir, temp_fake_dir
            )
    
    # Calculate mean metrics
    for key in ['psnr', 'ssim', 'lpips', 'bpp']:
        metrics[f"mean_{key}"] = np.mean(metrics[key])
    
    # Calculate FID
    metrics['fid'] = calculate_fid(temp_real_dir, temp_fake_dir, device=device)
    
    # Save metrics
    np.save(os.path.join(save_dir, 'metrics.npy'), metrics)
    
    # Create rate-distortion plot
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['bpp'], metrics['psnr'], alpha=0.7)
    plt.xlabel('BPP (Bits per Pixel)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'Rate-Distortion Performance\nMean PSNR: {metrics["mean_psnr"]:.2f} dB, Mean BPP: {metrics["mean_bpp"]:.4f}')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rate_distortion.png'), dpi=300)
    
    # Clean up temporary directories
    shutil.rmtree(temp_real_dir, ignore_errors=True)
    shutil.rmtree(temp_fake_dir, ignore_errors=True)
    
    # Print results
    print(f"Evaluation Results:")
    print(f"  PSNR:  {metrics['mean_psnr']:.2f} dB")
    print(f"  SSIM:  {metrics['mean_ssim']:.4f}")
    print(f"  LPIPS: {metrics['mean_lpips']:.4f}")
    print(f"  FID:   {metrics['fid']:.2f}")
    print(f"  BPP:   {metrics['mean_bpp']:.4f}")
    
    return metrics


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Evaluate β-VAE-GAN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./eval_results/beta_vae_gan',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL divergence weight (β parameter)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.beta = args.beta
    config.latent_dim = args.latent_dim
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get evaluation dataset (Kodak)
    eval_loader = get_kodak_dataset(config)
    
    # Initialize model
    model = BetaVAEGAN(config)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.device)))
    model = model.to(torch.device(config.device))
    
    # Evaluate model
    metrics = evaluate_model(model, eval_loader, config, args.output_dir)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()