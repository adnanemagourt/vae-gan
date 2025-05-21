"""
Evaluation script for VQ-VAE-GAN model
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
from model import VQVAEGAN, compress_image, decompress_image
from data import get_kodak_dataset, set_seed
from utils import (
    calculate_psnr, calculate_ssim, calculate_lpips,
    save_reconstructed_images, prepare_fid_calculation, calculate_fid
)


def evaluate_model(model, dataloader, config, save_dir):
    """Evaluate the model on a dataset and save results"""
    
    device = torch.device(config.device)
    model.eval()
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'reconstructions'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'compression_results'), exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'psnr': [], 'ssim': [], 'lpips': [], 'bpp': [], 'compression_ratio': [],
        'image_names': []
    }
    
    # Create temporary directories for FID calculation
    temp_real_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_real_eval')
    temp_fake_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_fake_eval')
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
            
            # Get reconstructions using the normal forward pass
            recon_images, _, _, _, _ = model(images, training=False)
            
            # Compress and decompress to get actual reconstruction
            compressed, metadata = compress_image(model, images, device)
            decompressed_images = decompress_image(model, compressed, metadata, device)
            
            # Get compression metrics
            bpp = metadata['bpp']
            compression_ratio = metadata['compression_ratio']
            
            # Convert to CPU numpy arrays for metric calculation
            original_np = images.cpu().numpy()
            recon_np = decompressed_images.cpu().numpy()
            
            # Calculate metrics
            psnr = calculate_psnr(original_np, recon_np)
            ssim = calculate_ssim(original_np, recon_np)
            lpips_value = calculate_lpips(images, decompressed_images, device=device)
            
            # Store metrics
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['lpips'].append(lpips_value)
            metrics['bpp'].append(bpp)
            metrics['compression_ratio'].append(compression_ratio)
            
            # Save reconstructed images
            save_reconstructed_images(
                images, decompressed_images,
                os.path.join(save_dir, 'reconstructions', f"{image_name}.png")
            )
            
            # Save reconstruction with actual model output (no compression)
            save_reconstructed_images(
                images, recon_images,
                os.path.join(save_dir, 'reconstructions', f"{image_name}_direct.png")
            )
            
            # Prepare images for FID calculation
            prepare_fid_calculation(
                original_np, recon_np,
                temp_real_dir, temp_fake_dir
            )
            
            # Save compression details
            with open(os.path.join(save_dir, 'compression_results', f"{image_name}_details.txt"), 'w') as f:
                f.write(f"Image: {image_name}\n")
                f.write(f"PSNR: {psnr:.2f} dB\n")
                f.write(f"SSIM: {ssim:.4f}\n")
                f.write(f"LPIPS: {lpips_value:.4f}\n")
                f.write(f"BPP: {bpp:.4f}\n")
                f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
                
                # Add codebook usage statistics
                f.write("\nCodebook Usage:\n")
                codebook_usage = calculate_codebook_usage(model, images)
                for i, (idx, count, percent) in enumerate(codebook_usage):
                    if i < 20 or percent > 0.5:  # Show top 20 or any with >0.5%
                        f.write(f"  Code {idx}: {count} occurrences ({percent:.2f}%)\n")
    
    # Calculate mean metrics
    for key in ['psnr', 'ssim', 'lpips', 'bpp', 'compression_ratio']:
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
    
    # Create PSNR-LPIPS plot (quality vs perceptual similarity)
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['psnr'], metrics['lpips'], alpha=0.7)
    plt.xlabel('PSNR (dB)')
    plt.ylabel('LPIPS (lower is better)')
    plt.title('Quality vs Perceptual Similarity')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'psnr_lpips.png'), dpi=300)
    
    # Create summary file
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write("VQ-VAE-GAN Evaluation Summary\n")
        f.write("============================\n\n")
        f.write(f"Mean PSNR: {metrics['mean_psnr']:.2f} dB\n")
        f.write(f"Mean SSIM: {metrics['mean_ssim']:.4f}\n")
        f.write(f"Mean LPIPS: {metrics['mean_lpips']:.4f}\n")
        f.write(f"Mean BPP: {metrics['mean_bpp']:.4f}\n")
        f.write(f"Mean Compression Ratio: {metrics['mean_compression_ratio']:.2f}x\n")
        f.write(f"FID: {metrics['fid']:.2f}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Embedding dimension: {config.embedding_dim}\n")
        f.write(f"  Codebook size: {config.num_embeddings}\n")
        f.write(f"  Commitment cost: {config.commitment_cost}\n")
    
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
    print(f"  Compression Ratio: {metrics['mean_compression_ratio']:.2f}x")
    
    return metrics


def calculate_codebook_usage(model, images):
    """Calculate and sort codebook usage statistics"""
    
    # Encode images
    with torch.no_grad():
        _, _, _, _, indices = model(images, training=False)
    
    # Count occurrences of each index
    flat_indices = indices.flatten().cpu().numpy()
    unique, counts = np.unique(flat_indices, return_counts=True)
    
    # Calculate percentage
    total = flat_indices.size
    percentages = (counts / total) * 100
    
    # Sort by count (descending)
    sorted_idx = np.argsort(-counts)
    
    # Return sorted list of (index, count, percentage)
    result = [(unique[i], counts[i], percentages[i]) for i in sorted_idx]
    
    return result


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE-GAN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./eval_results/vq_vae_gan',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_embeddings', type=int, default=512,
                        help='Number of embeddings (codebook size)')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost for VQ-VAE')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.embedding_dim = args.embedding_dim
    config.num_embeddings = args.num_embeddings
    config.commitment_cost = args.commitment_cost
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get evaluation dataset (Kodak)
    eval_loader = get_kodak_dataset(config)
    
    # Initialize model
    model = VQVAEGAN(config)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.device)))
    model = model.to(torch.device(config.device))
    
    # Evaluate model
    metrics = evaluate_model(model, eval_loader, config, args.output_dir)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()