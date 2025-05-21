"""
Evaluation script for Hierarchical VAE-GAN model
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
from model import HierarchicalVAEGAN, compress_hierarchical_model
from data import get_kodak_dataset, set_seed
from utils import (
    calculate_psnr, calculate_ssim, calculate_lpips,
    save_reconstructed_images, prepare_fid_calculation, calculate_fid
)


def evaluate_model(model, dataloader, config, save_dir, bit_allocation=None):
    """Evaluate the model on a dataset and save results"""
    
    device = torch.device(config.device)
    model.eval()
    
    # If bit allocation not provided, use default
    if bit_allocation is None:
        bit_allocation = [8, 6, 4]  # Default bit allocation for 3 levels
    
    # Ensure bit allocation matches number of hierarchies
    if len(bit_allocation) != config.num_hierarchies:
        bit_allocation = [8] * config.num_hierarchies
    
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
    temp_real_dir = os.path.join(tempfile.gettempdir(), 'hierarchical_vae_gan_real_eval')
    temp_fake_dir = os.path.join(tempfile.gettempdir(), 'hierarchical_vae_gan_fake_eval')
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
            recon_images, z_list, _, _ = model(images)
            
            # Simulate compression with bit allocation
            compression_metrics, compressed_images, quantized_z_list = compress_hierarchical_model(
                model, images, bit_allocation=bit_allocation, device=device
            )
            
            # Get compression metrics
            bpp = compression_metrics['bpp']
            compression_ratio = compression_metrics['compression_ratio']
            
            # Convert to CPU numpy arrays for metric calculation
            original_np = images.cpu().numpy()
            recon_np = compressed_images.cpu().numpy()
            
            # Calculate metrics
            psnr = calculate_psnr(original_np, recon_np)
            ssim = calculate_ssim(original_np, recon_np)
            lpips_value = calculate_lpips(images, compressed_images, device=device)
            
            # Store metrics
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['lpips'].append(lpips_value)
            metrics['bpp'].append(bpp)
            metrics['compression_ratio'].append(compression_ratio)
            
            # Save reconstructed images
            save_reconstructed_images(
                images, compressed_images,
                os.path.join(save_dir, 'reconstructions', f"{image_name}_compressed.png")
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
                f.write(f"Bit Allocation: {bit_allocation}\n\n")
                
                # Add details about each hierarchy level
                f.write("Hierarchy Level Details:\n")
                for j, z in enumerate(quantized_z_list):
                    level_size = z.numel() * bit_allocation[j] / 8  # in bytes
                    level_bpp = (level_size * 8) / (images.size(0) * images.size(2) * images.size(3) * 3)
                    f.write(f"  Level {j}: Shape {tuple(z.shape)}, Bits: {bit_allocation[j]}, ")
                    f.write(f"Size: {level_size:.1f} bytes, BPP: {level_bpp:.4f}\n")
    
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
        f.write("Hierarchical VAE-GAN Evaluation Summary\n")
        f.write("======================================\n\n")
        f.write(f"Mean PSNR: {metrics['mean_psnr']:.2f} dB\n")
        f.write(f"Mean SSIM: {metrics['mean_ssim']:.4f}\n")
        f.write(f"Mean LPIPS: {metrics['mean_lpips']:.4f}\n")
        f.write(f"Mean BPP: {metrics['mean_bpp']:.4f}\n")
        f.write(f"Mean Compression Ratio: {metrics['mean_compression_ratio']:.2f}x\n")
        f.write(f"FID: {metrics['fid']:.2f}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Number of hierarchies: {config.num_hierarchies}\n")
        f.write(f"  Latent dimensions: {config.latent_dims}\n")
        f.write(f"  Bit allocation: {bit_allocation}\n")
    
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


def compare_bit_allocations(model, dataloader, config, save_dir):
    """Compare different bit allocations for the hierarchical model"""
    
    # Create directory for bit allocation comparison
    bit_alloc_dir = os.path.join(save_dir, 'bit_allocation_comparison')
    os.makedirs(bit_alloc_dir, exist_ok=True)
    
    # Define different bit allocations to test
    # Format: [bits for level 1, bits for level 2, bits for level 3]
    bit_allocations = [
        [8, 6, 4],    # Default - more bits for higher levels
        [6, 6, 6],    # Equal allocation
        [4, 6, 8],    # More bits for deeper levels
        [10, 5, 2],   # Heavy focus on highest level
        [2, 5, 10],   # Heavy focus on deepest level
        [9, 4, 1],    # Extreme focus on highest level
    ]
    
    # Evaluate each bit allocation
    results = []
    
    for bits in bit_allocations:
        print(f"Evaluating bit allocation: {bits}")
        
        # Create directory for this bit allocation
        bits_str = "_".join(map(str, bits))
        bits_dir = os.path.join(bit_alloc_dir, f"bits_{bits_str}")
        os.makedirs(bits_dir, exist_ok=True)
        
        # Evaluate model with this bit allocation
        metrics = evaluate_model(model, dataloader, config, bits_dir, bit_allocation=bits)
        
        # Store result
        results.append({
            'bit_allocation': bits,
            'psnr': metrics['mean_psnr'],
            'ssim': metrics['mean_ssim'],
            'lpips': metrics['mean_lpips'],
            'bpp': metrics['mean_bpp'],
            'compression_ratio': metrics['mean_compression_ratio'],
            'fid': metrics['fid']
        })
    
    # Create comparison plots
    # Rate-distortion curves
    plt.figure(figsize=(12, 8))
    for result in results:
        bits_str = "_".join(map(str, result['bit_allocation']))
        plt.scatter(result['bpp'], result['psnr'], label=f"Bits: {bits_str}", s=100)
    
    plt.xlabel('BPP (Bits per Pixel)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Performance for Different Bit Allocations')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(bit_alloc_dir, 'rate_distortion_comparison.png'), dpi=300)
    
    # PSNR-LPIPS comparison
    plt.figure(figsize=(12, 8))
    for result in results:
        bits_str = "_".join(map(str, result['bit_allocation']))
        plt.scatter(result['psnr'], result['lpips'], label=f"Bits: {bits_str}", s=100)
    
    plt.xlabel('PSNR (dB)')
    plt.ylabel('LPIPS (lower is better)')
    plt.title('Quality vs Perceptual Similarity for Different Bit Allocations')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(bit_alloc_dir, 'psnr_lpips_comparison.png'), dpi=300)
    
    # Save comparison results
    with open(os.path.join(bit_alloc_dir, 'comparison_results.txt'), 'w') as f:
        f.write("Bit Allocation Comparison Results\n")
        f.write("================================\n\n")
        
        for result in results:
            bits_str = "_".join(map(str, result['bit_allocation']))
            f.write(f"Bit Allocation: {result['bit_allocation']}\n")
            f.write(f"  PSNR:  {result['psnr']:.2f} dB\n")
            f.write(f"  SSIM:  {result['ssim']:.4f}\n")
            f.write(f"  LPIPS: {result['lpips']:.4f}\n")
            f.write(f"  BPP:   {result['bpp']:.4f}\n")
            f.write(f"  Compression Ratio: {result['compression_ratio']:.2f}x\n")
            f.write(f"  FID:   {result['fid']:.2f}\n\n")
    
    return results


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical VAE-GAN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./eval_results/hierarchical_vae_gan',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--num_hierarchies', type=int, default=3,
                        help='Number of hierarchical levels')
    parser.add_argument('--compare_bits', action='store_true',
                        help='Compare different bit allocations')
    parser.add_argument('--bit_allocation', type=str, default=None,
                        help='Bit allocation for each level (comma-separated, e.g., "8,6,4")')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.num_hierarchies = args.num_hierarchies
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get evaluation dataset (Kodak)
    eval_loader = get_kodak_dataset(config)
    
    # Initialize model
    model = HierarchicalVAEGAN(config)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.device)))
    model = model.to(torch.device(config.device))
    
    # Parse bit allocation if provided
    bit_allocation = None
    if args.bit_allocation:
        bit_allocation = list(map(int, args.bit_allocation.split(',')))
        
        # Check if bit allocation matches number of hierarchies
        if len(bit_allocation) != config.num_hierarchies:
            print(f"Warning: Bit allocation length ({len(bit_allocation)}) doesn't match number of hierarchies ({config.num_hierarchies})")
            print("Using default bit allocation instead")
            bit_allocation = None
    
    # Evaluate model
    if args.compare_bits:
        print("Comparing different bit allocations...")
        results = compare_bit_allocations(model, eval_loader, config, args.output_dir)
    else:
        print(f"Evaluating model with bit allocation: {bit_allocation or [8, 6, 4]}")
        metrics = evaluate_model(model, eval_loader, config, args.output_dir, bit_allocation=bit_allocation)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()