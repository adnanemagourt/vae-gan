"""
Script to compare different VAE-GAN architectures for image compression
"""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import tempfile
import shutil
import json

# Import models
from beta_vae_gan.model import BetaVAEGAN
from beta_vae_gan.config import Config as BetaConfig
from vq_vae_gan.model import VQVAEGAN, compress_image
from vq_vae_gan.config import Config as VQConfig
from hierarchical_vae_gan.model import HierarchicalVAEGAN, compress_hierarchical_model
from hierarchical_vae_gan.config import Config as HierarchicalConfig

# Import utilities
from beta_vae_gan.data import get_kodak_dataset, set_seed
from beta_vae_gan.utils import (
    calculate_psnr, calculate_ssim, calculate_lpips,
    save_reconstructed_images, prepare_fid_calculation, calculate_fid
)


def load_model(model_type, model_path, device='cuda'):
    """
    Load a trained model based on its type
    
    Args:
        model_type: Type of model ('beta', 'vq', or 'hierarchical')
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model and its configuration
    """
    if model_type == 'beta':
        config = BetaConfig()
        model = BetaVAEGAN(config)
    elif model_type == 'vq':
        config = VQConfig()
        model = VQVAEGAN(config)
    elif model_type == 'hierarchical':
        config = HierarchicalConfig()
        model = HierarchicalVAEGAN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, config


def compress_and_reconstruct(model, model_type, image, device='cuda'):
    """
    Compress and reconstruct an image based on the model type
    
    Args:
        model: Trained model
        model_type: Type of model ('beta', 'vq', or 'hierarchical')
        image: Input image tensor
        device: Device to run the model on
        
    Returns:
        Dictionary containing reconstructed image, bits per pixel, and compression ratio
    """
    with torch.no_grad():
        if model_type == 'beta':
            # For beta-VAE-GAN, simply run forward pass
            recon_image, z, mu, logvar = model(image)
            
            # Calculate BPP based on latent dimension
            num_elements = z.numel()
            bits_per_element = 32  # assuming float32
            total_bits = num_elements * bits_per_element
            total_pixels = image.size(0) * image.size(2) * image.size(3) * 3  # batch * height * width * channels
            bpp = total_bits / total_pixels
            
            # Calculate compression ratio
            original_size = image.numel() * 8  # 8 bits per channel value
            compressed_size = z.numel() * 32  # 32 bits per latent value
            compression_ratio = original_size / compressed_size
            
            result = {
                'reconstructed': recon_image,
                'bpp': bpp,
                'compression_ratio': compression_ratio
            }
            
        elif model_type == 'vq':
            # For VQ-VAE-GAN, use the compress_image function
            compressed, metadata = compress_image(model, image, device)
            reconstructed = model.decompress_image(compressed, metadata, device)
            
            result = {
                'reconstructed': reconstructed,
                'bpp': metadata['bpp'],
                'compression_ratio': metadata['compression_ratio']
            }
            
        elif model_type == 'hierarchical':
            # For Hierarchical VAE-GAN, use compress_hierarchical_model
            bit_allocation = [8, 6, 4]  # Default bit allocation
            metrics, reconstructed, _ = compress_hierarchical_model(
                model, image, bit_allocation=bit_allocation, device=device
            )
            
            result = {
                'reconstructed': reconstructed,
                'bpp': metrics['bpp'],
                'compression_ratio': metrics['compression_ratio']
            }
    
    return result


def evaluate_model_metrics(model, model_type, dataloader, device='cuda'):
    """
    Evaluate a model on a dataset and calculate metrics
    
    Args:
        model: Trained model
        model_type: Type of model ('beta', 'vq', or 'hierarchical')
        dataloader: DataLoader for the evaluation dataset
        device: Device to run the model on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    bpp_values = []
    compression_ratio_values = []
    
    # Create temporary directories for FID calculation
    temp_real_dir = os.path.join(tempfile.gettempdir(), f'compare_{model_type}_real')
    temp_fake_dir = os.path.join(tempfile.gettempdir(), f'compare_{model_type}_fake')
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_fake_dir, exist_ok=True)
    
    # Clear temp directories
    for d in [temp_real_dir, temp_fake_dir]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    
    # Evaluate on dataset
    with torch.no_grad():
        for i, images in enumerate(tqdm(dataloader, desc=f"Evaluating {model_type}")):
            images = images.to(device)
            
            # Compress and reconstruct
            result = compress_and_reconstruct(model, model_type, images, device)
            reconstructed = result['reconstructed']
            
            # Calculate metrics
            original_np = images.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()
            
            psnr = calculate_psnr(original_np, recon_np)
            ssim = calculate_ssim(original_np, recon_np)
            lpips_value = calculate_lpips(images, reconstructed, device=device)
            
            # Store metrics
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_value)
            bpp_values.append(result['bpp'])
            compression_ratio_values.append(result['compression_ratio'])
            
            # Prepare images for FID calculation
            prepare_fid_calculation(
                original_np, recon_np,
                temp_real_dir, temp_fake_dir
            )
    
    # Calculate mean metrics
    metrics = {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'lpips': np.mean(lpips_values),
        'bpp': np.mean(bpp_values),
        'compression_ratio': np.mean(compression_ratio_values)
    }
    
    # Calculate FID
    metrics['fid'] = calculate_fid(temp_real_dir, temp_fake_dir, device=device)
    
    # Clean up temporary directories
    shutil.rmtree(temp_real_dir, ignore_errors=True)
    shutil.rmtree(temp_fake_dir, ignore_errors=True)
    
    return metrics


def compare_models(models_dict, dataloader, output_dir, device='cuda'):
    """
    Compare different VAE-GAN models on a dataset
    
    Args:
        models_dict: Dictionary of {model_name: (model, model_type)} pairs
        dataloader: DataLoader for the evaluation dataset
        output_dir: Directory to save comparison results
        device: Device to run the models on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reconstructions'), exist_ok=True)
    
    # Evaluate each model
    results = {}
    sample_images = []
    sample_reconstructions = {}
    
    # Get a batch of images for visual comparison
    for images in dataloader:
        sample_images = images[:4].to(device)  # Take only first 4 images
        break
    
    # Evaluate each model and collect reconstructions
    for model_name, (model, model_type) in models_dict.items():
        print(f"Evaluating {model_name}...")
        
        # Evaluate on full dataset
        metrics = evaluate_model_metrics(model, model_type, dataloader, device)
        results[model_name] = metrics
        
        # Get reconstructions for sample images
        result = compress_and_reconstruct(model, model_type, sample_images, device)
        sample_reconstructions[model_name] = result['reconstructed']
        
        # Print metrics
        print(f"  PSNR:  {metrics['psnr']:.2f} dB")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"  FID:   {metrics['fid']:.2f}")
        print(f"  BPP:   {metrics['bpp']:.4f}")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
    
    # Save individual reconstructions
    for i in range(len(sample_images)):
        # Create a figure with original and all reconstructions
        fig, axes = plt.subplots(1, len(models_dict) + 1, figsize=(4 * (len(models_dict) + 1), 4))
        
        # Plot original image
        original = sample_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(np.clip(original, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Plot reconstructions from each model
        for j, (model_name, recon) in enumerate(sample_reconstructions.items()):
            recon_i = recon[i].cpu().permute(1, 2, 0).numpy()
            axes[j+1].imshow(np.clip(recon_i, 0, 1))
            axes[j+1].set_title(f'{model_name}\nPSNR: {results[model_name]["psnr"]:.1f}dB')
            axes[j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstructions', f'comparison_image_{i}.png'), dpi=300)
        plt.close(fig)
    
    # Create comprehensive comparison visualization
    create_comparison_visualizations(results, output_dir)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create summary file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("VAE-GAN Models Comparison Summary\n")
        f.write("================================\n\n")
        
        f.write("Model Comparison:\n")
        for model_name, metrics in results.items():
            f.write(f"- {model_name}:\n")
            f.write(f"  PSNR:  {metrics['psnr']:.2f} dB\n")
            f.write(f"  SSIM:  {metrics['ssim']:.4f}\n")
            f.write(f"  LPIPS: {metrics['lpips']:.4f}\n")
            f.write(f"  FID:   {metrics['fid']:.2f}\n")
            f.write(f"  BPP:   {metrics['bpp']:.4f}\n")
            f.write(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x\n\n")
        
        # Determine best model for each metric
        best_psnr = max(results.items(), key=lambda x: x[1]['psnr'])
        best_ssim = max(results.items(), key=lambda x: x[1]['ssim'])
        best_lpips = min(results.items(), key=lambda x: x[1]['lpips'])  # Lower is better for LPIPS
        best_fid = min(results.items(), key=lambda x: x[1]['fid'])  # Lower is better for FID
        best_bpp = min(results.items(), key=lambda x: x[1]['bpp'])  # Lower is better for BPP
        best_ratio = max(results.items(), key=lambda x: x[1]['compression_ratio'])
        
        f.write("Best Models by Metric:\n")
        f.write(f"- Best PSNR:  {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)\n")
        f.write(f"- Best SSIM:  {best_ssim[0]} ({best_ssim[1]['ssim']:.4f})\n")
        f.write(f"- Best LPIPS: {best_lpips[0]} ({best_lpips[1]['lpips']:.4f})\n")
        f.write(f"- Best FID:   {best_fid[0]} ({best_fid[1]['fid']:.2f})\n")
        f.write(f"- Best BPP:   {best_bpp[0]} ({best_bpp[1]['bpp']:.4f})\n")
        f.write(f"- Best Compression Ratio: {best_ratio[0]} ({best_ratio[1]['compression_ratio']:.2f}x)\n")
    
    return results


def create_comparison_visualizations(results, output_dir):
    """Create visualizations comparing model performance"""
    
    # Create bar plots for each metric
    metrics = ['psnr', 'ssim', 'lpips', 'fid', 'bpp', 'compression_ratio']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS (lower is better)', 
              'FID (lower is better)', 'BPP (lower is better)', 'Compression Ratio']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        
        # Get values for each model
        model_names = list(results.keys())
        values = [results[model][metric] for model in model_names]
        
        # Create bar colors: green for better, red for worse
        # For metrics where lower is better, invert the coloring
        if metric in ['lpips', 'fid', 'bpp']:
            # Normalize values for color mapping (0 = best/green, 1 = worst/red)
            min_val = min(values)
            max_val = max(values)
            normalized = [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in values]
        else:
            # For metrics where higher is better
            min_val = min(values)
            max_val = max(values)
            normalized = [1 - (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in values]
        
        # Create colormap from red to green
        colors = [(1, normalized[i], normalized[i]) for i in range(len(normalized))]
        
        # Create bars
        bars = plt.bar(model_names, values, color=colors)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            value = values[i]
            if metric in ['psnr', 'compression_ratio']:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{value:.2f}", 
                        ha='center', va='bottom')
            elif metric in ['ssim', 'lpips', 'bpp']:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{value:.4f}", 
                        ha='center', va='bottom')
            else:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{value:.1f}", 
                        ha='center', va='bottom')
        
        # Set labels and title
        plt.ylabel(title)
        plt.title(f"Comparison of {title} across VAE-GAN Models")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=300)
        plt.close()
    
    # Create rate-distortion plot (BPP vs PSNR)
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.scatter(metrics['bpp'], metrics['psnr'], label=model_name, s=100)
    
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rate_distortion_comparison.png'), dpi=300)
    plt.close()
    
    # Create PSNR-LPIPS plot
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        plt.scatter(metrics['psnr'], metrics['lpips'], label=model_name, s=100)
    
    plt.xlabel('PSNR (dB)')
    plt.ylabel('LPIPS (lower is better)')
    plt.title('Quality vs Perceptual Similarity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'psnr_lpips_comparison.png'), dpi=300)
    plt.close()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Compare VAE-GAN architectures for image compression')
    parser.add_argument('--beta_model_path', type=str, required=True,
                        help='Path to trained β-VAE-GAN model')
    parser.add_argument('--vq_model_path', type=str, required=True,
                        help='Path to trained VQ-VAE-GAN model')
    parser.add_argument('--hierarchical_model_path', type=str, required=True,
                        help='Path to trained Hierarchical VAE-GAN model')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load models
    beta_model, beta_config = load_model('beta', args.beta_model_path, device)
    vq_model, vq_config = load_model('vq', args.vq_model_path, device)
    hierarchical_model, hierarchical_config = load_model('hierarchical', args.hierarchical_model_path, device)
    
    # Update configs with data directory
    beta_config.data_dir = args.data_dir
    vq_config.data_dir = args.data_dir
    hierarchical_config.data_dir = args.data_dir
    
    # Get evaluation dataset (Kodak)
    eval_loader = get_kodak_dataset(beta_config)
    
    # Create dictionary of models to compare
    models_dict = {
        'β-VAE-GAN': (beta_model, 'beta'),
        'VQ-VAE-GAN': (vq_model, 'vq'),
        'Hierarchical VAE-GAN': (hierarchical_model, 'hierarchical')
    }
    
    # Compare models
    results = compare_models(models_dict, eval_loader, args.output_dir, device)
    
    print(f"Comparison completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()