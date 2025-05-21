"""
Benchmark script to compare VAE-GAN models against standard image codecs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import cv2
import time
from tqdm import tqdm
import argparse
import json
import sys
from io import BytesIO
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model definitions
from beta_vae_gan.model import BetaVAEGAN
from beta_vae_gan.config import Config as BetaConfig
from vq_vae_gan.model import VQVAEGAN, compress_image
from vq_vae_gan.config import Config as VQConfig
from hierarchical_vae_gan.model import HierarchicalVAEGAN, compress_hierarchical_model
from hierarchical_vae_gan.config import Config as HierarchicalConfig

# Import evaluation metrics
from beta_vae_gan.utils import (
    calculate_psnr, calculate_ssim, calculate_lpips
)


class Benchmarker:
    """Class to benchmark different compression methods"""
    
    def __init__(self, image_dir, output_dir, device='cuda'):
        """
        Initialize benchmarker
        
        Args:
            image_dir: Directory containing test images
            output_dir: Directory to save benchmark results
            device: Device for model inference
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.device = device
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'compressed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reconstructed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Transform for images
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Load images
        self.load_images()
    
    def load_images(self):
        """Load test images"""
        extensions = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        
        self.image_files = sorted(image_files)
        self.images = []
        self.image_tensors = []
        
        print(f"Loading {len(self.image_files)} test images...")
        for img_path in self.image_files:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)
            
            self.images.append(img)
            self.image_tensors.append(tensor)
    
    def load_models(self, beta_model_path, vq_model_path, hierarchical_model_path):
        """
        Load VAE-GAN models from checkpoints
        
        Args:
            beta_model_path: Path to β-VAE-GAN checkpoint
            vq_model_path: Path to VQ-VAE-GAN checkpoint
            hierarchical_model_path: Path to Hierarchical VAE-GAN checkpoint
        """
        self.models = {}
        
        if beta_model_path and os.path.exists(beta_model_path):
            beta_config = BetaConfig()
            beta_model = BetaVAEGAN(beta_config)
            beta_model.load_state_dict(torch.load(beta_model_path, map_location=self.device))
            beta_model = beta_model.to(self.device)
            beta_model.eval()
            self.models['beta_vae_gan'] = beta_model
            print("Loaded β-VAE-GAN model")
        
        if vq_model_path and os.path.exists(vq_model_path):
            vq_config = VQConfig()
            vq_model = VQVAEGAN(vq_config)
            vq_model.load_state_dict(torch.load(vq_model_path, map_location=self.device))
            vq_model = vq_model.to(self.device)
            vq_model.eval()
            self.models['vq_vae_gan'] = vq_model
            print("Loaded VQ-VAE-GAN model")
        
        if hierarchical_model_path and os.path.exists(hierarchical_model_path):
            hier_config = HierarchicalConfig()
            hier_model = HierarchicalVAEGAN(hier_config)
            hier_model.load_state_dict(torch.load(hierarchical_model_path, map_location=self.device))
            hier_model = hier_model.to(self.device)
            hier_model.eval()
            self.models['hierarchical_vae_gan'] = hier_model
            print("Loaded Hierarchical VAE-GAN model")
    
    def benchmark_standard_codecs(self, quality_levels=None):
        """
        Benchmark standard image codecs (JPEG, JPEG2000, WebP, BPG)
        
        Args:
            quality_levels: Dictionary of quality levels for each codec
                           e.g., {'jpeg': [10, 30, 50, 70, 90]}
        """
        if quality_levels is None:
            quality_levels = {
                'jpeg': [10, 30, 50, 70, 90],
                'webp': [10, 30, 50, 70, 90],
                'jpeg2000': [10, 30, 50, 70, 90]
                # BPG requires external tools, not included by default
            }
        
        codecs = list(quality_levels.keys())
        self.codec_results = {codec: {
            'quality_levels': [],
            'bpp': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'encode_time': [],
            'decode_time': []
        } for codec in codecs}
        
        print("Benchmarking standard codecs...")
        
        for i, (img, tensor, img_path) in enumerate(tqdm(zip(self.images, self.image_tensors, self.image_files), total=len(self.images))):
            img_np = np.array(img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Get dimensions for BPP calculation
            h, w, _ = img_np.shape
            num_pixels = h * w
            
            for codec in codecs:
                qualities = quality_levels[codec]
                
                for quality in qualities:
                    # Encode and decode with the codec
                    if codec == 'jpeg':
                        encode_start = time.time()
                        buffer = BytesIO()
                        img.save(buffer, format='JPEG', quality=quality)
                        buffer.seek(0)
                        compressed_size = len(buffer.getvalue())
                        encode_time = time.time() - encode_start
                        
                        # Decode
                        decode_start = time.time()
                        recon_img = Image.open(buffer)
                        decode_time = time.time() - decode_start
                        
                        # Save compressed file for inspection
                        compressed_path = os.path.join(self.output_dir, 'compressed', f"{base_name}_{codec}_q{quality}.jpg")
                        recon_img.save(compressed_path)
                    
                    elif codec == 'webp':
                        encode_start = time.time()
                        buffer = BytesIO()
                        img.save(buffer, format='WEBP', quality=quality)
                        buffer.seek(0)
                        compressed_size = len(buffer.getvalue())
                        encode_time = time.time() - encode_start
                        
                        # Decode
                        decode_start = time.time()
                        recon_img = Image.open(buffer)
                        decode_time = time.time() - decode_start
                        
                        # Save compressed file for inspection
                        compressed_path = os.path.join(self.output_dir, 'compressed', f"{base_name}_{codec}_q{quality}.webp")
                        recon_img.save(compressed_path)
                    
                    elif codec == 'jpeg2000':
                        try:
                            # Using OpenCV for JPEG2000
                            encode_start = time.time()
                            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                            
                            # Map quality to compression ratio (100-quality)/100
                            compression_ratio = (100 - quality) / 100 * 10  # Range from 0.1x to 10x
                            if compression_ratio < 1:
                                compression_ratio = 1
                            
                            compressed_path = os.path.join(self.output_dir, 'compressed', f"{base_name}_{codec}_q{quality}.jp2")
                            cv2.imwrite(compressed_path, cv_img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(compression_ratio * 1000)])
                            
                            compressed_size = os.path.getsize(compressed_path)
                            encode_time = time.time() - encode_start
                            
                            # Decode
                            decode_start = time.time()
                            recon_cv_img = cv2.imread(compressed_path)
                            recon_img = Image.fromarray(cv2.cvtColor(recon_cv_img, cv2.COLOR_BGR2RGB))
                            decode_time = time.time() - decode_start
                        except Exception as e:
                            print(f"Error with JPEG2000: {e}")
                            continue
                    
                    # Calculate metrics
                    recon_np = np.array(recon_img)
                    recon_tensor = self.transform(recon_img).unsqueeze(0).to(self.device)
                    
                    # BPP
                    bpp = (compressed_size * 8) / num_pixels
                    
                    # PSNR and SSIM
                    psnr = calculate_psnr(img_np[np.newaxis, ...], recon_np[np.newaxis, ...])
                    ssim = calculate_ssim(img_np[np.newaxis, ...], recon_np[np.newaxis, ...])
                    
                    # LPIPS (perceptual similarity)
                    lpips_value = calculate_lpips(tensor.to(self.device), recon_tensor)
                    
                    # Save results
                    self.codec_results[codec]['quality_levels'].append(quality)
                    self.codec_results[codec]['bpp'].append(bpp)
                    self.codec_results[codec]['psnr'].append(psnr)
                    self.codec_results[codec]['ssim'].append(ssim)
                    self.codec_results[codec]['lpips'].append(lpips_value)
                    self.codec_results[codec]['encode_time'].append(encode_time)
                    self.codec_results[codec]['decode_time'].append(decode_time)
                    
                    # Save reconstructed image
                    recon_path = os.path.join(
                        self.output_dir, 'reconstructed', 
                        f"{base_name}_{codec}_q{quality}_psnr{psnr:.2f}_bpp{bpp:.4f}.png"
                    )
                    recon_img.save(recon_path)
        
        # Aggregate results
        for codec in codecs:
            self.codec_results[codec]['avg_psnr'] = {}
            self.codec_results[codec]['avg_ssim'] = {}
            self.codec_results[codec]['avg_lpips'] = {}
            self.codec_results[codec]['avg_encode_time'] = {}
            self.codec_results[codec]['avg_decode_time'] = {}
            
            qualities = sorted(set(self.codec_results[codec]['quality_levels']))
            for quality in qualities:
                idx = [i for i, q in enumerate(self.codec_results[codec]['quality_levels']) if q == quality]
                
                avg_bpp = np.mean([self.codec_results[codec]['bpp'][i] for i in idx])
                avg_psnr = np.mean([self.codec_results[codec]['psnr'][i] for i in idx])
                avg_ssim = np.mean([self.codec_results[codec]['ssim'][i] for i in idx])
                avg_lpips = np.mean([self.codec_results[codec]['lpips'][i] for i in idx])
                avg_encode_time = np.mean([self.codec_results[codec]['encode_time'][i] for i in idx])
                avg_decode_time = np.mean([self.codec_results[codec]['decode_time'][i] for i in idx])
                
                self.codec_results[codec]['avg_psnr'][str(quality)] = avg_psnr
                self.codec_results[codec]['avg_ssim'][str(quality)] = avg_ssim
                self.codec_results[codec]['avg_lpips'][str(quality)] = avg_lpips
                self.codec_results[codec]['avg_encode_time'][str(quality)] = avg_encode_time
                self.codec_results[codec]['avg_decode_time'][str(quality)] = avg_decode_time
                
                print(f"{codec} (Quality {quality}): BPP={avg_bpp:.4f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
    
    def benchmark_vae_gan_models(self):
        """Benchmark VAE-GAN models from the loaded checkpoints"""
        if not self.models:
            print("No VAE-GAN models loaded for benchmarking")
            return
        
        print("Benchmarking VAE-GAN models...")
        
        self.model_results = {model_name: {
            'bpp': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'encode_time': [],
            'decode_time': []
        } for model_name in self.models.keys()}
        
        for i, (img, tensor, img_path) in enumerate(tqdm(zip(self.images, self.image_tensors, self.image_files), total=len(self.images))):
            img_np = np.array(img)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            for model_name, model in self.models.items():
                tensor = tensor.to(self.device)
                
                with torch.no_grad():
                    if model_name == 'beta_vae_gan':
                        encode_start = time.time()
                        # Forward pass
                        recon_x, z, mu, logvar = model(tensor)
                        
                        # Calculate BPP and compression ratio
                        num_elements = z.numel()
                        bits_per_element = 32  # assuming float32
                        total_bits = num_elements * bits_per_element
                        total_pixels = tensor.numel() / 3  # divide by 3 channels
                        bpp = total_bits / total_pixels
                        encode_time = time.time() - encode_start
                        
                        # Decode time is negligible for beta-VAE-GAN
                        decode_time = 0.001
                        recon_tensor = recon_x
                    
                    elif model_name == 'vq_vae_gan':
                        encode_start = time.time()
                        # Compress image
                        compressed, metadata = compress_image(model, tensor, self.device)
                        encode_time = time.time() - encode_start
                        
                        # Decompress image
                        decode_start = time.time()
                        recon_tensor = model.decompress_image(compressed, metadata, self.device)
                        decode_time = time.time() - decode_start
                        
                        # Get BPP
                        bpp = metadata['bpp']
                    
                    elif model_name == 'hierarchical_vae_gan':
                        # For hierarchical model, use default bit allocation [8, 6, 4]
                        encode_start = time.time()
                        metrics, recon_tensor, _ = compress_hierarchical_model(
                            model, tensor, bit_allocation=[8, 6, 4], device=self.device
                        )
                        encode_time = time.time() - encode_start
                        
                        # Decode time is included in compression function
                        decode_time = 0.001
                        
                        # Get BPP
                        bpp = metrics['bpp']
                
                # Convert to numpy for metric calculation
                recon_np = recon_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
                recon_np = np.clip(recon_np, 0, 1)
                recon_np = (recon_np * 255).astype(np.uint8)
                
                # Calculate metrics
                psnr = calculate_psnr(img_np[np.newaxis, ...], recon_np[np.newaxis, ...])
                ssim = calculate_ssim(img_np[np.newaxis, ...], recon_np[np.newaxis, ...])
                lpips_value = calculate_lpips(tensor, recon_tensor.to(self.device))
                
                # Save results
                self.model_results[model_name]['bpp'].append(bpp)
                self.model_results[model_name]['psnr'].append(psnr)
                self.model_results[model_name]['ssim'].append(ssim)
                self.model_results[model_name]['lpips'].append(lpips_value)
                self.model_results[model_name]['encode_time'].append(encode_time)
                self.model_results[model_name]['decode_time'].append(decode_time)
                
                # Save reconstructed image
                recon_pil = Image.fromarray(recon_np)
                recon_path = os.path.join(
                    self.output_dir, 'reconstructed', 
                    f"{base_name}_{model_name}_psnr{psnr:.2f}_bpp{bpp:.4f}.png"
                )
                recon_pil.save(recon_path)
        
        # Calculate averages
        for model_name in self.models.keys():
            avg_bpp = np.mean(self.model_results[model_name]['bpp'])
            avg_psnr = np.mean(self.model_results[model_name]['psnr'])
            avg_ssim = np.mean(self.model_results[model_name]['ssim'])
            avg_lpips = np.mean(self.model_results[model_name]['lpips'])
            avg_encode_time = np.mean(self.model_results[model_name]['encode_time'])
            avg_decode_time = np.mean(self.model_results[model_name]['decode_time'])
            
            self.model_results[model_name]['avg_bpp'] = avg_bpp
            self.model_results[model_name]['avg_psnr'] = avg_psnr
            self.model_results[model_name]['avg_ssim'] = avg_ssim
            self.model_results[model_name]['avg_lpips'] = avg_lpips
            self.model_results[model_name]['avg_encode_time'] = avg_encode_time
            self.model_results[model_name]['avg_decode_time'] = avg_decode_time
            
            print(f"{model_name}: BPP={avg_bpp:.4f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
    
    def generate_plots(self):
        """Generate comparison plots"""
        # Only run if we have both codec and model results
        if not hasattr(self, 'codec_results') or not hasattr(self, 'model_results'):
            print("Need both codec and model benchmarks to generate plots")
            return
        
        # Prepare data for rate-distortion plots
        plot_data = {}
        
        # Add codec data
        for codec, results in self.codec_results.items():
            # Group by quality level
            quality_levels = sorted(set(results['quality_levels']))
            bpp_by_quality = {}
            psnr_by_quality = {}
            ssim_by_quality = {}
            lpips_by_quality = {}
            
            for quality in quality_levels:
                idx = [i for i, q in enumerate(results['quality_levels']) if q == quality]
                bpp_by_quality[quality] = np.mean([results['bpp'][i] for i in idx])
                psnr_by_quality[quality] = np.mean([results['psnr'][i] for i in idx])
                ssim_by_quality[quality] = np.mean([results['ssim'][i] for i in idx])
                lpips_by_quality[quality] = np.mean([results['lpips'][i] for i in idx])
            
            # Sort by BPP for plotting
            sorted_qualities = sorted(quality_levels, key=lambda q: bpp_by_quality[q])
            plot_data[codec] = {
                'bpp': [bpp_by_quality[q] for q in sorted_qualities],
                'psnr': [psnr_by_quality[q] for q in sorted_qualities],
                'ssim': [ssim_by_quality[q] for q in sorted_qualities],
                'lpips': [lpips_by_quality[q] for q in sorted_qualities],
            }
        
        # Add model data
        for model_name, results in self.model_results.items():
            plot_data[model_name] = {
                'bpp': [results['avg_bpp']],
                'psnr': [results['avg_psnr']],
                'ssim': [results['avg_ssim']],
                'lpips': [results['avg_lpips']],
            }
        
        # Create rate-distortion plot (BPP vs PSNR)
        plt.figure(figsize=(12, 8))
        
        # Plot codec curves
        for codec, data in plot_data.items():
            if codec in self.codec_results:
                plt.plot(data['bpp'], data['psnr'], 'o-', label=codec.upper())
        
        # Plot model points (with different markers)
        markers = ['*', 's', 'D']
        for i, model_name in enumerate(self.model_results.keys()):
            data = plot_data[model_name]
            plt.plot(data['bpp'], data['psnr'], markers[i % len(markers)], 
                     markersize=12, label=model_name)
        
        plt.xlabel('Bits per Pixel (BPP)')
        plt.ylabel('PSNR (dB)')
        plt.title('Rate-Distortion Comparison')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'plots', 'rate_distortion.png'), dpi=300)
        plt.close()
        
        # Create BPP-SSIM plot
        plt.figure(figsize=(12, 8))
        
        # Plot codec curves
        for codec, data in plot_data.items():
            if codec in self.codec_results:
                plt.plot(data['bpp'], data['ssim'], 'o-', label=codec.upper())
        
        # Plot model points
        for i, model_name in enumerate(self.model_results.keys()):
            data = plot_data[model_name]
            plt.plot(data['bpp'], data['ssim'], markers[i % len(markers)], 
                     markersize=12, label=model_name)
        
        plt.xlabel('Bits per Pixel (BPP)')
        plt.ylabel('SSIM')
        plt.title('BPP vs SSIM Comparison')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'plots', 'bpp_ssim.png'), dpi=300)
        plt.close()
        
        # Create BPP-LPIPS plot
        plt.figure(figsize=(12, 8))
        
        # Plot codec curves
        for codec, data in plot_data.items():
            if codec in self.codec_results:
                plt.plot(data['bpp'], data['lpips'], 'o-', label=codec.upper())
        
        # Plot model points
        for i, model_name in enumerate(self.model_results.keys()):
            data = plot_data[model_name]
            plt.plot(data['bpp'], data['lpips'], markers[i % len(markers)], 
                     markersize=12, label=model_name)
        
        plt.xlabel('Bits per Pixel (BPP)')
        plt.ylabel('LPIPS (lower is better)')
        plt.title('BPP vs Perceptual Similarity (LPIPS)')
        plt.grid(True)
        plt.legend()
        
        # Invert y-axis since lower LPIPS is better
        plt.gca().invert_yaxis()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'plots', 'bpp_lpips.png'), dpi=300)
        plt.close()
        
        # Create encoding/decoding time comparison
        plt.figure(figsize=(14, 8))
        
        # Calculate average encoding and decoding times
        encode_times = {}
        decode_times = {}
        
        for codec, results in self.codec_results.items():
            encode_times[codec] = np.mean(results['encode_time'])
            decode_times[codec] = np.mean(results['decode_time'])
        
        for model_name, results in self.model_results.items():
            encode_times[model_name] = results['avg_encode_time']
            decode_times[model_name] = results['avg_decode_time']
        
        # Sort by encoding time
        methods = sorted(encode_times.keys(), key=lambda x: encode_times[x])
        
        # Prepare data for bar chart
        x = np.arange(len(methods))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, [encode_times[m] for m in methods], width, label='Encoding Time')
        plt.bar(x + width/2, [decode_times[m] for m in methods], width, label='Decoding Time')
        
        # Add labels and legend
        plt.xlabel('Compression Method')
        plt.ylabel('Average Time (seconds)')
        plt.title('Encoding and Decoding Time Comparison')
        plt.xticks(x, [m.upper() if m in self.codec_results else m for m in methods])
        plt.yscale('log')  # Log scale for better visualization
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'plots', 'time_comparison.png'), dpi=300)
        plt.close()
        
        print(f"Plots saved to {os.path.join(self.output_dir, 'plots')}")
    
    def save_results(self):
        """Save benchmark results to JSON file"""
        results = {
            'codecs': self.codec_results if hasattr(self, 'codec_results') else {},
            'models': self.model_results if hasattr(self, 'model_results') else {}
        }
        
        results_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Benchmark results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark image compression methods')
    parser.add_argument('--image_dir', type=str, required=True, 
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Directory to save benchmark results')
    parser.add_argument('--beta_model_path', type=str, default=None,
                       help='Path to β-VAE-GAN checkpoint')
    parser.add_argument('--vq_model_path', type=str, default=None,
                       help='Path to VQ-VAE-GAN checkpoint')
    parser.add_argument('--hierarchical_model_path', type=str, default=None,
                       help='Path to Hierarchical VAE-GAN checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for model inference (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create benchmarker
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    benchmarker = Benchmarker(args.image_dir, args.output_dir, device)
    
    # Load models
    benchmarker.load_models(
        args.beta_model_path,
        args.vq_model_path,
        args.hierarchical_model_path
    )
    
    # Benchmark standard codecs
    benchmarker.benchmark_standard_codecs()
    
    # Benchmark VAE-GAN models
    benchmarker.benchmark_vae_gan_models()
    
    # Generate plots
    benchmarker.generate_plots()
    
    # Save results
    benchmarker.save_results()
    
    print("Benchmarking completed!")


if __name__ == "__main__":
    main()