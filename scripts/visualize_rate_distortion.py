"""
Script to visualize rate-distortion curves across training iterations for VAE-GAN models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation
import glob


def load_metrics(metrics_file):
    """
    Load metrics from numpy file
    
    Args:
        metrics_file: Path to metrics file (.npy)
        
    Returns:
        Dictionary containing all metrics
    """
    try:
        metrics = np.load(metrics_file, allow_pickle=True).item()
        return metrics
    except Exception as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        return None


def plot_rate_distortion_curve(train_metrics, eval_metrics, output_path, model_name, metrics_type='psnr'):
    """
    Plot rate-distortion curve (BPP vs. quality metric)
    
    Args:
        train_metrics: Training metrics
        eval_metrics: Evaluation metrics
        output_path: Path to save the plot
        model_name: Name of the model
        metrics_type: Type of quality metric ('psnr', 'ssim', or 'lpips')
    """
    plt.figure(figsize=(10, 6))
    
    # Get BPP values
    train_bpp = train_metrics['bpp']
    
    # Get quality metric values
    if metrics_type == 'psnr':
        # Need to calculate PSNR from MSE (reconstruction loss)
        train_recon_loss = train_metrics['recon']
        train_quality = [20 * np.log10(1.0 / np.sqrt(max(1e-10, loss))) for loss in train_recon_loss]
        y_label = 'PSNR (dB)'
    elif metrics_type == 'ssim':
        # We don't have SSIM for training, so we'll create a proxy
        train_recon_loss = train_metrics['recon']
        # Rough approximation of SSIM from MSE
        train_quality = [1.0 - np.sqrt(min(0.999, loss)) for loss in train_recon_loss]
        y_label = 'SSIM'
    elif metrics_type == 'lpips':
        # We don't have LPIPS for training, so we'll create a proxy
        train_recon_loss = train_metrics['recon']
        # Rough approximation of LPIPS from MSE
        train_quality = [min(0.5, loss*5) for loss in train_recon_loss]
        y_label = 'LPIPS (lower is better)'
    else:
        raise ValueError(f"Unknown metrics type: {metrics_type}")
    
    # Filter out any NaN or infinity values
    valid_indices = [i for i in range(len(train_bpp)) if np.isfinite(train_bpp[i]) and np.isfinite(train_quality[i])]
    train_bpp = [train_bpp[i] for i in valid_indices]
    train_quality = [train_quality[i] for i in valid_indices]
    
    # Plot scatter points of training metrics
    if metrics_type != 'lpips':
        plt.scatter(train_bpp, train_quality, alpha=0.1, c='blue', label='Training iterations')
    else:
        plt.scatter(train_bpp, train_quality, alpha=0.1, c='blue', label='Training iterations')
        # For LPIPS, better is lower, so invert y-axis
        plt.gca().invert_yaxis()
    
    # If eval metrics are available, plot them
    if eval_metrics is not None and 'bpp' in eval_metrics and metrics_type in eval_metrics:
        eval_bpp = eval_metrics['bpp']
        eval_quality = eval_metrics[metrics_type]
        
        # Check if they are lists
        if not isinstance(eval_bpp, list):
            eval_bpp = [eval_bpp]
            eval_quality = [eval_quality]
        
        # Filter out any NaN or infinity values
        valid_indices = [i for i in range(len(eval_bpp)) if np.isfinite(eval_bpp[i]) and np.isfinite(eval_quality[i])]
        eval_bpp = [eval_bpp[i] for i in valid_indices]
        eval_quality = [eval_quality[i] for i in valid_indices]
        
        plt.scatter(eval_bpp, eval_quality, alpha=0.8, c='red', s=100, marker='*', label='Validation')
    
    # Add labels and title
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel(y_label)
    plt.title(f'Rate-Distortion Curve for {model_name}')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_rate_distortion_animation(train_metrics, eval_metrics, output_path, model_name, metrics_type='psnr', frames=100):
    """
    Create an animation of rate-distortion curve evolution during training
    
    Args:
        train_metrics: Training metrics
        eval_metrics: Evaluation metrics
        output_path: Path to save the animation
        model_name: Name of the model
        metrics_type: Type of quality metric ('psnr', 'ssim', or 'lpips')
        frames: Number of frames in the animation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get BPP values
    train_bpp = train_metrics['bpp']
    
    # Get quality metric values
    if metrics_type == 'psnr':
        # Need to calculate PSNR from MSE (reconstruction loss)
        train_recon_loss = train_metrics['recon']
        train_quality = [20 * np.log10(1.0 / np.sqrt(max(1e-10, loss))) for loss in train_recon_loss]
        y_label = 'PSNR (dB)'
    elif metrics_type == 'ssim':
        # We don't have SSIM for training, so we'll create a proxy
        train_recon_loss = train_metrics['recon']
        # Rough approximation of SSIM from MSE
        train_quality = [1.0 - np.sqrt(min(0.999, loss)) for loss in train_recon_loss]
        y_label = 'SSIM'
    elif metrics_type == 'lpips':
        # We don't have LPIPS for training, so we'll create a proxy
        train_recon_loss = train_metrics['recon']
        # Rough approximation of LPIPS from MSE
        train_quality = [min(0.5, loss*5) for loss in train_recon_loss]
        y_label = 'LPIPS (lower is better)'
        # For LPIPS, better is lower, so invert y-axis
        ax.invert_yaxis()
    else:
        raise ValueError(f"Unknown metrics type: {metrics_type}")
    
    # Filter out any NaN or infinity values
    valid_indices = [i for i in range(len(train_bpp)) if np.isfinite(train_bpp[i]) and np.isfinite(train_quality[i])]
    train_bpp = [train_bpp[i] for i in valid_indices]
    train_quality = [train_quality[i] for i in valid_indices]
    
    # Set axis labels and title
    ax.set_xlabel('Bits per Pixel (BPP)')
    ax.set_ylabel(y_label)
    ax.set_title(f'Rate-Distortion Evolution for {model_name}')
    ax.grid(True)
    
    # Set axis limits
    ax.set_xlim(min(train_bpp) * 0.9, max(train_bpp) * 1.1)
    y_min = min(train_quality) * 0.9 if metrics_type != 'lpips' else min(train_quality) * 0.5
    y_max = max(train_quality) * 1.1 if metrics_type != 'lpips' else max(train_quality) * 2.0
    ax.set_ylim(y_min, y_max)
    
    # Create scatter plot
    sc = ax.scatter([], [], alpha=0.6, c='blue')
    
    # Add iteration text
    iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    # If eval metrics are available, plot them as static points
    if eval_metrics is not None and 'bpp' in eval_metrics and metrics_type in eval_metrics:
        eval_bpp = eval_metrics['bpp']
        eval_quality = eval_metrics[metrics_type]
        
        # Check if they are lists
        if not isinstance(eval_bpp, list):
            eval_bpp = [eval_bpp]
            eval_quality = [eval_quality]
        
        # Filter out any NaN or infinity values
        valid_indices = [i for i in range(len(eval_bpp)) if np.isfinite(eval_bpp[i]) and np.isfinite(eval_quality[i])]
        eval_bpp = [eval_bpp[i] for i in valid_indices]
        eval_quality = [eval_quality[i] for i in valid_indices]
        
        ax.scatter(eval_bpp, eval_quality, alpha=0.8, c='red', s=100, marker='*', label='Validation')
        ax.legend()
    
    # Calculate frames to show
    num_points = len(train_bpp)
    frame_indices = np.linspace(0, num_points-1, frames, dtype=int)
    
    # Animation update function
    def update(frame):
        idx = frame_indices[frame]
        if idx == 0:
            sc.set_offsets(np.array([]))
        else:
            sc.set_offsets(np.column_stack((train_bpp[:idx], train_quality[:idx])))
        iter_text.set_text(f'Iteration: {idx}/{num_points}')
        return sc, iter_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    
    # Save animation
    ani.save(output_path, writer='pillow', fps=10)
    plt.close()


def visualize_metrics_directory(metrics_dir, output_dir, model_name):
    """
    Visualize metrics from a directory for a VAE-GAN model
    
    Args:
        metrics_dir: Directory containing metrics files
        output_dir: Directory to save visualizations
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find train and eval metrics files
    train_metrics_file = os.path.join(metrics_dir, 'train_metrics.npy')
    eval_metrics_file = os.path.join(metrics_dir, 'eval_metrics.npy')
    
    # Load metrics
    train_metrics = load_metrics(train_metrics_file) if os.path.exists(train_metrics_file) else None
    eval_metrics = load_metrics(eval_metrics_file) if os.path.exists(eval_metrics_file) else None
    
    if train_metrics is None:
        print(f"No training metrics found in {metrics_dir}")
        return
    
    # Plot rate-distortion curves for different quality metrics
    for metric_type in ['psnr', 'ssim', 'lpips']:
        output_path = os.path.join(output_dir, f'{model_name}_rd_{metric_type}.png')
        plot_rate_distortion_curve(train_metrics, eval_metrics, output_path, model_name, metric_type)
        
        # Create animation
        animation_path = os.path.join(output_dir, f'{model_name}_rd_{metric_type}_animation.gif')
        create_rate_distortion_animation(train_metrics, eval_metrics, animation_path, model_name, metric_type)
    
    # Plot loss curves
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics['total'], label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.grid(True)
    
    # Plot reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(train_metrics['recon'], label='Reconstruction Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.grid(True)
    
    # Plot regularization loss (KL for beta-VAE, VQ for VQ-VAE)
    plt.subplot(2, 2, 3)
    if 'kl' in train_metrics:
        plt.plot(train_metrics['kl'], label='KL Divergence')
        title = 'KL Divergence'
    elif 'vq' in train_metrics:
        plt.plot(train_metrics['vq'], label='VQ Loss')
        title = 'VQ Loss'
    else:
        title = 'Regularization Loss'
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    
    # Plot adversarial loss
    plt.subplot(2, 2, 4)
    plt.plot(train_metrics['adv'], label='Adversarial Loss')
    plt.plot(train_metrics['disc'], label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Adversarial and Discriminator Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss_curves.png'), dpi=300)
    plt.close()
    
    # Plot BPP histogram
    plt.figure(figsize=(10, 6))
    plt.hist(train_metrics['bpp'], bins=50, alpha=0.7)
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('Frequency')
    plt.title(f'BPP Histogram for {model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_bpp_histogram.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations for {model_name} saved in {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize rate-distortion curves for VAE-GAN models')
    parser.add_argument('--metrics_dir', type=str, required=True,
                        help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--model_name', type=str, default='VAE-GAN',
                        help='Name of the model')
    args = parser.parse_args()
    
    visualize_metrics_directory(args.metrics_dir, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()