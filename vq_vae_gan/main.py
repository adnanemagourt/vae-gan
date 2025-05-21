"""
Main training script for VQ-VAE-GAN model
"""

import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import argparse
import tempfile
import shutil

from config import Config
from model import VQVAEGAN, compress_image
from data import get_dataloader, get_kodak_dataset, set_seed
from utils import (
    calculate_bpp, calculate_psnr, calculate_ssim, calculate_lpips,
    save_reconstructed_images, prepare_fid_calculation, calculate_fid,
    MetricLogger
)


def train(model, train_loader, val_loader, config, logger):
    """Train the VQ-VAE-GAN model"""
    
    # Set device
    device = torch.device(config.device)
    model = model.to(device)
    
    # Setup optimizers
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=config.lr / 2,  # Discriminator often converges faster, so use lower LR
        betas=(config.beta1, config.beta2)
    )
    
    # Setup schedulers
    scheduler_G = StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    # Create temporary directories for FID calculation
    temp_real_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_real')
    temp_fake_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_fake')
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_fake_dir, exist_ok=True)
    
    # Start training
    best_psnr = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = {
            'total': [], 'recon': [], 'vq': [], 'adv': [], 'disc': []
        }
        epoch_bpp = []
        
        # Training loop
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")
        for i, images in pbar:
            batch_size = images.size(0)
            images = images.to(device)
            
            # ----------------------
            # Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()
            
            # Get reconstructions
            recon_images, _, _, vq_loss, _ = model(images)
            
            # Get discriminator outputs
            disc_real = model.discriminator(images)
            disc_fake = model.discriminator(recon_images.detach())  # Detach to avoid backpropagation to generator
            
            # Calculate discriminator loss
            d_loss_dict = model.loss_function(
                images, recon_images, vq_loss, disc_real, disc_fake, train_generator=False
            )
            
            # Backpropagate discriminator loss
            d_loss_dict['total'].backward()
            optimizer_D.step()
            
            # ----------------------
            # Train Generator (Encoder + Decoder)
            # ----------------------
            optimizer_G.zero_grad()
            
            # Get reconstructions (again for fresh computation graph)
            recon_images, _, _, vq_loss, indices = model(images)
            
            # Get new discriminator outputs for reconstructed images
            disc_fake = model.discriminator(recon_images)
            
            # Calculate generator loss
            g_loss_dict = model.loss_function(
                images, recon_images, vq_loss, disc_real, disc_fake, train_generator=True
            )
            
            # Backpropagate generator loss
            g_loss_dict['total'].backward()
            optimizer_G.step()
            
            # Calculate approximate bits per pixel
            # First, compress the first image in the batch to get actual BPP
            if i % 50 == 0 and batch_size > 0:
                with torch.no_grad():
                    compressed, metadata = compress_image(model, images[:1], device)
                    bpp = metadata['bpp']
            else:
                # For other iterations, use an approximation
                embedding_dim = config.embedding_dim
                image_size = config.image_size
                codebook_size = config.num_embeddings
                
                # Theoretical BPP calculation
                bits_per_index = np.log2(codebook_size)  # bits needed to encode an index
                indices_area = indices.size(1) * indices.size(2)  # number of indices per image
                pixels_area = image_size * image_size  # number of pixels per image
                
                # Approximate BPP
                bpp = (indices_area * bits_per_index) / (pixels_area * 3)  # 3 for RGB channels
            
            # Update metrics
            for key in epoch_losses.keys():
                if key in g_loss_dict:
                    epoch_losses[key].append(g_loss_dict[key])
                elif key in d_loss_dict:
                    epoch_losses[key].append(d_loss_dict[key])
            
            epoch_bpp.append(bpp)
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{g_loss_dict['total'].item():.4f}",
                'd_loss': f"{d_loss_dict['total'].item():.4f}",
                'bpp': f"{bpp:.4f}"
            })
            
            # Log metrics
            if i % config.log_interval == 0:
                logger.log_train_metrics(
                    epoch, i, g_loss_dict, bpp, images=images, reconstructed=recon_images
                )
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Calculate and log epoch average metrics
        avg_metrics = {key: np.mean(vals) for key, vals in epoch_losses.items()}
        avg_bpp = np.mean(epoch_bpp)
        
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"G Loss: {avg_metrics['total']:.4f}, "
              f"D Loss: {avg_metrics['disc']:.4f}, "
              f"BPP: {avg_bpp:.4f}")
        
        # Evaluate on validation set
        if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
            psnr, ssim, lpips_value, fid, eval_bpp, reconstructed_images, original_images = evaluate(
                model, val_loader, config, temp_real_dir, temp_fake_dir
            )
            
            print(f"Validation - "
                  f"PSNR: {psnr:.2f}, "
                  f"SSIM: {ssim:.4f}, "
                  f"LPIPS: {lpips_value:.4f}, "
                  f"FID: {fid:.2f}, "
                  f"BPP: {eval_bpp:.4f}")
            
            # Log evaluation metrics
            logger.log_eval_metrics(
                epoch, psnr, ssim, lpips_value, fid, eval_bpp,
                images=original_images, reconstructed=reconstructed_images
            )
            
            # Save best model
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(logger.log_dir, 'checkpoints', 'best_model.pth'))
            
            # Save latest model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(logger.log_dir, 'checkpoints', 'latest.pth'))
    
    # Save final metrics
    logger.save_metrics()
    
    # Clean up temporary directories
    shutil.rmtree(temp_real_dir, ignore_errors=True)
    shutil.rmtree(temp_fake_dir, ignore_errors=True)
    
    return model


def evaluate(model, dataloader, config, temp_real_dir=None, temp_fake_dir=None):
    """Evaluate the model on a dataset"""
    
    device = torch.device(config.device)
    model.eval()
    
    # Initialize metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    bpp_values = []
    
    # If temp directories for FID calculation are not provided, create them
    if temp_real_dir is None or temp_fake_dir is None:
        temp_real_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_real_eval')
        temp_fake_dir = os.path.join(tempfile.gettempdir(), 'vq_vae_gan_fake_eval')
        os.makedirs(temp_real_dir, exist_ok=True)
        os.makedirs(temp_fake_dir, exist_ok=True)
    
    # Clear temp directories
    for d in [temp_real_dir, temp_fake_dir]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    
    # Collect reconstructed images and metrics
    all_original = []
    all_reconstructed = []
    
    with torch.no_grad():
        for i, images in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            
            # Get reconstructions
            recon_images, _, _, _, indices = model(images, training=False)
            
            # Calculate actual BPP using compression
            compressed, metadata = compress_image(model, images, device)
            bpp = metadata['bpp']
            
            # Convert to CPU numpy arrays for metric calculation
            original_np = images.cpu().numpy()
            recon_np = recon_images.cpu().numpy()
            
            # Calculate metrics
            psnr = calculate_psnr(original_np, recon_np)
            ssim = calculate_ssim(original_np, recon_np)
            lpips_value = calculate_lpips(images, recon_images, device=device)
            
            # Store metrics
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_value)
            bpp_values.append(bpp)
            
            # Keep track of images for FID calculation
            all_original.append(images)
            all_reconstructed.append(recon_images)
            
            # Prepare images for FID calculation
            prepare_fid_calculation(
                original_np, recon_np,
                temp_real_dir, temp_fake_dir
            )
    
    # Calculate mean metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_bpp = np.mean(bpp_values)
    
    # Calculate FID
    fid = calculate_fid(temp_real_dir, temp_fake_dir, device=device)
    
    # Concatenate all images for visualization
    original_images = torch.cat(all_original[:4], dim=0)
    reconstructed_images = torch.cat(all_reconstructed[:4], dim=0)
    
    return avg_psnr, avg_ssim, avg_lpips, fid, avg_bpp, reconstructed_images, original_images


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Train VQ-VAE-GAN model')
    parser.add_argument('--output_dir', type=str, default='./output/vq_vae_gan',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num_embeddings', type=int, default=512,
                        help='Number of embeddings (codebook size)')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost for VQ-VAE')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                        help='Adversarial loss weight')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    config.output_dir = args.output_dir
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.embedding_dim = args.embedding_dim
    config.num_embeddings = args.num_embeddings
    config.commitment_cost = args.commitment_cost
    config.lambda_adv = args.lambda_adv
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Initialize logger
    logger = MetricLogger(config.output_dir, config)
    
    # Get data loaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_kodak_dataset(config)  # Use Kodak dataset for evaluation
    
    # Initialize model
    model = VQVAEGAN(config)
    
    # Resume training if specified
    if args.resume:
        latest_checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'latest.pth')
        if os.path.isfile(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
        else:
            print("No checkpoint found, starting from scratch")
    
    # Train model
    model = train(model, train_loader, val_loader, config, logger)
    
    # Close logger
    logger.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()