"""
Script to visualize the latent space of VAE-GAN models
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
import glob
import seaborn as sns
import umap

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model definitions
from beta_vae_gan.model import BetaVAEGAN
from beta_vae_gan.config import Config as BetaConfig
from vq_vae_gan.model import VQVAEGAN
from vq_vae_gan.config import Config as VQConfig
from hierarchical_vae_gan.model import HierarchicalVAEGAN
from hierarchical_vae_gan.config import Config as HierarchicalConfig


class LatentSpaceVisualizer:
    """Class to visualize the latent space of VAE-GAN models"""
    
    def __init__(self, model_type, model_path, output_dir, device='cuda'):
        """
        Initialize visualizer
        
        Args:
            model_type: Type of model ('beta', 'vq', 'hierarchical')
            model_path: Path to model checkpoint
            output_dir: Directory to save visualizations
            device: Device for model inference
        """
        self.model_type = model_type
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Load model
        self.model = self.load_model()
    
    def load_model(self):
        """Load the specified model"""
        if self.model_type == 'beta':
            config = BetaConfig()
            model = BetaVAEGAN(config)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"Loaded β-VAE-GAN model from {self.model_path}")
        
        elif self.model_type == 'vq':
            config = VQConfig()
            model = VQVAEGAN(config)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"Loaded VQ-VAE-GAN model from {self.model_path}")
        
        elif self.model_type == 'hierarchical':
            config = HierarchicalConfig()
            model = HierarchicalVAEGAN(config)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"Loaded Hierarchical VAE-GAN model from {self.model_path}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def encode_images(self, image_dir, max_images=100):
        """
        Encode images to latent space
        
        Args:
            image_dir: Directory containing images
            max_images: Maximum number of images to encode
            
        Returns:
            Dictionary of latent vectors and metadata
        """
        # Find images
        extensions = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        # Limit number of images
        if len(image_files) > max_images:
            print(f"Found {len(image_files)} images, limiting to {max_images}")
            image_files = image_files[:max_images]
        else:
            print(f"Found {len(image_files)} images")
        
        # Dictionary to store results
        results = {
            'latent_vectors': [],
            'image_paths': [],
            'reconstructions': []
        }
        
        # Encode images
        for img_path in tqdm(image_files, desc="Encoding images"):
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.model_type == 'beta':
                    # Get the latent vector from encoder
                    recon_x, z, mu, logvar = self.model(tensor)
                    latent = mu.cpu().numpy()  # Use mean as latent vector
                    reconstruction = recon_x.cpu().squeeze().permute(1, 2, 0).numpy()
                
                elif self.model_type == 'vq':
                    # Get the latent vector (pre-quantization)
                    z_q, z, vq_loss, indices = self.model.encode(tensor)
                    # Use pre-quantization continuous vector
                    latent = z.cpu().numpy()
                    # Forward to get reconstruction
                    recon_x = self.model.decode(z_q)
                    reconstruction = recon_x.cpu().squeeze().permute(1, 2, 0).numpy()
                
                elif self.model_type == 'hierarchical':
                    # Get the latent vectors from all levels
                    z_list, mu_list, logvar_list = self.model.encode(tensor)
                    # Concatenate the means from all levels
                    latent = torch.cat([mu.view(mu.size(0), -1) for mu in mu_list], dim=1).cpu().numpy()
                    # Forward to get reconstruction
                    recon_x = self.model.decode(z_list)
                    reconstruction = recon_x.cpu().squeeze().permute(1, 2, 0).numpy()
            
            results['latent_vectors'].append(latent)
            results['image_paths'].append(img_path)
            results['reconstructions'].append(reconstruction)
        
        # Convert lists to arrays
        results['latent_vectors'] = np.vstack([v.reshape(1, -1) for v in results['latent_vectors']])
        results['reconstructions'] = np.array(results['reconstructions'])
        
        print(f"Encoded {len(results['image_paths'])} images to latent vectors of shape {results['latent_vectors'].shape}")
        return results
    
    def visualize_latent_space(self, latent_vectors, image_paths, reconstructions):
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            latent_vectors: Array of latent vectors
            image_paths: List of image paths
            reconstructions: Array of reconstructed images
        """
        print("Visualizing latent space...")
        
        # Check dimensionality
        n_samples, latent_dim = latent_vectors.shape
        print(f"Latent space dimensionality: {latent_dim}")
        
        # Create output directory for visualizations
        viz_dir = os.path.join(self.output_dir, 'latent_visualization')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. PCA visualization
        print("Performing PCA...")
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.8, s=100)
        plt.colorbar(label='Sample index')
        plt.title(f'PCA projection of {self.model_type} latent space')
        plt.xlabel(f'PC1 (variance: {pca.explained_variance_ratio_[0]:.4f})')
        plt.ylabel(f'PC2 (variance: {pca.explained_variance_ratio_[1]:.4f})')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'pca_projection.png'), dpi=300)
        plt.close()
        
        # 2. t-SNE visualization
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples-1), 
                   learning_rate='auto', init='pca', random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.8, s=100)
        plt.colorbar(label='Sample index')
        plt.title(f't-SNE projection of {self.model_type} latent space')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'tsne_projection.png'), dpi=300)
        plt.close()
        
        # 3. UMAP visualization
        print("Performing UMAP...")
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, n_samples-1), 
                          min_dist=0.1, random_state=42)
        latent_umap = reducer.fit_transform(latent_vectors)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_umap[:, 0], latent_umap[:, 1], alpha=0.8, s=100)
        plt.colorbar(label='Sample index')
        plt.title(f'UMAP projection of {self.model_type} latent space')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'umap_projection.png'), dpi=300)
        plt.close()
        
        # 4. Visualize reconstructions in latent space
        self.visualize_reconstructions_in_latent_space(
            latent_pca, image_paths, reconstructions, 'pca', viz_dir)
        self.visualize_reconstructions_in_latent_space(
            latent_tsne, image_paths, reconstructions, 'tsne', viz_dir)
        self.visualize_reconstructions_in_latent_space(
            latent_umap, image_paths, reconstructions, 'umap', viz_dir)
        
        # 5. Latent space heatmap (correlations)
        if latent_dim <= 100:  # Only for reasonable dimensions
            print("Generating latent space correlation heatmap...")
            plt.figure(figsize=(12, 10))
            corr_matrix = np.corrcoef(latent_vectors, rowvar=False)
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
            plt.title(f'Latent dimension correlations ({self.model_type})')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'latent_correlations.png'), dpi=300)
            plt.close()
        
        # 6. Latent space distribution
        print("Analyzing latent space distribution...")
        plt.figure(figsize=(16, 8))
        
        # If many dimensions, show only subset or aggregate
        if latent_dim <= 20:
            # Show individual distributions for each dimension
            for i in range(min(10, latent_dim)):
                plt.subplot(2, 5, i+1)
                sns.histplot(latent_vectors[:, i], kde=True)
                plt.title(f'Dimension {i+1}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
        else:
            # Show summary statistics
            plt.subplot(1, 2, 1)
            means = np.mean(latent_vectors, axis=0)
            plt.hist(means, bins=20)
            plt.title('Distribution of dimension means')
            plt.xlabel('Mean value')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            stds = np.std(latent_vectors, axis=0)
            plt.hist(stds, bins=20)
            plt.title('Distribution of dimension standard deviations')
            plt.xlabel('Standard deviation')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'latent_distributions.png'), dpi=300)
        plt.close()
        
        # 7. Save projections for interactive visualization
        np.save(os.path.join(viz_dir, 'latent_vectors.npy'), latent_vectors)
        np.save(os.path.join(viz_dir, 'pca_projection.npy'), latent_pca)
        np.save(os.path.join(viz_dir, 'tsne_projection.npy'), latent_tsne)
        np.save(os.path.join(viz_dir, 'umap_projection.npy'), latent_umap)
        
        print(f"Latent space visualization saved to {viz_dir}")
    
    def visualize_reconstructions_in_latent_space(self, projection, image_paths, 
                                                reconstructions, method, output_dir):
        """
        Visualize reconstructed images in the latent space projection
        
        Args:
            projection: 2D projection of latent vectors
            image_paths: List of image paths
            reconstructions: Array of reconstructed images
            method: Name of projection method
            output_dir: Directory to save visualizations
        """
        print(f"Creating {method.upper()} projection with reconstructions...")
        
        # Create figure
        plt.figure(figsize=(20, 20))
        
        # Scatter points without images first
        plt.scatter(projection[:, 0], projection[:, 1], alpha=0.3, s=100, c='gray')
        
        # Calculate grid for positioning images
        x_min, x_max = projection[:, 0].min(), projection[:, 0].max()
        y_min, y_max = projection[:, 1].min(), projection[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Choose subset of images to display (avoid overcrowding)
        n_samples = len(image_paths)
        n_display = min(25, n_samples)
        
        if n_samples > n_display:
            # Use k-means to select representative samples
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_display, random_state=42)
            clusters = kmeans.fit_predict(projection)
            centroids = kmeans.cluster_centers_
            
            # Find closest sample to each centroid
            selected_indices = []
            for i in range(n_display):
                dist = np.sum((projection - centroids[i])**2, axis=1)
                closest_idx = np.argmin(dist)
                selected_indices.append(closest_idx)
        else:
            selected_indices = list(range(n_samples))
        
        # Plot selected reconstructions
        for idx in selected_indices:
            x, y = projection[idx]
            img = reconstructions[idx]
            
            # Create inset for image
            img_size = 0.1  # Size relative to figure
            img_extent = [
                x - img_size * x_range / 2, 
                x + img_size * x_range / 2,
                y - img_size * y_range / 2, 
                y + img_size * y_range / 2
            ]
            
            plt.imshow(np.clip(img, 0, 1), extent=img_extent, zorder=10)
            
            # Annotate with file name
            file_name = os.path.basename(image_paths[idx])
            plt.annotate(file_name[:10], (x, y - img_size * y_range / 1.8), 
                         ha='center', fontsize=8, color='blue')
        
        plt.title(f'{method.upper()} projection with reconstructions')
        plt.xlabel(f'{method.upper()} dimension 1')
        plt.ylabel(f'{method.upper()} dimension 2')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{method}_with_reconstructions.png'), dpi=300)
        plt.close()
    
    def visualize_latent_interpolation(self, image_dir, n_steps=10):
        """
        Visualize interpolation between images in the latent space
        
        Args:
            image_dir: Directory containing images
            n_steps: Number of interpolation steps
        """
        print("Generating latent space interpolations...")
        
        # Create output directory
        interpolation_dir = os.path.join(self.output_dir, 'interpolation')
        os.makedirs(interpolation_dir, exist_ok=True)
        
        # Find images
        extensions = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        # Need at least 2 images
        if len(image_files) < 2:
            print("Need at least 2 images for interpolation")
            return
        
        # Randomly select pairs of images for interpolation
        n_pairs = min(5, len(image_files))
        pairs = []
        
        for _ in range(n_pairs):
            idx1, idx2 = np.random.choice(len(image_files), 2, replace=False)
            pairs.append((image_files[idx1], image_files[idx2]))
        
        # Process each pair
        for i, (img1_path, img2_path) in enumerate(pairs):
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            tensor1 = self.transform(img1).unsqueeze(0).to(self.device)
            tensor2 = self.transform(img2).unsqueeze(0).to(self.device)
            
            # Encode images
            with torch.no_grad():
                if self.model_type == 'beta':
                    # Get the latent vectors
                    _, _, mu1, _ = self.model(tensor1)
                    _, _, mu2, _ = self.model(tensor2)
                    
                    # Interpolate in latent space
                    interpolations = []
                    for alpha in np.linspace(0, 1, n_steps):
                        z = (1 - alpha) * mu1 + alpha * mu2
                        recon = self.model.decode(z)
                        interpolations.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
                
                elif self.model_type == 'vq':
                    # For VQ-VAE, interpolate in the pre-quantization continuous space
                    z_q1, z1, _, _ = self.model.encode(tensor1)
                    z_q2, z2, _, _ = self.model.encode(tensor2)
                    
                    # Interpolate in continuous latent space
                    interpolations = []
                    for alpha in np.linspace(0, 1, n_steps):
                        # Interpolate pre-quantization
                        z_interp = (1 - alpha) * z1 + alpha * z2
                        
                        # Quantize interpolated vector
                        z_q_interp, _, _ = self.model.vector_quantizer(z_interp, training=False)
                        
                        # Decode
                        recon = self.model.decode(z_q_interp)
                        interpolations.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
                
                elif self.model_type == 'hierarchical':
                    # For hierarchical model, interpolate each level separately
                    z_list1, mu_list1, _ = self.model.encode(tensor1)
                    z_list2, mu_list2, _ = self.model.encode(tensor2)
                    
                    # Interpolate in latent space
                    interpolations = []
                    for alpha in np.linspace(0, 1, n_steps):
                        # Interpolate at each level
                        z_interp_list = [(1 - alpha) * z1 + alpha * z2 
                                        for z1, z2 in zip(z_list1, z_list2)]
                        
                        # Decode
                        recon = self.model.decode(z_interp_list)
                        interpolations.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
            
            # Create interpolation grid
            fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 3))
            
            for j, img in enumerate(interpolations):
                axes[j].imshow(np.clip(img, 0, 1))
                axes[j].set_title(f'α={j/(n_steps-1):.2f}')
                axes[j].axis('off')
            
            plt.suptitle(f'Latent Space Interpolation (Pair {i+1})')
            plt.tight_layout()
            plt.savefig(os.path.join(interpolation_dir, f'interpolation_pair{i+1}.png'), dpi=300)
            plt.close()
        
        print(f"Interpolation visualizations saved to {interpolation_dir}")
    
    def visualize_latent_traversal(self, image_path, n_dims=10, n_steps=7, scale=3.0):
        """
        Visualize traversal along latent dimensions
        
        Args:
            image_path: Path to a seed image
            n_dims: Number of dimensions to traverse
            n_steps: Number of steps in each traversal
            scale: Scale of the traversal (in standard deviations)
        """
        print("Generating latent traversal visualizations...")
        
        # Create output directory
        traversal_dir = os.path.join(self.output_dir, 'traversal')
        os.makedirs(traversal_dir, exist_ok=True)
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            if self.model_type == 'beta':
                # Get the latent vector
                _, _, mu, _ = self.model(tensor)
                latent = mu
                
                # Get dimension statistics from model (if available)
                if hasattr(self.model, 'z_mean') and hasattr(self.model, 'z_std'):
                    z_mean = self.model.z_mean
                    z_std = self.model.z_std
                else:
                    # Use standard normal prior
                    z_mean = torch.zeros_like(latent)
                    z_std = torch.ones_like(latent)
                
                # Select dimensions to traverse
                latent_dim = latent.shape[1]
                n_dims = min(n_dims, latent_dim)
                
                # Generate traversals
                traversal_imgs = []
                for dim in range(n_dims):
                    dim_traversal = []
                    for step in np.linspace(-scale, scale, n_steps):
                        # Create modified latent
                        z_mod = latent.clone()
                        z_mod[0, dim] = latent[0, dim] + step * z_std[dim]
                        
                        # Decode
                        recon = self.model.decode(z_mod)
                        dim_traversal.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
                    
                    traversal_imgs.append(dim_traversal)
            
            elif self.model_type == 'vq':
                print("Latent traversal not implemented for VQ-VAE-GAN")
                return
            
            elif self.model_type == 'hierarchical':
                # For hierarchical model, traverse first level dimensions
                z_list, mu_list, _ = self.model.encode(tensor)
                
                # Use the first level for traversal
                latent = z_list[0]
                
                # Select dimensions to traverse
                latent_dim = latent.shape[1]
                n_dims = min(n_dims, latent_dim)
                
                # Generate traversals
                traversal_imgs = []
                for dim in range(n_dims):
                    dim_traversal = []
                    for step in np.linspace(-scale, scale, n_steps):
                        # Create modified latent list
                        z_mod_list = z_list.copy()
                        
                        # Modify the first level
                        z_mod = z_mod_list[0].clone()
                        z_mod[0, dim] = z_mod[0, dim] + step
                        z_mod_list[0] = z_mod
                        
                        # Decode
                        recon = self.model.decode(z_mod_list)
                        dim_traversal.append(recon.cpu().squeeze().permute(1, 2, 0).numpy())
                    
                    traversal_imgs.append(dim_traversal)
        
        # Create visualization grid
        if traversal_imgs:
            fig, axes = plt.subplots(n_dims, n_steps, figsize=(n_steps * 2, n_dims * 2))
            
            for i, dim_traversal in enumerate(traversal_imgs):
                for j, img in enumerate(dim_traversal):
                    if n_dims > 1:
                        ax = axes[i, j]
                    else:
                        ax = axes[j]
                    
                    ax.imshow(np.clip(img, 0, 1))
                    
                    if j == 0:
                        ax.set_ylabel(f'Dim {i}')
                    if i == 0:
                        ax.set_title(f'Step {j+1}')
                    
                    ax.axis('off')
            
            plt.suptitle(f'Latent Space Traversal ({n_dims} dimensions, {n_steps} steps, scale={scale})')
            plt.tight_layout()
            plt.savefig(os.path.join(traversal_dir, 'latent_traversal.png'), dpi=300)
            plt.close()
            
            print(f"Traversal visualization saved to {traversal_dir}")
    
    def run_visualizations(self, image_dir, seed_image=None, max_images=100):
        """
        Run all visualization methods
        
        Args:
            image_dir: Directory containing images
            seed_image: Path to seed image for traversal
            max_images: Maximum number of images to process
        """
        # Encode images to latent space
        results = self.encode_images(image_dir, max_images)
        
        # Visualize latent space
        self.visualize_latent_space(
            results['latent_vectors'],
            results['image_paths'],
            results['reconstructions']
        )
        
        # Visualize interpolation
        self.visualize_latent_interpolation(image_dir)
        
        # Visualize traversal (if seed image provided)
        if seed_image and os.path.exists(seed_image):
            self.visualize_latent_traversal(seed_image)
        elif len(results['image_paths']) > 0:
            # Use first image as seed
            self.visualize_latent_traversal(results['image_paths'][0])


def main():
    parser = argparse.ArgumentParser(description='Visualize latent space of VAE-GAN models')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['beta', 'vq', 'hierarchical'],
                       help='Type of model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--seed_image', type=str, default=None,
                       help='Path to seed image for traversal')
    parser.add_argument('--max_images', type=int, default=100,
                       help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for model inference (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create visualizer
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    visualizer = LatentSpaceVisualizer(
        args.model_type,
        args.model_path,
        args.output_dir,
        device
    )
    
    # Run visualizations
    visualizer.run_visualizations(
        args.image_dir,
        args.seed_image,
        args.max_images
    )
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()