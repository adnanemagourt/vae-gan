# VAE-GAN Architectures for Image Compression

This repository provides implementations of three different VAE-GAN (Variational Autoencoder - Generative Adversarial Network) architectures for image compression:

1. **β-VAE-GAN**: A basic VAE-GAN with a continuous latent space, controlled by the β parameter.
2. **VQ-VAE-GAN**: A Vector-Quantized VAE-GAN with a discrete latent space using a learned codebook.
3. **Hierarchical VAE-GAN**: A hierarchical model with multiple levels of latent representations at different scales.

## Features

- Complete PyTorch implementations for all three architectures
- Training scripts with monitoring and checkpointing
- Evaluation metrics: PSNR, SSIM, LPIPS, FID, BPP, Compression Ratio
- Visualization utilities for comparing reconstruction quality
- Support for standard image compression datasets (Kodak, CLIC, DIV2K)
- Rate-distortion performance comparison

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/vae-gan-compression.git
cd vae-gan-compression
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```bash
mkdir -p data
# Download Kodak dataset
python -c "import os; os.makedirs('data/kodak', exist_ok=True)"
python scripts/download_kodak.py
```

## Usage

### Training Models

#### β-VAE-GAN

```bash
cd beta_vae_gan
python main.py --output_dir ./output/beta_vae_gan --data_dir ../data --beta 1.0 --lambda_adv 0.1 --batch_size 16 --epochs 100
```

#### VQ-VAE-GAN

```bash
cd vq_vae_gan
python main.py --output_dir ./output/vq_vae_gan --data_dir ../data --num_embeddings 512 --embedding_dim 64 --commitment_cost 0.25 --lambda_adv 0.1 --batch_size 16 --epochs 100
```

#### Hierarchical VAE-GAN

```bash
cd hierarchical_vae_gan
python main.py --output_dir ./output/hierarchical_vae_gan --data_dir ../data --num_hierarchies 3 --beta 1.0 --lambda_adv 0.1 --batch_size 16 --epochs 100
```

### Evaluating Models

#### β-VAE-GAN

```bash
cd beta_vae_gan
python evaluate.py --model_path ./output/beta_vae_gan/checkpoints/best_model.pth --output_dir ./eval_results/beta_vae_gan --data_dir ../data
```

#### VQ-VAE-GAN

```bash
cd vq_vae_gan
python evaluate.py --model_path ./output/vq_vae_gan/checkpoints/best_model.pth --output_dir ./eval_results/vq_vae_gan --data_dir ../data
```

#### Hierarchical VAE-GAN

```bash
cd hierarchical_vae_gan
python evaluate.py --model_path ./output/hierarchical_vae_gan/checkpoints/best_model.pth --output_dir ./eval_results/hierarchical_vae_gan --data_dir ../data --compare_bits
```

### Comparing Models

```bash
python compare_models.py \
  --beta_model_path ./beta_vae_gan/output/beta_vae_gan/checkpoints/best_model.pth \
  --vq_model_path ./vq_vae_gan/output/vq_vae_gan/checkpoints/best_model.pth \
  --hierarchical_model_path ./hierarchical_vae_gan/output/hierarchical_vae_gan/checkpoints/best_model.pth \
  --output_dir ./comparison_results \
  --data_dir ./data
```

## Architecture Details

### β-VAE-GAN

The β-VAE-GAN combines a variational autoencoder with a GAN discriminator. The architecture consists of:

- **Encoder**: CNN that maps input image to a latent distribution (μ, σ)
- **Reparameterization**: Samples from the latent distribution using the reparameterization trick
- **Decoder**: CNN that reconstructs images from latent vectors
- **Discriminator**: CNN that distinguishes between real and reconstructed images
- **Loss function**: Combines reconstruction loss, KL divergence (weighted by β), and adversarial loss

### VQ-VAE-GAN

The VQ-VAE-GAN uses vector quantization to create a discrete latent space:

- **Encoder**: CNN that maps input image to continuous encodings
- **Vector Quantization**: Maps continuous encodings to nearest vectors in a learnable codebook
- **Decoder**: CNN that reconstructs images from quantized vectors
- **Discriminator**: CNN that distinguishes between real and reconstructed images
- **Loss function**: Combines reconstruction loss, VQ commitment loss, and adversarial loss
- **Entropy Coding**: Additional arithmetic coding for efficient representation of discrete codes

### Hierarchical VAE-GAN

The Hierarchical VAE-GAN uses multiple levels of latent representations:

- **Hierarchical Encoder**: Produces latent representations at different scales/resolutions
- **Hierarchical Decoder**: Reconstructs images using latent codes from all levels
- **Self-Attention**: Incorporates spatial attention at multiple levels
- **Discriminator**: CNN with matching hierarchical structure
- **Loss function**: Combines reconstruction loss, weighted KL divergence for each level, and adversarial loss
- **Bit Allocation**: Different bit allocations can be used for different hierarchy levels

## Performance Metrics

The models are evaluated using the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction fidelity
- **SSIM (Structural Similarity Index)**: Measures structural similarity between original and reconstructed images
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity
- **FID (Fréchet Inception Distance)**: Measures the realism/quality of reconstructed images
- **BPP (Bits Per Pixel)**: Measures compression efficiency
- **Compression Ratio**: Ratio of original file size to compressed file size

## References

- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. International Conference on Learning Representations (ICLR).
- Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems.
- Ballé, J., et al. (2018). Variational image compression with a scale hyperprior. International Conference on Learning Representations (ICLR).
- van den Oord, A., et al. (2017). Neural discrete representation learning. Advances in Neural Information Processing Systems.

## License

This project is licensed under the MIT License - see the LICENSE file for details.