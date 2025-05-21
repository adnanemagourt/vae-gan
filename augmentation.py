"""
Image augmentation utilities for training VAE-GAN models
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import random
import cv2
import os
from tqdm import tqdm


class AdvancedAugmentation:
    """Advanced data augmentation for image compression training"""
    
    def __init__(self,
                 rotation_range=15,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 noise_range=(0.0, 0.05),
                 blur_prob=0.2,
                 jpeg_compression_prob=0.3,
                 jpeg_quality_range=(60, 95)):
        """
        Initialize augmentation parameters
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_range: Range for noise level
            blur_prob: Probability of applying blur
            jpeg_compression_prob: Probability of applying JPEG compression artifacts
            jpeg_quality_range: Range for JPEG quality factor
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_range = noise_range
        self.blur_prob = blur_prob
        self.jpeg_compression_prob = jpeg_compression_prob
        self.jpeg_quality_range = jpeg_quality_range
    
    def __call__(self, img):
        """
        Apply augmentations to image
        
        Args:
            img: PIL Image or torch tensor
            
        Returns:
            Augmented image
        """
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL Image
            img = TF.to_pil_image(img)
        
        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        img = img.rotate(angle, Image.BILINEAR, expand=False)
        
        # Random brightness and contrast
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        
        # Random blur
        if random.random() < self.blur_prob:
            radius = random.uniform(0.1, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # JPEG compression artifacts
        if random.random() < self.jpeg_compression_prob:
            quality = random.randint(*self.jpeg_quality_range)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        
        # Convert to tensor
        img = transforms.ToTensor()(img)
        
        # Add noise
        if self.noise_range[1] > 0:
            noise_level = random.uniform(*self.noise_range)
            noise = torch.randn_like(img) * noise_level
            img = img + noise
            img = torch.clamp(img, 0, 1)
        
        return img


class NoisyJPEGAugmentation:
    """Simulate compression artifacts and noise"""
    
    def __init__(self, jpeg_quality_range=(50, 95), noise_level_range=(0.0, 0.05)):
        """
        Initialize the augmentation
        
        Args:
            jpeg_quality_range: Range of JPEG quality factors to use
            noise_level_range: Range of noise levels to add
        """
        self.jpeg_quality_range = jpeg_quality_range
        self.noise_level_range = noise_level_range
    
    def __call__(self, img):
        """Apply augmentation to image"""
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Apply JPEG compression
        quality = random.randint(*self.jpeg_quality_range)
        img_np = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', img_np[:, :, ::-1], encode_param)
        decoded_img = cv2.imdecode(encoded_img, 1)
        img_np = decoded_img[:, :, ::-1]
        
        # Convert back to tensor
        img = transforms.ToTensor()(img_np)
        
        # Add noise
        noise_level = random.uniform(*self.noise_level_range)
        noise = torch.randn_like(img) * noise_level
        img = img + noise
        img = torch.clamp(img, 0, 1)
        
        return img


def create_augmented_dataset(input_dir, output_dir, num_augmentations=5):
    """
    Create an augmented dataset from original images
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmented versions per original image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create augmentation pipeline
    augmentation = AdvancedAugmentation(
        rotation_range=20,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        noise_range=(0.0, 0.05),
        blur_prob=0.3,
        jpeg_compression_prob=0.4,
        jpeg_quality_range=(60, 95)
    )
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Augmenting images"):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        
        # Create augmented versions
        for i in range(num_augmentations):
            aug_img = augmentation(img)
            
            # Save augmented image
            base_name, ext = os.path.splitext(img_file)
            aug_file = f"{base_name}_aug{i}{ext}"
            TF.to_pil_image(aug_img).save(os.path.join(output_dir, aug_file))
    
    print(f"Created {len(image_files) * num_augmentations} augmented images in {output_dir}")


def load_with_custom_augmentation(path, transform=None, augmentation=None):
    """
    Load an image with optional transformation and augmentation
    
    Args:
        path: Path to image file
        transform: Optional torchvision transform to apply
        augmentation: Optional custom augmentation to apply
    
    Returns:
        Transformed and augmented image tensor
    """
    img = Image.open(path).convert('RGB')
    
    if transform:
        img = transform(img)
    
    if augmentation:
        img = augmentation(img)
    
    return img


if __name__ == "__main__":
    # Example usage
    import argparse
    from io import BytesIO
    
    parser = argparse.ArgumentParser(description='Create augmented dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with original images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for augmented images')
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmented versions per image')
    args = parser.parse_args()
    
    create_augmented_dataset(args.input_dir, args.output_dir, args.num_augmentations)