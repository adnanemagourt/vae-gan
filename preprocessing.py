"""
Preprocessing utilities for image compression datasets
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
import shutil


def resize_and_crop_images(input_dir, output_dir, target_size=(256, 256), 
                          method='center_crop', output_format='png'):
    """
    Resize and crop images to target size
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        target_size: Target image size (width, height)
        method: 'center_crop', 'random_crop', or 'resize' 
        output_format: 'png' or 'jpg'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List image files
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(extensions)]
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        
        if method == 'center_crop':
            # Resize maintaining aspect ratio
            w, h = img.size
            aspect_ratio = w / h
            target_w, target_h = target_size
            
            if aspect_ratio > 1:  # width > height
                new_w = int(target_h * aspect_ratio)
                new_h = target_h
            else:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            
            img = img.resize((new_w, new_h), Image.BICUBIC)
            
            # Center crop
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            img = img.crop((left, top, right, bottom))
            
        elif method == 'random_crop':
            # Resize maintaining aspect ratio but slightly larger
            w, h = img.size
            aspect_ratio = w / h
            target_w, target_h = target_size
            
            if aspect_ratio > 1:
                new_w = int(target_h * aspect_ratio * 1.2)
                new_h = int(target_h * 1.2)
            else:
                new_w = int(target_w * 1.2)
                new_h = int(target_w / aspect_ratio * 1.2)
            
            img = img.resize((new_w, new_h), Image.BICUBIC)
            
            # Random crop
            left = random.randint(0, max(0, new_w - target_w))
            top = random.randint(0, max(0, new_h - target_h))
            right = left + target_w
            bottom = top + target_h
            img = img.crop((left, top, right, bottom))
            
        elif method == 'resize':
            # Direct resize to target size
            img = img.resize(target_size, Image.BICUBIC)
        
        # Save processed image
        base_name, _ = os.path.splitext(img_file)
        out_path = os.path.join(output_dir, f"{base_name}.{output_format}")
        
        if output_format.lower() == 'png':
            img.save(out_path, format='PNG')
        elif output_format.lower() == 'jpg':
            img.save(out_path, format='JPEG', quality=95)
        else:
            img.save(out_path)
    
    print(f"Processed {len(image_files)} images to {output_dir}")


def normalize_dataset(input_dir, output_dir, calculate_stats=True, 
                     mean=None, std=None):
    """
    Normalize dataset images and optionally calculate dataset statistics
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save normalized images
        calculate_stats: Whether to calculate dataset statistics
        mean: Manual RGB mean values (if not calculating)
        std: Manual RGB standard deviation values (if not calculating)
    
    Returns:
        mean: Dataset mean values per channel
        std: Dataset standard deviation values per channel
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if calculate_stats:
        print("Calculating dataset statistics...")
        # Initialize arrays to collect pixel values
        all_pixels = []
        
        # List image files
        extensions = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(extensions)]
        
        # Sample a subset of images for large datasets
        if len(image_files) > 100:
            random.shuffle(image_files)
            image_files = image_files[:100]
        
        # Collect pixel values
        for img_file in tqdm(image_files, desc="Calculating statistics"):
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(img)  # Convert to tensor [0,1]
            
            # Flatten and collect pixels (keep channels separate)
            pixels = img_tensor.reshape(3, -1).numpy()
            all_pixels.append(pixels)
        
        # Calculate mean and std
        all_pixels = np.hstack(all_pixels)
        mean = np.mean(all_pixels, axis=1)
        std = np.std(all_pixels, axis=1)
        
        print(f"Dataset statistics - Mean: {mean}, Std: {std}")
    
    # List all images for normalization
    extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(extensions)]
    
    # Normalize images
    for img_file in tqdm(image_files, desc="Normalizing images"):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img)  # Convert to tensor [0,1]
        
        # Normalize
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]
        
        # Save normalized image as a .npy file to preserve values
        base_name, _ = os.path.splitext(img_file)
        out_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(out_path, img_tensor.numpy())
    
    # Save normalization parameters
    stats_path = os.path.join(output_dir, "normalization_stats.npy")
    np.save(stats_path, {'mean': mean, 'std': std})
    
    print(f"Normalized {len(image_files)} images to {output_dir}")
    return mean, std


def create_train_val_test_split(image_dir, output_dir, split=(0.8, 0.1, 0.1),
                               random_seed=42):
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        image_dir: Directory containing all images
        output_dir: Base directory to save split datasets
        split: Tuple of (train_ratio, val_ratio, test_ratio)
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # List all image files
    extensions = ('.png', '.jpg', '.jpeg', '.npy')
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(extensions)]
    
    # Shuffle and split
    random.shuffle(image_files)
    train_end = int(len(image_files) * split[0])
    val_end = train_end + int(len(image_files) * split[1])
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    # Copy files to respective directories
    for files, dir_name in [
        (train_files, train_dir),
        (val_files, val_dir),
        (test_files, test_dir)
    ]:
        for file in tqdm(files, desc=f"Copying to {os.path.basename(dir_name)}"):
            src = os.path.join(image_dir, file)
            dst = os.path.join(dir_name, file)
            shutil.copy2(src, dst)
    
    print(f"Split dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")


def extract_patches(image_dir, output_dir, patch_size=128, stride=64,
                   min_variance=0.01):
    """
    Extract patches from images for training
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save extracted patches
        patch_size: Size of patches to extract
        stride: Stride between patches
        min_variance: Minimum variance for a patch to be kept (avoid blank patches)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files
    extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(extensions)]
    
    patch_count = 0
    
    for img_file in tqdm(image_files, desc="Extracting patches"):
        img_path = os.path.join(image_dir, img_file)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        h, w, _ = img.shape
        
        # Extract patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img[y:y+patch_size, x:x+patch_size]
                
                # Check patch variance (avoid blank/uniform patches)
                if np.var(patch) >= min_variance:
                    base_name, ext = os.path.splitext(img_file)
                    patch_file = f"{base_name}_patch_{patch_count}{ext}"
                    patch_path = os.path.join(output_dir, patch_file)
                    
                    Image.fromarray(patch).save(patch_path)
                    patch_count += 1
    
    print(f"Extracted {patch_count} patches to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess image dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['resize', 'normalize', 'split', 'patches'],
                       help='Preprocessing mode')
    
    # Arguments for resize
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Target size (width height)')
    parser.add_argument('--method', type=str, default='center_crop',
                       choices=['center_crop', 'random_crop', 'resize'],
                       help='Resize method')
    
    # Arguments for patches
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride between patches')
    
    args = parser.parse_args()
    
    if args.mode == 'resize':
        resize_and_crop_images(args.input_dir, args.output_dir, 
                              tuple(args.target_size), args.method)
    elif args.mode == 'normalize':
        normalize_dataset(args.input_dir, args.output_dir)
    elif args.mode == 'split':
        create_train_val_test_split(args.input_dir, args.output_dir)
    elif args.mode == 'patches':
        extract_patches(args.input_dir, args.output_dir, 
                       args.patch_size, args.stride)