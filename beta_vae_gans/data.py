"""
Data loading utilities for VAE-GAN compression models
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import random
import numpy as np

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class ImageDataset(Dataset):
    """Dataset for loading images for compression tasks"""
    
    def __init__(self, root_dir, dataset_name, image_size=256, split='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            dataset_name (str): Name of the dataset ('kodak', 'clic', 'div2k')
            image_size (int): Size to which images will be resized
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.split = split
        
        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        # Load file paths based on dataset and split
        self.file_paths = self._get_file_paths()
        
    def _get_file_paths(self):
        """Get all image file paths based on dataset and split"""
        if self.dataset_name == 'kodak':
            # Kodak dataset has 24 images, typically used for testing
            paths = sorted(glob.glob(os.path.join(self.root_dir, 'kodak', '*.png')))
            if self.split == 'train':
                # For training, we can use other datasets or augment these images
                return paths
            else:
                return paths
                
        elif self.dataset_name == 'clic':
            # CLIC dataset paths
            if self.split == 'train':
                paths = glob.glob(os.path.join(self.root_dir, 'clic', 'train', '*.png'))
            elif self.split == 'val':
                paths = glob.glob(os.path.join(self.root_dir, 'clic', 'val', '*.png'))
            else:  # test
                paths = glob.glob(os.path.join(self.root_dir, 'clic', 'test', '*.png'))
            return sorted(paths)
            
        elif self.dataset_name == 'div2k':
            # DIV2K dataset paths
            if self.split == 'train':
                paths = glob.glob(os.path.join(self.root_dir, 'div2k', 'DIV2K_train_HR', '*.png'))
            elif self.split == 'val':
                paths = glob.glob(os.path.join(self.root_dir, 'div2k', 'DIV2K_valid_HR', '*.png'))
            else:  # test
                paths = glob.glob(os.path.join(self.root_dir, 'div2k', 'DIV2K_test_HR', '*.png'))
            return sorted(paths)
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataloader(config, split='train'):
    """Create data loaders for the specified dataset split"""
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Define transforms
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ])
        batch_size = config.batch_size
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
        batch_size = config.eval_batch_size
        shuffle = False
    
    # Create dataset
    dataset = ImageDataset(
        root_dir=config.data_dir,
        dataset_name=config.dataset,
        image_size=config.image_size,
        split=split,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_kodak_dataset(config):
    """Specifically get the Kodak dataset for evaluation"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(
        root_dir=config.data_dir,
        dataset_name='kodak',
        image_size=None,  # Don't resize Kodak images for proper evaluation
        split='test',
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return dataloader