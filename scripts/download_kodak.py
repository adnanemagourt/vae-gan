"""
Script to download the Kodak dataset for image compression benchmarking
"""

import os
import urllib.request
import argparse
from tqdm import tqdm


def download_kodak_dataset(output_dir):
    """
    Download the Kodak dataset (24 high-quality images)
    
    Args:
        output_dir: Directory to save the dataset
    """
    base_url = "http://r0k.us/graphics/kodak/kodim"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download 24 Kodak images
    for i in range(1, 25):
        img_url = f"{base_url}{i:02d}.png"
        output_path = os.path.join(output_dir, f"kodim{i:02d}.png")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Image {i} already exists at {output_path}, skipping")
            continue
        
        # Create a download progress bar
        print(f"Downloading image {i}/24: {img_url}")
        
        # Define a progress hook for tqdm
        def progress_hook(count, block_size, total_size):
            pbar.update(block_size)
            
        # Download with progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=None) as pbar:
            try:
                urllib.request.urlretrieve(img_url, output_path, reporthook=progress_hook)
                print(f"Downloaded to {output_path}")
            except Exception as e:
                print(f"Error downloading image {i}: {e}")
    
    print(f"Download complete. Dataset saved in {output_dir}")
    print(f"Total images: {len(os.listdir(output_dir))}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download Kodak dataset')
    parser.add_argument('--output_dir', type=str, default='./data/kodak',
                        help='Output directory for dataset')
    args = parser.parse_args()
    
    download_kodak_dataset(args.output_dir)


if __name__ == "__main__":
    main()