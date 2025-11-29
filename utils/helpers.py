import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns
# Some constants
DATA_ROOT = "dataset"
CLASSES = ['COVID', 'Healthy', 'Non-COVID']

# Count images per class
def get_dataset_stats(data_root: str=DATA_ROOT, classes: list[str]=CLASSES) -> dict:
    stats = {}
    for cls in classes:
        img_path = os.path.join(data_root, cls, "images")
        mask_path = os.path.join(data_root, cls, "masks")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image directory not found: {img_path}")
            img_files = []
        else:
            img_files = glob.glob(os.path.join(img_path, "*.png"))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask directory not found: {mask_path}")
            mask_files = []
        else:
            mask_files = glob.glob(os.path.join(mask_path, "*.png"))
        
        stats[cls] = {
            'images': len(img_files),
            'masks': len(mask_files),
            'image_paths': img_files[:10],  # Sample paths
            'mask_paths': mask_files[:10]
        }
    return stats

def visualize_samples(n_samples: int=6, classes: list[str]=CLASSES, data_root: str=DATA_ROOT):
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(20, 10))
    
    for row, cls in enumerate(classes):
        img_files = glob.glob(os.path.join(data_root, cls, "images", "*.png"))[:n_samples]
        
        for col, img_file in enumerate(img_files):
            img = Image.open(img_file)
            axes[row, col].imshow(img, cmap='gray' if img.mode == 'L' else None)
            axes[row, col].set_title(f'{cls}\n{os.path.basename(img_file)}', 
                                    fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def visualize_image_mask_pairs(n_samples:int=3, classes:list[str]=CLASSES, data_root:str=DATA_ROOT):
    fig, axes = plt.subplots(len(classes), n_samples*3, figsize=(20, 10))
    
    for row, cls in enumerate(classes):
        img_files = glob.glob(os.path.join(data_root, cls, "images", "*.png"))[:n_samples]
        
        for col, img_file in enumerate(img_files):
            # Load image
            img = Image.open(img_file).convert('RGB')
            img_name = os.path.basename(img_file)
            
            # Load corresponding mask
            mask_file = os.path.join(data_root, cls, "masks", img_name)
            mask = Image.open(mask_file).convert('L') if os.path.exists(mask_file) else None
            
            # Resize mask to match image dimensions
            if mask:
                mask = mask.resize(img.size, Image.NEAREST)
            
            # Original image
            axes[row, col*3].imshow(img)
            axes[row, col*3].set_title(f'{cls} - Original', fontsize=10)
            axes[row, col*3].axis('off')
            
            # Mask
            if mask:
                axes[row, col*3+1].imshow(mask, cmap='gray')
                axes[row, col*3+1].set_title('Mask', fontsize=10)
                axes[row, col*3+1].axis('off')
                
                # Overlay
                img_array = np.array(img)
                mask_array = np.array(mask) > 128
                overlay = img_array.copy()
                overlay[mask_array] = [255, 0, 0]  # Red overlay
                axes[row, col*3+2].imshow(overlay)
                axes[row, col*3+2].set_title('Overlay', fontsize=10)
                axes[row, col*3+2].axis('off')
    
    plt.tight_layout()
    plt.show()