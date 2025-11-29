import os, glob, torch, warnings, time, numpy as np
import sys
from pathlib import Path
import cv2  # Required for Albumentations border mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# FIX: Import Pipeline from the fixed design module
from utils.pipeline import Pipeline
from utils.dataset import *
from utils.helpers import *

# [CONFIG]
DATA_ROOT = "dataset"
SAVE_ROOT = "weights"
CLS_SAVE, SEG_SAVE = [os.path.join(SAVE_ROOT, p) for p in ("classification_models", "segmentation_models")]
CLASSES = ["COVID", "Healthy", "Non-COVID"]
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")  # Directory containing train.csv, val.csv, test.csv
IMG_SIZE = 256  # Fixed input image size

# [SUBSET TESTING MODE] - Set to True for quick testing with small data subset
USE_SUBSET = False  # Change to True to enable subset mode
SUBSET_SIZE = 100   # Number of samples to use per split (train/val/test) when USE_SUBSET=True

# [MAIN]
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(CLS_SAVE, exist_ok=True)
    os.makedirs(SEG_SAVE, exist_ok=True)

    # --- NEW: Define robust Augmentation and standard Validation transforms directly ---
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 1. Training Augmentation (Crucial for preventing overfitting in Stage 2)
    train_cls_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        # Pad to ensure 256x256 input while maintaining aspect ratio
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        # Geometric Augmentations (Rotation/Shifts are key for X-rays)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.HorizontalFlip(p=0.5),
        # Lightness Augmentations (X-ray images often vary in contrast/exposure)
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # 2. Validation/Test Transform (Only resize and normalize, NO augmentation)
    val_cls_transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # 3. Use Pipeline object for segmentation transform if needed
    pipeline = Pipeline()
    seg_transform = pipeline.get_seg_transform()

    # --- Data Loading using CSV splits (train/val/test) ---
    # Load classification datasets for each split
    cls_train = ClassificationDataset(DATA_ROOT, train_cls_transform, split='train')
    cls_val = ClassificationDataset(DATA_ROOT, val_cls_transform, split='val')
    cls_test = ClassificationDataset(DATA_ROOT, val_cls_transform, split='test')
    
    if len(cls_train) == 0:
        print("🛑 Classification training dataset is empty. Cannot proceed with training.")
        exit()
    
    print(f"[INFO] Classification - Train: {len(cls_train)}, Val: {len(cls_val)}, Test: {len(cls_test)}")

    # Load segmentation datasets for each split
    try:
        seg_train = SegmentationDataset(DATA_ROOT, seg_transform, split='train')
        seg_val = SegmentationDataset(DATA_ROOT, seg_transform, split='val')
        seg_test = SegmentationDataset(DATA_ROOT, seg_transform, split='test')
        
        if len(seg_train) == 0:
            print("⚠️ Segmentation dataset is empty. Skipping segmentation training.")
            seg_train, seg_val, seg_test = [], [], []
        else:
            print(f"[INFO] Segmentation - Train: {len(seg_train)}, Val: {len(seg_val)}, Test: {len(seg_test)}")
    except Exception as e:
        print(f"⚠️ Error loading segmentation dataset: {e}. Skipping segmentation training.")
        seg_train, seg_val, seg_test = [], [], []

    # --- Apply Subset for Quick Testing (if enabled) ---
    if USE_SUBSET:
        print(f"\n⚡ SUBSET MODE ENABLED: Using only {SUBSET_SIZE} samples per split for quick testing")
        
        # Apply subset to classification datasets
        cls_train = Subset(cls_train, range(min(SUBSET_SIZE, len(cls_train))))
        cls_val = Subset(cls_val, range(min(SUBSET_SIZE, len(cls_val))))
        cls_test = Subset(cls_test, range(min(SUBSET_SIZE, len(cls_test))))
        print(f"[SUBSET] Classification - Train: {len(cls_train)}, Val: {len(cls_val)}, Test: {len(cls_test)}")
        
        # Apply subset to segmentation datasets (if available)
        if seg_train and not isinstance(seg_train, list):
            seg_train = Subset(seg_train, range(min(SUBSET_SIZE, len(seg_train))))
            seg_val = Subset(seg_val, range(min(SUBSET_SIZE, len(seg_val))))
            seg_test = Subset(seg_test, range(min(SUBSET_SIZE, len(seg_test))))
            print(f"[SUBSET] Segmentation - Train: {len(seg_train)}, Val: {len(seg_val)}, Test: {len(seg_test)}")
        print()


    def make_loader(ds, bs, shuffle=True):
        if not ds or (isinstance(ds, list) and len(ds) == 0):
            return None
        # Increased num_workers to 4 for faster data loading
        return DataLoader(ds, bs, shuffle=shuffle, num_workers=4, pin_memory=True)

    # Create DataLoaders for classification
    cls_train_dl = make_loader(cls_train, 16, shuffle=True)
    cls_val_dl = make_loader(cls_val, 16, shuffle=False)
    cls_test_dl = make_loader(cls_test, 16, shuffle=False)
    
    # Create DataLoaders for segmentation
    seg_train_dl = make_loader(seg_train, 8, shuffle=True)
    seg_val_dl = make_loader(seg_val, 8, shuffle=False)
    seg_test_dl = make_loader(seg_test, 8, shuffle=False)

    models_to_train = {
        # Prioritizing ResNet50 for its superior feature extraction capability
        "classification": ["ResNet50", "ResNet18", "VGG16", "VGG19"],
        "segmentation": ["ResNetUnet", "AttentionUNet", "R2Unet", "R2AttUnet"]
    }

    results = {}

    for task, model_names in models_to_train.items():
        if task == "segmentation" and not seg_train_dl:
            print("\n--- Skipping Segmentation Training (No data found) ---")
            continue

        for name in model_names:
            print(f"\n--- Training {task} model: {name} ---")

            try:
                if task == "classification":
                    # Now calling the new, fixed model loading function
                    model, cls_head_name = get_class_model(name)
                else:
                    model = get_seg_model(name)
                    cls_head_name = None # Not applicable for segmentation
            except Exception as e:
                print(f"❌ Error loading model {name}: {e}. Skipping.")
                continue

            if task == "classification":
                train_dl, val_dl = cls_train_dl, cls_val_dl
            else:
                train_dl, val_dl = seg_train_dl, seg_val_dl

            if train_dl is None or val_dl is None:
                 print(f"🛑 Skipping training for {name}: Data loaders are not available.")
                 continue

            # Pass the classification head name only for classification tasks
            best = train(model,
                         train_dl,
                         val_dl,
                         device, epochs=20, lr=1e-6, name=name,
                         save_dir=CLS_SAVE if task=="classification" else SEG_SAVE,
                         seg=(task=="segmentation"),
                         cls_head_name=cls_head_name)
            results[name] = best

    # [Report]
    print("\n\n=============== Training Summary ===============")
    for k, v in sorted(results.items()):
        if "ResNet" in k or "VGG" in k:
            print(f"{k:<15}: {v:.2f}% Acc (Classification)")
        elif "Unet" in k:
            print(f"{k:<15}: {v:.4f} IoU (Segmentation)")

    best_cls = max([v for k, v in results.items() if "ResNet" in k or "VGG" in k], default=0)
    best_seg = min([v for k, v in results.items() if "Unet" in k], default=float('inf'))

    print("================================================")
    print(f"Best Classification Accuracy: {best_cls:.2f}%")
    print(f"Best Segmentation IoU Loss: {best_seg:.4f}")
    print("================================================")
